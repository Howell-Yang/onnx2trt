# coding:utf-8
from __future__ import print_function

import argparse
import glob
import os
from tabnanny import verbose
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # fix init error of cuda
from google.protobuf.json_format import MessageToDict
import onnx
from onnxsim import simplify
try:
    import onnxoptimizer as optimizer
except:
    from onnx import optimizer

from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
import numpy as np
from trt_utils import (
    create_image_stream,
    create_calibrator,
    create_tensorrt_engine,
    evaluate_engine,
)

parser = argparse.ArgumentParser(description="Onnx Calibration Params")
parser.add_argument("--onnx", type=str, default=None, required=True, help="原始的onnx路径")
parser.add_argument(
    "--trt_engine", type=str, default=None, required=False, help="tensorRT engine的保存路径"
)

parser.add_argument(
    "--engine_type",
    type=str,
    default="int8",
    choices=["int8", "fp32", "fp16", "best"],
    required=False,
    help="模型的计算精度",
)

parser.add_argument(
    "--trt_calib_cache",
    type=str,
    default="./trt_int8.cache",
    required=False,
    help="用来存储每个节点动态范围的路径",
)
parser.add_argument(
    "--calib_dir", type=str, default=None, required=False, help="进行精度测试以及量化校准使用的图片路径"
)
parser.add_argument(
    "--calib_algo",
    type=str,
    default=None,
    required=False,
    choices=["Search", "TRTEntropy", "TRTMinMax", "TRTPercentile", "ONNXEntropy", "ONNXMinMax", "ONNXPercentile"],
    help="""量化校准使用的算法:
    Search 进行自动化搜索, 自动选择最终输出的cosine距离最高的校准算法
    TRTEntropy 使用交叉熵评估量化前后的量化误差,自动选择误差最小的动态范围值
    TRTMinMax 计算每个节点输出的最大最小值，作为最终的动态范围值
    TRTPercentile 计算每个节点输出值，然后求其分位点作为动态范围值
""",
)

parser.add_argument(
    "--channel_order",
    type=str,
    default="RGB",
    required=False,
    choices=["RGB", "BGR"],
    help="图片的输入顺序, 可选BGR、RGB",
)
parser.add_argument(
    "--means", type=str, default="0.0,0.0,0.0", required=False, help="图片预处理的均值"
)
parser.add_argument(
    "--stds", type=str, default="1.0,1.0,1.0", required=False, help="图片预处理的方差"
)
parser.add_argument(
    "--pixel_type",
    type=str,
    default="NCHW",
    required=False,
    choices=["NCHW", "NHWC"],
    help="模型输入的通道顺序, 一般而言",
)

args = parser.parse_args()
onnx_path = args.onnx
engine_type = args.engine_type
trt_engine = args.trt_engine
calib_algo = args.calib_algo
calib_dir = args.calib_dir
means = args.means
stds = args.stds
pixel_type = args.pixel_type
trt_calib_cache = args.trt_calib_cache
channel_order = args.channel_order

# 获取输入输出信息
print("[ONNX2TRT] Optimizing Onnx Model....")
INPUT_SHAPES = []
INPUT_NAMES = []
onnx_model = onnx.load(onnx_path)
onnx_model, check = simplify(onnx_model) # simplify 
optimized_model = optimizer.optimize(onnx_model) # optimize
onnx_model = SymbolicShapeInference.infer_shapes(
    onnx_model,
    int_max=2**31 - 1,
    auto_merge=True,
    guess_output_rank=True,
    verbose=2
)

onnx_path = onnx_path.replace(".onnx", "") + "_with_shape.onnx"
onnx.save(onnx_model, onnx_path)

input_all = [node.name for node in onnx_model.graph.input]
input_initializer = [node.name for node in onnx_model.graph.initializer]
net_feed_input_names = list(set(input_all) - set(input_initializer))

for _input in onnx_model.graph.input:
    m_dict = MessageToDict(_input)
    dim_info = m_dict.get("type").get("tensorType").get("shape").get("dim")
    input_shape = [int(d.get("dimValue")) for d in dim_info]  # [4,3,384,640]
    input_name = m_dict.get("name")
    if input_name in net_feed_input_names:
        INPUT_SHAPES.append(input_shape)
        INPUT_NAMES.append(input_name)
        print(INPUT_NAMES[-1], INPUT_SHAPES[-1])

if len(INPUT_SHAPES) > 1:
    print("模型存在多个输入, 本工具暂不支持多输入模型")
    raise NameError("模型存在多个输入, 本工具暂不支持多输入模型")

elif len(INPUT_SHAPES[0]) != 4:
    print("模型的输入不是NCHW或NHWC, 本工具暂不支持这种输入格式")
    raise NameError("模型的输入不是NCHW或NHWC, 本工具暂不支持这种输入格式")

if engine_type == "int8":
    if calib_algo == "Search":
        search_types = ["TRTEntropy", "TRTMinMax", "TRTPercentile"]
    else:
        search_types = [calib_algo]
    image_stream = create_image_stream(
        calib_dir, INPUT_SHAPES[0], means, stds, pixel_type, channel_order
    )
    final_cos_similarity = -1.0
    final_engine = None
    print("[ONNX2TRT] Start Calibration with {}".format(search_types))
    for calibrator_type in search_types:
        calibrator = create_calibrator(
            image_stream, INPUT_NAMES, trt_calib_cache, calib_algo, onnx_path
        )
        engine = create_tensorrt_engine(onnx_path, engine_type, calibrator)
        cos_similarity, infer_time = evaluate_engine(onnx_path, engine, image_stream)
        if cos_similarity > final_cos_similarity:
            final_cos_similarity = cos_similarity
            final_engine = engine
            final_infer_time = infer_time
        print("[ONNX2TRT] INFO: 校准算法 = ", calib_algo)
        print("[ONNX2TRT] INFO: 与onnx输出的cos相似度 = ", cos_similarity)
        print("[ONNX2TRT] INFO: 模型infer的平均耗时 = ", infer_time)

else:
    final_engine = create_tensorrt_engine(onnx_path, engine_type)
    if calib_dir != "":
        image_stream = create_image_stream(
            calib_dir, INPUT_SHAPES[0], means, stds, pixel_type, channel_order
        )
        cos_similarity, infer_time = evaluate_engine(
            onnx_path, final_engine, image_stream
        )
        print("[ONNX2TRT] INFO: 校准算法 = ", None)
        print("[ONNX2TRT] INFO: 与onnx输出的cos相似度 = ", cos_similarity)
        print("[ONNX2TRT] INFO: 模型infer的平均耗时 = ", infer_time)

# 将trt engine写入文件
print("[ONNX2TRT] INFO: 模型构建完成, 将模型写入路径 = ", trt_engine)
if not os.path.exists(os.path.dirname(trt_engine)):
    os.makedirs(os.path.dirname(trt_engine), exist_ok=True)
with open(trt_engine, "wb") as f:
    f.write(final_engine.serialize())
