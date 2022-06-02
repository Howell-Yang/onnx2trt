#coding:utf-8

import argparse
import glob
import os
from __future__ import print_function
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # fix init error of cuda
form google.protobuf.json_format import MessageToDict
import onnx
from trt_calibrator import ImageBatchStream, TRTEntropyCalibrator, TRTMinMaxCalibrator, TRTPercentileCalibrator
import numpy as np

parser = argparse.ArgumentParser(description='Onnx Calibration Params')
parser.add_argument('--onnx', type = str, default=None, required = True, help='原始的onnx路径')
parser.add_argument('--trt_engine', type = str, default=None, required = False, help='tensorRT engine的保存路径')

parser.add_argument('--engine_type', type = str, default="int8", choices= ["int8", "fp32", "fp16", "best"],required = False, help='模型的计算精度')

parser.add_argument('--trt_calib_cache', type = str, default= "./trt_int8.cache", required = False, help='用来存储每个节点动态范围的路径')
parser.add_argument('--calib_dir', type = str, default=None,required = False, help='量化校准使用的图片路径')
parser.add_argument('--calibrator_type', type = str, default=None,required = False, choices = ["MinMax", "Entropy", "Percentile"], help='量化校准使用的算法')

parser.add_argument('--channel_order', type = str, default="RGB",required = False, choices = ["RGB", "BGR"], help='图片的输入顺序, 可选BGR、RGB')
parser.add_argument('--means', type = str, default="0.0,0.0,0.0", required = False, help='图片预处理的均值')
parser.add_argument('--stds', type = str, default="1.0,1.0,1.0", required = False, help='图片预处理的方差')
parser.add_argument('--pixel_type', type = str, default="NCHW",required = False, choices = ["NCHW", "NHWC"], help='模型输入的通道顺序, 一般而言')

args = parser.parse_args()


onnx_path = args.onnx
engine_type = args.engine_type
trt_engine = args.trt_engine





# step 1. 在输入部分添加卷积层，实现RGB2BGR以及减均值、除方差的操作
INPUT_SHAPES = [] 
INPUT_NAMES = []
onnx_model = onnx.load(onnx_path)
for _input in onnx_model.graph.input:
    m_dict = MessageToDict(_input)
    dim_info = m_dict.get("type").get("tensorType").get("shape").get("dim")
    input_shape = [d.get("dimValue") for d in dim_info]  # [4,3,384,640]
    INPUT_SHAPES.append(input_shape)
    INPUT_NAMES.append(m_dict.get("name"))

if len(INPUT_SHAPES) > 1:
    print("模型存在多个输入, 本工具暂不支持多输入模型")
    exit(-1)
elif len(INPUT_SHAPES[0]) != 4:
    print("模型的输入不是NCHW或NHWC, 本工具暂不支持这种输入格式")
    exit(-1) 


# TODO

# step 2. 使用onnx/trt的quant工具，来统计每个节点的动态范围: _min, _max
calibrator = None

if engine_type in ["int8", "best"]:
    calibration_dir = args.calib_dir
    calibration_files = glob.glob(os.path.join(calibration_dir, "*jpg"))
    calibration_files += glob.glob(os.path.join(calibration_dir, "*png"))

    CHANNEL = INPUT_SHAPES[1]
    HEIGHT = INPUT_SHAPES[2]
    WIDTH = INPUT_SHAPES[3]
    means = args.means
    means = np.array(list(map(float, means.split(","))))
    stds = args.stds
    stds = np.array(list(map(float, stds.split(","))))
    image_stream = ImageBatchStream(calibration_files, WIDTH, HEIGHT, pixel_type=args.pixel_type, means = means, stds = stds, channel_order=args.channel_order)
    calibrator = TRTEntropyCalibrator(INPUT_NAMES, image_stream, args.trt_calib_cache)
    # 生成calib_cache: 当存在calib cache时，后续的calibrator会自动加载已有的calib_cache,而不是重新校准


# step 3. 使用tensorRT的转换工具，完成模型的转换
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
# EXPLICIT_PRECISION = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)
TRT_LOGGER = trt.Logger()
with trt.Builder(TRT_LOGGER) as builder, builder.create_network((EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    print('3.1 Beginning ONNX file parsing')
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit(-1)

    print('3.2 Network Configuration')
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        print("Input{}".format(i), tensor.name, trt.nptype(tensor.dtype), tensor.shape)

    config = builder.create_builder_config()
    config.max_workspace_size = 163840 << 20 # 16GB
    config.avg_timing_iterations = 5

    if engine_type == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        builder.fp16_mode = True
    elif engine_type == "int8":
        assert calibrator is not None, "with int8 mode, calibrator must be set"
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator
    elif engine_type == "best":
        assert calibrator is not None, "with int8 mode, calibrator must be set" 
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator
    else:
        print("[WARNING] using FP32 mode directly, add fp16 precision may increate performance")

    print('3.3 Building an engine from file {}; this may take a while...'.format(onnx_path))
    engine = builder.build_engine(network, config)

    print("3.4 Completed creating Engine. Writing file to: {}".format(trt_engine))
    if not os.path.exists(os.path.dirname(trt_engine)):
        os.makedirs(os.path.dirname(trt_engine), exist_ok=True)
    with open(trt_engine, "wb") as f:
        f.write(engine.serialize())

# step 4. 进行速度和精度测试
# means = (0.485, 0.456, 0.406)
# scales = (0.229, 0.224, 0.225)
