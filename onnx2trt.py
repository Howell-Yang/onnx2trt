# coding:utf-8

import argparse
import glob
import os
from __future__ import print_function
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # fix init error of cuda
from google.protobuf.json_format import MessageToDict
import onnx
import onnxruntime
import time
from trt_calibrator import (
    ImageBatchStream,
    TRTEntropyCalibrator,
    TRTMinMaxCalibrator,
    TRTPercentileCalibrator,
)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
    choices=["Search", "TRTEntropy", "TRTMinMax", "TRTPercentile"],
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


def create_image_stream(
    calib_dir, input_shapes, means, stds, pixel_type, channel_order
):
    calibration_files = glob.glob(os.path.join(calib_dir, "*jpg"))
    calibration_files += glob.glob(os.path.join(calib_dir, "*png"))
    channel = input_shapes[1]
    height = input_shapes[2]
    width = input_shapes[3]
    means = np.array(list(map(float, means.split(","))))
    stds = np.array(list(map(float, stds.split(","))))
    image_stream = ImageBatchStream(
        calibration_files,
        width,
        height,
        pixel_type=pixel_type,
        means=means,
        stds=stds,
        channel_order=channel_order,
    )
    return image_stream


def create_calibrator(image_stream, input_names, trt_calib_cache, calib_algo):
    CALIB_ALGO_MAP = {
        "TRTEntropy": TRTEntropyCalibrator,
        "TRTMinMax": TRTMinMaxCalibrator,
        "TRTPercentile": TRTPercentileCalibrator,
    }
    CalibratorType = CALIB_ALGO_MAP[calib_algo]
    calibrator = CalibratorType(input_names, image_stream, trt_calib_cache)
    return calibrator


def create_tensorrt_engine(onnx_path, engine_type, calibrator=None):
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # EXPLICIT_PRECISION = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)
    TRT_LOGGER = trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        (EXPLICIT_BATCH)
    ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        print("[ONNX2TRT] INFO: Beginning ONNX file parsing")
        with open(onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                print("[ONNX2TRT] ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                exit(-1)

        print("[ONNX2TRT] INFO: Network Configuration")
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            print(
                "[ONNX2TRT] INFO: Input{}".format(i),
                tensor.name,
                trt.nptype(tensor.dtype),
                tensor.shape,
            )

        config = builder.create_builder_config()
        config.max_workspace_size = 163840 << 20  # 16GB
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
            print(
                "[ONNX2TRT] WARNING: using FP32 mode directly, add fp16 precision may increate performance"
            )

        print(
            "[ONNX2TRT] INFO: Building an engine from file {}; this may take a while...".format(
                onnx_path
            )
        )
        engine = builder.build_engine(network, config)
    return engine


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    output_names = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        if not isinstance(binding, str):
            binding = engine.get_binding_name(binding)

        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            output_names.append(binding)
    return inputs, outputs, bindings, stream, output_names


def do_inference_v2(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]


def evaluate_engine(onnx_path, engine, image_stream):
    onnx_model = onnx.load_model(onnx_path)
    sess = onnxruntime.InferenceSession(onnx_model.SerializeToString())
    inputs_memo, outputs_memo, bindings, stream, trt_output_names = allocate_buffers(
        engine
    )
    context = engine.create_execution_context()

    # sess.set_providers(['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    onnx_output_names = [output.name for output in sess.get_outputs()]
    image_stream.reset()
    image_data = image_stream.next_batch()
    cos_similarity = []
    infer_time = []
    while len(image_data) > 0:
        # onnx infer
        onnx_outputs = sess.run(onnx_output_names, {input_name: image_data})
        onnx_outputs = {
            onnx_output_names[i]: onnx_outputs[i] for i in range(len(onnx_output_names))
        }

        # trt infer
        start = time.time()
        np.copyto(inputs_memo[0].host, image_data)
        trt_outputs = do_inference_v2(
            context,
            bindings=bindings,
            inputs=inputs_memo,
            outputs=outputs_memo,
            stream=stream,
        )
        trt_outputs = {
            trt_output_names[i]: trt_outputs[i] for i in range(len(trt_output_names))
        }
        end = time.time()

        cos_sims = []
        for name, value in trt_outputs.items():
            trt_output = np.reshape(value, (1, -1))
            onnx_output = np.reshape(onnx_outputs[name], (1, -1))
            cos_sim = cosine_similarity(trt_output, onnx_output)
            cos_sims.append(cos_sim)
        # cosine, runtime
        infer_time.append(end - start)
        cos_similarity.append(np.mean(cos_sims))
        image_data = image_stream.next_batch()

    return np.mean(cos_similarity), np.mean(infer_time)


# 获取输入输出信息
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
        calib_dir, INPUT_SHAPES, means, stds, pixel_type, channel_order
    )
    final_cos_similarity = -1.0
    final_engine = None
    for calibrator_type in search_types:
        calibrator = create_calibrator(
            image_stream, INPUT_NAMES, trt_calib_cache, calib_algo
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
            calib_dir, INPUT_SHAPES, means, stds, pixel_type, channel_order
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
