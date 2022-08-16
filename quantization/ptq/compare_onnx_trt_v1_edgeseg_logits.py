#!/usr/bin/python3           #指定解释器
# encoding:utf-8
import sys
from cmath import pi
import onnxruntime
import torch
from tqdm import tqdm
import glob
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity
import onnx
from copy import deepcopy
import random
import json
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

dev = cuda.Device(0)
ctx = dev.make_context()


print(sys.getdefaultencoding())
s = "中文乱码问题解决"
print(s)

# ---------------------------------------------------------------
# 这个脚本向你展示了如何使用 tensorRT 对 PPQ 导出的模型进行推理

# This script shows you how to export ppq internal graph to tensorRT
# ---------------------------------------------------------------

# For this inference test, all test data is randomly picked.
# If you want to use real data, just rewrite the defination of SAMPLES
print("开始import")


def convert_any_to_numpy(x, accepet_none: bool = True) -> np.ndarray:
    if x is None and not accepet_none:
        raise ValueError("Trying to convert an empty value.")
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, int) or isinstance(x, float):
        return np.array(
            [
                x,
            ]
        )
    elif isinstance(x, torch.Tensor):
        if x.numel() == 0 and accepet_none:
            return None
        if x.numel() == 0 and not accepet_none:
            raise ValueError("Trying to convert an empty value.")
        if x.numel() == 1:
            return convert_any_to_numpy(x.detach().cpu().item())
        if x.numel() > 1:
            return x.detach().cpu().numpy()
    elif isinstance(x, list) or isinstance(x, tuple):
        return np.array(x)
    else:
        raise TypeError(
            f"input value {x}({type(x)}) can not be converted as numpy type."
        )


def read_image_v1(path):
    # 多任务模型
    _img_transforms = transforms.Compose(
        [
            transforms.Resize((384, 768)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    img = Image.open(path).convert("RGB")
    img_w, img_h = img.size[0], img.size[1]
    img = _img_transforms(img)
    img = img.unsqueeze(0)
    return img


def read_image_v2(path):
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    input_w = 960
    input_h = 480
    mean = np.array(mean)
    std = np.array(std)
    img = cv2.imread(path)
    img = cv2.resize(img, (input_w, input_h))
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Norm
    for i in range(3):
        img[..., i] = (img[..., i] - mean[i]) / std[i]

    # hwc -> nchw ----> 这里输入方式不对
    # h, w, c = img.shape
    # img = img.reshape((1, c, h ,w))
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return np.ascontiguousarray(img, dtype=np.float32)


calibration_files = glob.glob(
    os.path.join(
        "/mapai/howellyang/code/road-service/road_service/calib_images/", "*.jpg")
)
# random.seed(103600)
random.seed(103601)
random.shuffle(calibration_files)
calibration_files = calibration_files[:100]

if len(sys.argv) >= 4:
    read_image = eval("read_image_{}".format(sys.argv[3]))
else:
    read_image = read_image_v1

SAMPLES = [
    read_image(path) for path in calibration_files
]  # rewirte this to use real data.


DEVICE = "cuda"
FINETUNE = True
EXECUTING_DEVICE = "cuda"
REQUIRE_ANALYSE = True

# -------------------------------------------------------------------
# 启动 tensorRT 进行推理，你先装一下 trt
# -------------------------------------------------------------------


def infer_with_trt(trt_int8_path=""):
    import tensorrt as trt
    import trt_infer
    trt.init_libnvinfer_plugins(None, "")
    samples = [convert_any_to_numpy(sample) for sample in SAMPLES]
    logger = trt.Logger(trt.Logger.INFO)
    with open(trt_int8_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    trt_outpus_all = {}
    with engine.create_execution_context() as context:
        inputs, outputs, bindings, stream, trt_output_names = trt_infer.allocate_buffers(
            context.engine
        )
        for k, sample in enumerate(tqdm(samples, desc="TensorRT is running...")):
            # trt infer
            inputs[0].host = convert_any_to_numpy(sample)
            trt_outputs_list = trt_infer.do_inference_v2(
                context,
                bindings=bindings,
                inputs=inputs,
                outputs=outputs,
                stream=stream,
            )
            for i in range(len(trt_output_names)):
                if trt_output_names[i] not in trt_outpus_all:
                    trt_outpus_all[trt_output_names[i]] = []
                trt_outpus_all[trt_output_names[i]].append(deepcopy(trt_outputs_list[i]))
    return trt_outpus_all


def get_post_nodes(onnx_model, tensor_name):
    post_nodes = []
    for node in onnx_model.graph.node:
        for input_tensor in node.input:
            if input_tensor == tensor_name:
                post_nodes.append(node)
                break
    return post_nodes


def infer_with_onnx(onnx_path=""):
    model = onnx.load(onnx_path)
    # ununsed constance nodes
    all_nodes = []
    for node in model.graph.node:
        all_nodes.append(node)

    input_name_to_nodes = {}
    for node in all_nodes:
        for input_name in node.input:
            if input_name not in input_name_to_nodes:
                input_name_to_nodes[input_name] = [node]
            else:
                input_name_to_nodes[input_name].append(node)

    unused_nodes = []
    for node in all_nodes:
        if node.op_type == "Constant" and node.output[0] not in input_name_to_nodes:
            unused_nodes.append(node)

    for node in unused_nodes:
        if node in model.graph.node:
            model.graph.node.remove(node)

    # remove inits in inputs
    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    unused_initializers = []
    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

        post_nodes = get_post_nodes(model, initializer.name)
        if len(post_nodes) == 0:
            unused_initializers.append(initializer)

    for initializer in unused_initializers:
        model.graph.initializer.remove(initializer)
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    sess = onnxruntime.InferenceSession(
        onnx._serialize(model), providers=["CUDAExecutionProvider", "CPUExecutionProvider"], sess_options=options
    )
    # sess = onnxruntime.InferenceSession(
    #     onnx._serialize(model), providers=["CPUExecutionProvider"]
    # )
    input_name = sess.get_inputs()[0].name
    onnx_output_names = [output.name for output in sess.get_outputs()]
    samples = [convert_any_to_numpy(sample) for sample in SAMPLES]
    # "input.1": np.array(read_image_v2(path))

    onnx_outpus_all = {}
    for k, sample in enumerate(tqdm(samples, desc="Onnx is running...")):
        onnx_outputs = sess.run(onnx_output_names, {"input.1": sample})
        for i in range(len(onnx_output_names)):
            if onnx_output_names[i] not in onnx_outpus_all:
                onnx_outpus_all[onnx_output_names[i]] = []
            onnx_outpus_all[onnx_output_names[i]].append(deepcopy(onnx_outputs[i]))
    return onnx_outpus_all


def eval_quantize(fp32_output, int8_output):
    sims = []
    diffs = []
    rel_diffs = []
    bias_diffs = []
    snrs = []
    # for fp32, int8 in zip(fp32_output, int8_output):
    for n in range(np.shape(fp32_output)[0]):
        fp32 = fp32_output[n, ...]
        int8 = int8_output[n, ...]
        fp32 = np.reshape(fp32, (1, -1))
        int8 = np.reshape(int8, (1, -1))
        sim = cosine_similarity(fp32, int8)
        diff = np.abs(fp32 - int8)
        rel_diff = diff / (np.abs(fp32) + 0.1)
        sims.append(sim)
        diffs.append(np.mean(diff))
        rel_diffs.append(np.mean(rel_diff))
        # judge if the mean noise is zero centered
        bias_diffs.append(np.mean(fp32 - int8))

        noise_power = np.sum(np.power(fp32 - int8, 2), axis=-1)
        signal_power = np.sum(np.power(fp32, 2), axis=-1)
        snr = (noise_power) / (signal_power + 1e-7)
        snrs.append(snr)

    return np.mean(sims), np.mean(diffs), np.mean(rel_diffs), np.mean(bias_diffs), np.mean(snrs)


if len(sys.argv) > 2:
    onnx_path = sys.argv[1]
    trt_path = sys.argv[2]
else:
    onnx_path = "/apdcephfs/private_howellyang/onnx2trt/Models_Fp16/RMTNet_release20220609_v2.opt.onnx"
    trt_path = "/apdcephfs/private_howellyang/onnx2trt/Models_Fp16/RMTNet_release20220609.fp16.trtmodel"


output_shapes = None

if trt_path.endswith(".onnx"):
    trt_outpus_all = infer_with_onnx(trt_path)
    output_shapes = {}
    for onnx_name, onnx_value in trt_outpus_all.items():
        output_shapes[onnx_name] = np.shape(onnx_value)
else:
    trt_outpus_all = infer_with_trt(trt_path)


if onnx_path.endswith(".onnx"):
    onnx_outputs_all = infer_with_onnx(onnx_path)
    output_shapes = {}
    for onnx_name, onnx_value in onnx_outputs_all.items():
        output_shapes[onnx_name] = np.shape(onnx_value)
else:
    onnx_outputs_all = infer_with_trt(onnx_path)

# 进行channel wise的结果对比: 每个channel的cosine和diff
if output_shapes is None and os.path.exists("onnx_output_shapes.json"):
    with open("onnx_output_shapes.json", "r") as fr:
        output_shapes = json.load(fr)

# else:
#     output_shapes = {}
#     for onnx_name, onnx_value in onnx_outputs_all.items():
#         output_shapes[onnx_name] = np.shape(onnx_value)
#     with open("onnx_output_shapes.json", "w") as fw:
#         json.dump(output_shapes, fw)

for trt_name, trt_value in trt_outpus_all.items():
    onnx_value = onnx_outputs_all[trt_name]  # (N, 1, C, H, W)
    # onnx_value = np.reshape(
    #     onnx_value, [-1] + list(output_shapes[trt_name][1:]))
    trt_value = [np.reshape(t, np.shape(o)) for t, o in zip(trt_value, onnx_value)]
    onnx_value = np.array(onnx_value)
    onnx_value = onnx_value.squeeze()
    if len(np.shape(onnx_value)) < 3:
        onnx_value = np.expand_dims(onnx_value, axis=1)

    trt_value = np.array(trt_value)
    trt_value = trt_value.squeeze()

    # 问题在于这里的reshape N C H W
    trt_value = np.reshape(trt_value, np.shape(onnx_value))

    print("=" * 20, trt_name, ":", np.shape(onnx_value), "="*20)
    for channel in range(np.shape(onnx_value)[1]):
        # compare channel
        trt_out = trt_value[:, channel, ...]
        onnx_out = onnx_value[:, channel, ...]
        cos_sim, abs_diff, rel_diff, bias_diff, snr = eval_quantize(
            onnx_out, trt_out)
        print("{}-{}: cos={:6.2f}, abs_diff={:6.2f}, rel_diff={:6.2f}, bias_diff={:6.2f}, snr={:6.2f}".format(
            trt_name, channel, cos_sim * 100.0, abs_diff, rel_diff, bias_diff, snr))

ctx.pop()
