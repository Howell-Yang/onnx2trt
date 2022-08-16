#!/usr/bin/python3           #指定解释器
# encoding:utf-8

from ntpath import join
from posixpath import basename
import sys

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
import numpy as np
import os

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

# def read_image(path):
#     # 道路面分割模型
#     mean_val = [103.53, 116.28, 123.675]
#     std_val = [57.375, 57.12, 58.395]
#     input_size = [768, 448]

#     # img = np.random.randint(255, size=input_size + [3]).astype(np.uint8)
#     img_raw = cv2.imread(path)
#     img = cv2.resize(img_raw, (input_size[0],input_size[1])).astype(np.float32)  # BGR 图片
#     img -= mean_val
#     img /= std_val
#     img = np.transpose(img, (2, 0, 1)).astype(np.float32)
#     img = np.expand_dims(img, axis=0)

#     img = np.ascontiguousarray(img, dtype=np.float32)  # 输入就是BGR ---> 代码里做了转换，这不太对
#     # img_tensor = torch.from_numpy(img)
#     # dummy_input = torch.autograd.Variable(img_tensor)
#     return img

def read_image(path):
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
    os.path.join("/apdcephfs/private_howellyang/road_service_app/road-service/road_service/images/", "*.jpg")
)[:5]

print(calibration_files)

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


def infer_with_trt(trt_int8_path = ""):
    import tensorrt as trt
    import trt_infer
    trt.init_libnvinfer_plugins(None, "")
    samples = [convert_any_to_numpy(sample) for sample in SAMPLES]
    logger = trt.Logger(trt.Logger.INFO)
    with open(trt_int8_path, "rb") as f, trt.Runtime(
        logger
    ) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    trt_outpus_all  = []
    with engine.create_execution_context() as context:
        inputs, outputs, bindings, stream, trt_output_names = trt_infer.allocate_buffers(
            context.engine
        )
        for k, sample in enumerate(tqdm(samples, desc="TensorRT is running...")):
            # trt infer
            inputs[0].host = convert_any_to_numpy(sample)
            trt_outputs_list = trt_infer.do_inference(
                context,
                bindings=bindings,
                inputs=inputs,
                outputs=outputs,
                stream=stream,
                batch_size=1,
            )
            sample_base = os.path.basename(calibration_files[k])
            trt_outputs_dict = {
                trt_output_names[i]: trt_outputs_list[i] for i in range(len(trt_output_names))
            }
            trt_outpus_all.append(deepcopy(trt_outputs_dict))
    return trt_outpus_all


def infer_with_onnx(onnx_path = ""):

    sess = onnxruntime.InferenceSession(
        onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    onnx_output_names = [output.name for output in sess.get_outputs()]
    samples = [convert_any_to_numpy(sample) for sample in SAMPLES]

    onnx_outpus_all  = []
    for k, sample in enumerate(tqdm(samples, desc="Onnx is running...")):
        onnx_outputs = sess.run(onnx_output_names, {input_name: sample})
        onnx_outputs_dict = {
            onnx_output_names[i]: onnx_outputs[i] for i in range(len(onnx_output_names))
        }  
        onnx_outpus_all.append(deepcopy(onnx_outputs_dict))
    return onnx_outpus_all


import sys

if len(sys.argv) > 2:
    onnx_path = sys.argv[1]
    trt_path = sys.argv[2]
else:
    onnx_path = "/apdcephfs/private_howellyang/onnx2trt/Models_Fp16/RMTNet_release20220609_v2.opt.onnx"
    trt_path = "/apdcephfs/private_howellyang/onnx2trt/Models_Fp16/RMTNet_release20220609.fp16.trtmodel"

trt_outpus_all = infer_with_trt(trt_path)
onnx_outputs_all = infer_with_onnx(onnx_path)

sims = {}
diffs = {}
for i in range(len(trt_outpus_all)):
    for output_name, _ in trt_outpus_all[i].items():
        trt_output = np.reshape(trt_outpus_all[i][output_name], (1, -1))
        trt_fp32_output = np.reshape(onnx_outputs_all[i][output_name], (1, -1))
        cos_sim = cosine_similarity(trt_output, trt_fp32_output)
        abs_diff_mean = np.mean(np.abs(trt_output - trt_fp32_output))
        if output_name not in sims:
            sims[output_name] = []
            diffs[output_name] = []
        sims[output_name].append(cos_sim.ravel())
        diffs[output_name].append(abs_diff_mean.ravel())
        # if cos_sim < 0.985:
        #     print(output_name, cos_sim)
        #     print(trt_fp32_output[0, :5])
        #     print(trt_output[0, :5])

print("===================")
mean_sims = []
mean_diffs = []
for key, value in sims.items():
    print(key, np.mean(value), np.min(value), np.mean(diffs[key]), np.max(diffs[key]))
    mean_sims.append(np.mean(value))
    mean_diffs.append(np.mean(diffs[key]))
print("average cosine sim = ", np.mean(mean_sims))
print("average dff abs = ", np.mean(mean_diffs))