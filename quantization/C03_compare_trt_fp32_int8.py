#!/usr/bin/python3           #指定解释器
# encoding:utf-8

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

def read_image(path):
    mean_val = [103.53, 116.28, 123.675]
    std_val = [57.375, 57.12, 58.395]
    input_size = [768, 448]

    # img = np.random.randint(255, size=input_size + [3]).astype(np.uint8)
    img_raw = cv2.imread(path)
    img = cv2.resize(img_raw, (input_size[0],input_size[1])).astype(np.float32)
    img -= mean_val
    img /= std_val
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img = np.expand_dims(img, axis=0)

    img = np.ascontiguousarray(img, dtype=np.float32)
    # img_tensor = torch.from_numpy(img)
    # dummy_input = torch.autograd.Variable(img_tensor)
    return img

calibration_files = glob.glob(
    os.path.join("/apdcephfs/private_howellyang/data/Calib1k5/", "*.jpg")
)[-100:]

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
        for sample in tqdm(samples, desc="TensorRT is running..."):
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
            trt_outputs_dict = {
                trt_output_names[i]: trt_outputs_list[i] for i in range(len(trt_output_names))
            }
            trt_outpus_all.append(deepcopy(trt_outputs_dict))
    return trt_outpus_all


# trt_outpus_all = infer_with_trt("/apdcephfs/private_howellyang/road_service_app/road-service/road_service/engine/mod_road_multi_tasks/model/RMTNet_release20220609.trtmodel")
trt_outpus_all = infer_with_trt("/apdcephfs/private_howellyang/road_service_app/LaneModel/onnx_infer/model/epoch_390_mm2conv.opt.trt_int8_with_1578pics_calib_entropy.trtmodel") # 原始QAT转换的模型
# trt_outpus_all = infer_with_trt("/apdcephfs/private_howellyang/onnx2trt/model.weight_quantized.int8.trtmodel") # 进行虚拟量化转换后的模型
# trt_outpus_all = infer_with_trt("/apdcephfs/private_howellyang/onnx2trt/model.no_weight_quant.int8.trtmodel") # 不虚拟量化，仅使用min max值
# trt_outpus_all = infer_with_trt("/apdcephfs/private_howellyang/onnx2trt/model.weight_quantized_v2.trtmodel") # 进行虚拟量化
# trt_outpus_all = infer_with_trt("/apdcephfs/private_howellyang/onnx2trt/model.weight_quantized_v2.trtmodel") # 进行虚拟量化
trt_outpus_all_fp32 = infer_with_trt("/apdcephfs/private_howellyang/road_service_app/LaneModel/onnx_infer/model/epoch_390_mm2conv.opt.fp16.trtmodel")


sims = {}
for i in range(len(trt_outpus_all)):
    for output_name, _ in trt_outpus_all[i].items():
        trt_output = np.reshape(trt_outpus_all[i][output_name], (1, -1))
        trt_fp32_output = np.reshape(trt_outpus_all_fp32[i][output_name], (1, -1))
        cos_sim = cosine_similarity(trt_output, trt_fp32_output)
        if output_name not in sims:
            sims[output_name] = []
        sims[output_name].append(cos_sim.ravel())
        # if cos_sim < 0.985:
        #     print(output_name, cos_sim)
        #     print(trt_fp32_output[0, :5])
        #     print(trt_output[0, :5])

print("===================")
mean_sims = []
for key, value in sims.items():
    print(key, np.mean(value), np.min(value))
    mean_sims.append(np.mean(value))
print("average cosine sim = ", np.mean(mean_sims))