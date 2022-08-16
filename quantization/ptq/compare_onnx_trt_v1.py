#!/usr/bin/python3           #指定解释器
# encoding:utf-8

from cmath import pi
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

calibration_files = glob.glob(
    os.path.join("/mapai/howellyang/code/road-service/road_service/calib_images/", "*.jpg")
)
import random
random.seed(103600)
random.shuffle(calibration_files)
calibration_files = calibration_files[-100:]
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
    with open(trt_int8_path, "rb") as f, trt.Runtime(logger) as runtime:
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
            # for i in range(len(trt_output_names)):
            #     save_path = os.path.join("/mapai/howellyang/code/road-service/road_service/engine/mod_road_multi_tasks/outputs_trt", sample_base + "_{}.npy".format(i))
            #     np.save(save_path, trt_outputs_list[i])

            trt_outputs_dict = {
                trt_output_names[i]: trt_outputs_list[i] for i in range(len(trt_output_names))
            }
            trt_outpus_all.append(deepcopy(trt_outputs_dict))
    return trt_outpus_all


def get_post_nodes(onnx_model, tensor_name):
    post_nodes = []
    for node in onnx_model.graph.node:
        for input_tensor in node.input:
            if input_tensor == tensor_name:
                post_nodes.append(node)
                break
    return post_nodes

def infer_with_onnx(onnx_path = ""):
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
    # options = onnxruntime.SessionOptions()
    # options.intra_op_num_threads = 1
    # options.inter_op_num_threads = 1
    # sess = onnxruntime.InferenceSession(
    #     onnx._serialize(model), providers=["CUDAExecutionProvider", "CPUExecutionProvider"], #sess_options=options
    # )
    sess = onnxruntime.InferenceSession(
        onnx._serialize(model), providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    onnx_output_names = [output.name for output in sess.get_outputs()]
    samples = [convert_any_to_numpy(sample) for sample in SAMPLES]

    onnx_outpus_all  = []
    for k, sample in enumerate(tqdm(samples, desc="Onnx is running...")):
        onnx_outputs = sess.run(onnx_output_names, {input_name: sample})

        sample_base = os.path.basename(calibration_files[k])
        # for i in range(len(onnx_output_names)):
        #    save_path = os.path.join("/mapai/howellyang/code/road-service/road_service/engine/mod_road_multi_tasks/outputs_onnx", sample_base + "_{}.npy".format(i))
        #    np.save(save_path, onnx_outputs[i])

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
rel_diffs = {}
trt_medians = {}
trt_fp32_medians = {}
bias = {}
for i in range(len(trt_outpus_all)):
    for output_name, _ in trt_outpus_all[i].items():
        trt_output = np.reshape(trt_outpus_all[i][output_name], (1, -1))
        best_key = None
        best_value = None
        best_cosine = -1
        for key, value in onnx_outputs_all[i].items():
            if np.size(value) == np.size(trt_outpus_all[i][output_name]):
                onnx_output = np.reshape(value, (1, -1))
                cos_sim = cosine_similarity(trt_output, onnx_output)
                if cos_sim > best_cosine:
                    best_cosine = cos_sim
                    best_key = key
                    best_value = value

        onnx_output = np.reshape(best_value, (1, -1))
        cos_sim = cosine_similarity(trt_output, onnx_output)
        abs_diff_mean = np.mean(np.abs(trt_output - onnx_output))
        rel_diff_mean = np.mean(np.abs(trt_output - onnx_output)/(np.abs(onnx_output) + 1e-8))
        # 选择最相似的输出，作为最终的输出的
        if output_name not in sims:
            sims[output_name] = []
            diffs[output_name] = []
            rel_diffs[output_name] = []
            trt_medians[output_name] = []
            trt_fp32_medians[output_name] = []
            bias[output_name] = []

        sims[output_name].append(cos_sim.ravel())
        diffs[output_name].append(abs_diff_mean.ravel())
        rel_diffs[output_name].append(rel_diff_mean.ravel())
        trt_medians[output_name].append(np.median(trt_output))
        trt_fp32_medians[output_name].append(np.median(onnx_output))

        onnx_out = np.squeeze(best_value)
        dims = np.arange(len(np.shape(onnx_out)))
        trt_out = np.reshape(trt_outpus_all[i][output_name], np.shape(best_value))
        trt_out = np.squeeze(trt_out)
        if i == 0:
            print(np.shape(onnx_out), np.shape(trt_out), dims)
        if len(dims) > 1:
            onnx_channel_wise_mean = np.mean(onnx_out, axis=(1,2))
            trt_channel_wise_mean = np.mean(trt_out, axis=(1,2))
        else:
            onnx_channel_wise_mean = onnx_out
            trt_channel_wise_mean = trt_out

        bias[output_name].append(onnx_channel_wise_mean - trt_channel_wise_mean)

print("===================")
channel_wise_bias = {}
for name, value in bias.items():
    channel_wise_bias[name] = np.mean(value, axis=0)
    print(name, np.shape(value), np.mean(value), channel_wise_bias[name])


import pickle
import json



def scrub(x):
    import copy
    # Converts None to empty string
    ret = copy.deepcopy(x)
    # Handle dictionaries, lits & tuples. Scrub all values
    if isinstance(x, dict):
        for k, v in ret.items():
            ret[k] = scrub(v)
    elif isinstance(x, (list, tuple)):
        for k, v in enumerate(ret):
            ret[k] = scrub(v)
    elif isinstance(x, (int, str)):
        return ret
    elif isinstance(x, np.int32):
        return int(ret)
    elif isinstance(x, (np.float32, float, np.float64, np.float)):
        return float(np.round(ret, 4))
    elif x is None:
        return "None"
    elif isinstance(x, np.ndarray):
        return scrub(x.tolist())
    else:
        print(x, type(x))
    # Handle None
    if x is None:
        ret = ''
    # Finished scrubbing
    return ret


with open(trt_path + "_channel_wise_bias.pickle", "wb") as fw:
    pickle.dump(channel_wise_bias, fw)


with open(trt_path + "_channel_wise_bias.json", "w") as fw:
    json.dump(scrub(channel_wise_bias), fw, indent=4, ensure_ascii=False)


mean_sims = []
mean_diffs = []
for key, value in sims.items():
    print(key, np.mean(value), np.min(value), np.mean(diffs[key]), np.max(diffs[key]), np.mean(rel_diffs[key]), np.mean(trt_medians[key]), np.mean(trt_fp32_medians[key]))
    mean_sims.append(np.mean(value))
    mean_diffs.append(np.mean(diffs[key]))
print("average cosine sim = ", np.mean(mean_sims))
print("average dff abs = ", np.mean(mean_diffs))


# Bias Correction
sims = {}
diffs = {}
trt_medians = {}
trt_fp32_medians = {}
bias = {}
for i in range(len(trt_outpus_all)):
    for output_name, _ in trt_outpus_all[i].items():
        # 对trt输出，加上bias
        trt_outpus_all[i][output_name] = np.reshape(trt_outpus_all[i][output_name], np.shape(onnx_outputs_all[i][output_name]))
        if len(np.shape(trt_outpus_all[i][output_name])) == 2:
            trt_outputs_bias = np.reshape(channel_wise_bias[output_name], (1, -1))
        else:
            trt_outputs_bias = np.reshape(channel_wise_bias[output_name], (1, -1, 1, 1))

        if i == 0:
            print("===output shape===", np.shape(trt_outpus_all[i][output_name]))
            print("===bias shape===", np.shape(trt_outputs_bias))
        trt_outpus_all[i][output_name] += trt_outputs_bias

        trt_output = np.reshape(trt_outpus_all[i][output_name], (1, -1))
        best_key = None
        best_value = None
        best_cosine = -1
        for key, value in onnx_outputs_all[i].items():
            if np.size(value) == np.size(trt_outpus_all[i][output_name]):
                onnx_output = np.reshape(value, (1, -1))
                cos_sim = cosine_similarity(trt_output, onnx_output)
                if cos_sim > best_cosine:
                    best_cosine = cos_sim
                    best_key = key
                    best_value = value

        onnx_output = np.reshape(best_value, (1, -1))
        cos_sim = cosine_similarity(trt_output, onnx_output)
        abs_diff_mean = np.mean(np.abs(trt_output - onnx_output))
        # 选择最相似的输出，作为最终的输出的
        if output_name not in sims:
            sims[output_name] = []
            diffs[output_name] = []
            trt_medians[output_name] = []
            trt_fp32_medians[output_name] = []
            bias[output_name] = []

        sims[output_name].append(cos_sim.ravel())
        diffs[output_name].append(abs_diff_mean.ravel())
        trt_medians[output_name].append(np.median(trt_output))
        trt_fp32_medians[output_name].append(np.median(onnx_output))

        onnx_out = np.squeeze(best_value)
        dims = np.arange(len(np.shape(onnx_out)))
        trt_out = np.reshape(trt_outpus_all[i][output_name], np.shape(best_value))
        trt_out = np.squeeze(trt_out)
        if i == 0:
            print("onnx vs trt shape", np.shape(onnx_out), np.shape(trt_out), dims)
        if len(dims) > 1:
            onnx_channel_wise_mean = np.mean(onnx_out, axis=(1,2))
            trt_channel_wise_mean = np.mean(trt_out, axis=(1,2))
        else:
            onnx_channel_wise_mean = onnx_out
            trt_channel_wise_mean = trt_out

        bias[output_name].append(onnx_channel_wise_mean - trt_channel_wise_mean)

print("===================")
channel_wise_bias = {}
for name, value in bias.items():
    channel_wise_bias[name] = np.mean(value, axis=0)
    print(name, np.shape(value), np.mean(value), channel_wise_bias[name])

mean_sims = []
mean_diffs = []
for key, value in sims.items():
    print(key, np.mean(value), np.min(value), np.mean(diffs[key]), np.max(diffs[key]), np.mean(trt_medians[key]), np.mean(trt_fp32_medians[key]))
    mean_sims.append(np.mean(value))
    mean_diffs.append(np.mean(diffs[key]))
print("average cosine sim = ", np.mean(mean_sims))
print("average dff abs = ", np.mean(mean_diffs))


# 1. 直接使用最新生成的模型，评测一个指标
# 2. 进行bias correction，评测一个指标