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


# def infer_with_trt(trt_int8_path = ""):
#     import tensorrt as trt
#     import trt_infer

#     samples = [convert_any_to_numpy(sample) for sample in SAMPLES]
#     logger = trt.Logger(trt.Logger.INFO)
#     with open(trt_int8_path, "rb") as f, trt.Runtime(
#         logger
#     ) as runtime:
#         engine = runtime.deserialize_cuda_engine(f.read())

#     trt_outpus_all  = []
#     with engine.create_execution_context() as context:
#         inputs, outputs, bindings, stream, trt_output_names = trt_infer.allocate_buffers(
#             context.engine
#         )
#         for sample in tqdm(samples, desc="TensorRT is running..."):
#             # trt infer
#             inputs[0].host = convert_any_to_numpy(sample)
#             trt_outputs_list = trt_infer.do_inference(
#                 context,
#                 bindings=bindings,
#                 inputs=inputs,
#                 outputs=outputs,
#                 stream=stream,
#                 batch_size=1,
#             )
#             trt_outputs_dict = {
#                 trt_output_names[i]: trt_outputs_list[i] for i in range(len(trt_output_names))
#             }
#             trt_outpus_all.append(deepcopy(trt_outputs_dict))
#     return trt_outpus_all


def infer_with_onnx(onnx_path = ""):
    # options = onnxruntime.SessionOptions()
    # options.intra_op_num_threads = 1
    # options.inter_op_num_threads = 1
    # sess = onnxruntime.InferenceSession(
    #     onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"] #, sess_options=options
    # )
    sess = onnxruntime.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"] #, sess_options=options
    )
    input_name = sess.get_inputs()[0].name
    onnx_output_names = [output.name for output in sess.get_outputs()]
    samples = [convert_any_to_numpy(sample) for sample in SAMPLES]

    onnx_outpus_all  = []
    for sample in tqdm(samples, desc="Onnx is running..."):
        onnx_outputs = sess.run(onnx_output_names, {input_name: sample})
        onnx_outputs_dict = {
            onnx_output_names[i]: onnx_outputs[i] for i in range(len(onnx_output_names))
        }
        onnx_outpus_all.append(deepcopy(onnx_outputs_dict))
    return onnx_outpus_all


def infer_with_onnx_extra_resize(onnx_path = ""):
    import torch.nn.functional as F
    sess = onnxruntime.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    onnx_output_names = [output.name for output in sess.get_outputs()]
    samples = [convert_any_to_numpy(sample) for sample in SAMPLES]

    onnx_outpus_all  = []
    for sample in tqdm(samples, desc="Onnx is running..."):
        onnx_outputs = sess.run(onnx_output_names, {input_name: sample})
        input_size = (384, 768)

        onnx_outputs_dict = {}
        for i in range(len(onnx_output_names)):
            if onnx_output_names[i] not in '''
                # edge_seg_logits = F.interpolate(edge_seg_logits, input_size, mode='bilinear', align_corners=False)
                # edge_ins_logits = F.interpolate(edge_ins_logits, input_size, mode='bilinear', align_corners=False)
                # edge_dense_seg_logits = F.interpolate(edge_dense_seg_logits, input_size, mode='bilinear', align_corners=False)
                # quality_seg_logits = F.interpolate(quality_seg_logits, input_size, mode='bilinear', align_corners=False)
                # sepzone_seg_logits = F.interpolate(sepzone_seg_logits, input_size, mode='bilinear', align_corners=False)
                # road_seg_logits = F.interpolate(road_seg_logits, input_size, mode='bilinear', align_corners=False)
                # lanes_seg_logits = F.interpolate(lanes_seg_logits, input_size, mode='bilinear', align_corners=False)
                # cur_road_seg_logits = F.interpolate(cur_road_seg_logits, input_size, mode='bilinear', align_corners=False)
                # passable_area_seg_logits = F.interpolate(passable_area_seg_logits, input_size, mode='bilinear',
                #                                          align_corners=False)''':
                onnx_outputs_dict[onnx_output_names[i]] = onnx_outputs[i]
            else:
                onnx_outputs_dict[onnx_output_names[i]] = F.interpolate(torch.from_numpy(onnx_outputs[i]), input_size, mode='bilinear', align_corners=False).numpy()
        onnx_outpus_all.append(deepcopy(onnx_outputs_dict))
    return onnx_outpus_all

import sys

if len(sys.argv) > 2:
    onnx_path = sys.argv[1]
    trt_path = sys.argv[2]
else:
    onnx_path = "/apdcephfs/private_howellyang/onnx2trt/Models_Fp16/RMTNet_release20220609_v2.opt.onnx"
    trt_path = "/apdcephfs/private_howellyang/onnx2trt/model.weight_quantized_v2.onnx"

trt_outpus_all = infer_with_onnx(trt_path)
onnx_outputs_all = infer_with_onnx(onnx_path)

sims = {}
diffs = {}
trt_medians = {}
trt_fp32_medians = {}
for i in range(len(trt_outpus_all)):
    for output_name, _ in trt_outpus_all[i].items():
        trt_output = np.reshape(trt_outpus_all[i][output_name], (1, -1))

        best_key = None
        best_value = None
        best_cosine = -1
        for key, value in onnx_outputs_all[i].items():
            if np.size(value) == np.size(trt_outpus_all[i][output_name]):
                trt_fp32_output = np.reshape(value, (1, -1))
                cos_sim = cosine_similarity(trt_output, trt_fp32_output)
                if cos_sim > best_cosine:
                    best_cosine = cos_sim
                    best_key = key
                    best_value = value

        trt_fp32_output = np.reshape(best_value, (1, -1))
        cos_sim = cosine_similarity(trt_output, trt_fp32_output)
        abs_diff_mean = np.mean(np.abs(trt_output - trt_fp32_output))
        # 选择最相似的输出，作为最终的输出的
        if output_name not in sims:
            sims[output_name] = []
            diffs[output_name] = []
            trt_medians[output_name] = []
            trt_fp32_medians[output_name] = []
        sims[output_name].append(cos_sim.ravel())
        diffs[output_name].append(abs_diff_mean.ravel())
        trt_medians[output_name].append(np.median(trt_output))
        trt_fp32_medians[output_name].append(np.median(trt_fp32_output))

        # if cos_sim < 0.985:
        #     print(output_name, cos_sim)
        #     print(trt_fp32_output[0, :5])
        #     print(trt_output[0, :5])

print("===================")
mean_sims = []
mean_diffs = []
for key, value in sims.items():
    print(key, np.mean(value), np.min(value), np.mean(diffs[key]), np.max(diffs[key]), np.mean(trt_medians[key]), np.mean(trt_fp32_medians[key]))
    mean_sims.append(np.mean(value))
    mean_diffs.append(np.mean(diffs[key]))
print("average cosine sim = ", np.mean(mean_sims))
print("average dff abs = ", np.mean(mean_diffs))
