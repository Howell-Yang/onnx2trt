import onnx
from onnx import numpy_helper
import sys
import numpy as np
from requests import post
import onnx
import onnx
from onnx import helper, shape_inference
from onnx import TensorProto
from onnx.tools import update_model_dims
import onnxoptimizer
from onnxsim import simplify
import sys
from copy import deepcopy
import onnxruntime
from tqdm import tqdm

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
)
import random
random.seed(103600)
random.shuffle(calibration_files)
calibration_files = calibration_files[:100]
input_data = [convert_any_to_numpy(read_image(path))
              for path in calibration_files]

def get_pre_convs(input_tensor, model):
    pre_nodes = []
    for node in model.graph.node:
        for output in node.output:
            if output == input_tensor:
                pre_nodes.append(node)

    if len(pre_nodes) > 1 or len(pre_nodes) < 1:
        # print("too many/too few output nodes", input_tensor)
        return None

    pre_node = pre_nodes[0]
    if pre_node.op_type in ["Conv"]:
        for attr in pre_node.attribute:
            if (attr.name == "group"):
                group_num = attr.i
        if group_num > 1:
            return None
        return pre_node

    elif pre_node.op_type in ["Relu", "Resize"]:
        return get_pre_convs(pre_node.input[0], model)
    else:
        # print("not supported pre tyoe", pre_node.name, pre_node.op_type)
        return None

def get_next_convs(output_tensor, model):
    post_nodes = []
    for node in model.graph.node:
        for input in node.input:
            if input == output_tensor:
                post_nodes.append(node)

    if len(post_nodes) > 1 or len(post_nodes) < 1:
        # print("too many/too few output nodes", output_tensor)
        return None, None

    post_node = post_nodes[0]
    if post_node.op_type in ["Conv"]:
        for attr in post_node.attribute:
            if (attr.name == "group"):
                group_num = attr.i
        if group_num > 1:
            return None, None
        return [], [post_node]

    elif post_node.op_type in ["Relu", "Resize"]:
        return get_next_convs(post_node.output[0], model)
    elif post_node.op_type in ["Concat"]:
        # 根据输入，前向获取卷积
        pre_convs = []
        for input in post_node.input:
            pre_conv = get_pre_convs(input, model)
            if pre_conv is None:
                return None, None
            assert pre_conv.op_type == "Conv"
            pre_convs.append(pre_conv)

        if pre_convs is None:
            return None, None
        # 根据输出，后向获取卷积
        cur_convs, next_convs = get_next_convs(post_node.output[0], model)
        assert len(cur_convs) == 0, "Cocat cur convs should be empty"
        return pre_convs, next_convs
    else:
        # print("not supported post type", post_node.name, post_node.op_type)
        return None, None


# bias correction of output layer(only output layer)
import json
with open("/mapai/howellyang/code/road-service/road_service/engine/mod_road_multi_tasks/model-bak/RMTNet_release20220609.trt_int8_with_531pics_calib_percentile595.trtmodel_channel_wise_bias.json") as fr:
    channel_wise_bias = json.load(fr)


output2weight = {
    "lane_lljx": "lane_lljx_classifier.clasifier.3.bias",
    "road_scene_understanding_logits": "road_scene_understanding_classifier.clasifier.3.bias",
    "scene":"scene_classifier.bmx_classifier.1.bias",
    "bmx_lljx": "bmx_classifier.bmx_classifier.1.bias",
    "edge_seg_logits": "edge_seg_decoder.final_project.6.bias",
    "edge_ins_logits": "edge_ins_decoder.final_project.6.bias",
    "edge_dense_seg_logits":"edge_dense_decoder.final_project.6.bias",
    "edge_quality_seg_logits": "edge_quality_decoder.final_project.6.bias",
    "sepzone_seg_logits":"sepzone_seg_decoder.final_project.6.bias",
    "road_seg": "road_seg_decoder.final_project.6.bias",
    "lane_seg":"lanes_seg_decoder.final_project.6.bias",
    "cur_road_seg_logits": "cur_road_decoder.final_project.6.bias",
    "passable_area_seg_logits":"passable_area_seg_decoder.final_project.6.bias"
}



model = onnx.load(sys.argv[1])

weight_name2tensor = {}
for weight in model.graph.initializer:
    weight_name2tensor[weight.name] = weight


for output_name, bias_name in output2weight.items():
    bias_correction = np.array(channel_wise_bias[output_name])
    bias_tensor = weight_name2tensor[bias_name]
    bias_weight = numpy_helper.to_array(bias_tensor) # out_ch, in_ch, ker, ker
    print(bias_name, np.shape(bias_weight), np.shape(bias_correction))
    new_bias_weight = bias_weight + bias_correction
    new_bias_weight = new_bias_weight.astype(np.float32)
    print(bias_weight, bias_correction, new_bias_weight)
    bias_tensor.ClearField("float_data")
    bias_tensor.ClearField("int32_data")
    bias_tensor.ClearField("int64_data")
    bias_tensor.raw_data = new_bias_weight.tobytes()
    # print(bias_tensor)


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



def get_post_nodes(onnx_model, tensor_name):
    post_nodes = []
    for node in onnx_model.graph.node:
        for input_tensor in node.input:
            if input_tensor == tensor_name:
                post_nodes.append(node)
                break
    return post_nodes


unused_initializers = []
for initializer in model.graph.initializer:
    if initializer.name in name_to_input:
        inputs.remove(name_to_input[initializer.name])

    post_nodes = get_post_nodes(model, initializer.name)
    if len(post_nodes) == 0:
        unused_initializers.append(initializer)

for initializer in unused_initializers:
    model.graph.initializer.remove(initializer)


model_opt = onnxoptimizer.optimize(model)
model_opt, check = simplify(model_opt)
model_opt = shape_inference.infer_shapes(model_opt)
onnx.save(model_opt, sys.argv[2])
