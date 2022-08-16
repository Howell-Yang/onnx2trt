

'''
算法思路：
针对低精度模型中，部分channel的权重过大或者过小的问题，进行权重的balance

'''

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


def get_post_nodes(onnx_model, tensor_name):
    post_nodes = []
    for node in onnx_model.graph.node:
        for input_tensor in node.input:
            if input_tensor == tensor_name:
                post_nodes.append(node)
                break
    return post_nodes


def get_all_fp32_outputs_means(fp32_model):
    # remove origin outputs
    model = deepcopy(fp32_model)

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

    original_outputs = [output for output in model.graph.output]
    for o in original_outputs:
        model.graph.output.remove(o)

    # add all output to graph output
    output_names = []
    for node in model.graph.node:
        for o in node.output:
            intermediate_layer_value_info = onnx.helper.ValueInfoProto()
            intermediate_layer_value_info.name = o
            model.graph.output.append(intermediate_layer_value_info)
            output_names.append(o)

    # run onnx infer
    options = onnxruntime.SessionOptions()
    # options.intra_op_num_threads = 1
    # options.inter_op_num_threads = 1
    # sess = onnxruntime.InferenceSession(
    #     onnx._serialize(model),
    #     providers=["CUDAExecutionProvider", "CPUExecutionProvider"], sess_options=options)
    sess = onnxruntime.InferenceSession(
        onnx._serialize(model),
        providers=["CPUExecutionProvider"])

    input_name = sess.get_inputs()[0].name
    samples = [convert_any_to_numpy(sample) for sample in input_data]

    fp32_outputs_means = {output_name: [] for output_name in output_names}
    count = 0
    for sample in tqdm(samples):
        count += 1
        onnx_outputs = sess.run(output_names, {input_name: sample })
        for i, output_name in enumerate(output_names):
            onnx_out = np.squeeze(onnx_outputs[i]) # 1, C, H , W
            dims = np.shape(onnx_out)
            if len(dims) > 1:
                onnx_channel_wise_mean = np.median(np.abs(onnx_out), axis=(1,2))
            else:
                onnx_channel_wise_mean = np.abs(onnx_out)

            if count == 1:
                fp32_outputs_means[output_name] = onnx_channel_wise_mean
            else:
                fp32_outputs_means[output_name] = (count - 1)/count * fp32_outputs_means[output_name] + onnx_channel_wise_mean/count

    return fp32_outputs_means

def balance_model(model):
    weight_name2tensor = {}
    for weight in model.graph.initializer:
        weight_name2tensor[weight.name] = weight

    # fp32_outputs_means = get_all_fp32_outputs_means(model)

    balances_convs = {}
    for node in model.graph.node:
        if node.op_type in ["Conv"]:
            for attr in node.attribute:
                if (attr.name == "group"):
                    group_num = attr.i
            # print(node.name, group_num)
            if group_num > 1:
                continue
            weight_name = node.input[1]
            if weight_name not in weight_name2tensor:
                continue
            # 获取与当前节点同级的卷积，以及下一层级的卷积
            cur_convs, next_convs = get_next_convs(node.output[0], model)
            if next_convs is None:
                # print(node.name, "is not balancable")
                continue

            if len(cur_convs) > 0: # Concat结构
                # print("=" * 50)
                # print(node.name, [n.name for n in cur_convs])
                # print([n.name for n in next_convs])
                assert len(next_convs) == 1, "next conv nodes should only be one"
            else: # Conv结构
                assert len(next_convs) == 1, "next conv nodes should only be one"
                assert len(cur_convs) == 0
                # print("+" * 50)
                # print(node.name, [n.name for n in cur_convs])
                # print([n.name for n in next_convs])
                cur_convs.append(node)
            # for debug 
            balances_convs[node.name] = (cur_convs, next_convs)

            if len(cur_convs) == 1:  # Conv层的平衡
                # c(ax + b) = ca ^ ( cs ^ + b / cs^)
                assert len(next_convs) == 1
                cur_node = cur_convs[0]
                next_node = next_convs[0]

                weight_name = cur_node.input[1]
                cur_weight_tensor = weight_name2tensor[weight_name]
                cur_weight = numpy_helper.to_array(cur_weight_tensor) # out_ch, in_ch, ker, ker

                weight_name = next_node.input[1]
                next_weight_tensor = weight_name2tensor[weight_name]
                next_weight = numpy_helper.to_array(next_weight_tensor) # out_ch, in_ch, ker, ker


                # 第一步输出的均值平衡(而不是权重的均值，权重的per channel量化损失较少)
                # out_ch_mean = fp32_outputs_means[cur_node.output[0]]
                # print("===out_ch_mean===", out_ch_mean)
                # 第二步,权重的整体平衡
                out_ch_weight_mean = np.median(np.abs(cur_weight), axis=(1,2,3)) # 关注当前节点的输出
                in_ch_weight_mean = np.median(np.abs(next_weight), axis=(0,2,3)) # 下一个节点的输入
                balanced_mean = np.sqrt(out_ch_weight_mean * in_ch_weight_mean)
                out_ch_weight_scale_vector = balanced_mean/out_ch_weight_mean
                in_ch_weight_scale_vector = balanced_mean/in_ch_weight_mean

                # out_ch_scale_vector = np.mean(out_ch_mean)/out_ch_mean * out_ch_weight_scale_vector
                # in_ch_scale_vector = out_ch_mean/np.mean(out_ch_mean) * in_ch_weight_scale_vector
                out_ch_scale_vector = out_ch_weight_scale_vector  #  np.mean(out_ch_mean)/out_ch_mean
                in_ch_scale_vector = in_ch_weight_scale_vector  #  out_ch_mean/np.mean(out_ch_mean)

                # if np.sum(out_ch_scale_vector < 1e-7) > 0 or np.sum(in_ch_scale_vector < 1e-7) > 0:
                #     out_ch_scale_vector = np.ones_like(out_ch_scale_vector)
                #     in_ch_scale_vector = np.ones_like(in_ch_scale_vector)
                print("===out_ch_scale_vector===", out_ch_scale_vector)

                # print(node.name, "===pre===>", np.min(out_ch_weight_mean), np.min(np.median(np.abs(next_weight), axis=(1,2,3))))
                balanced_cur_weight = cur_weight * np.expand_dims(out_ch_scale_vector, axis=(1,2,3))
                raw_shape = tuple([i for i in cur_weight_tensor.dims])
                new_shape = np.shape(balanced_cur_weight)
                assert new_shape == raw_shape
                cur_weight_tensor.ClearField("float_data")
                cur_weight_tensor.ClearField("int32_data")
                cur_weight_tensor.ClearField("int64_data")
                cur_weight_tensor.raw_data = balanced_cur_weight.tobytes()
                if len(cur_node.input) == 3:
                    bias_name = cur_node.input[2]
                    bias_tensor = weight_name2tensor[bias_name]
                    cur_bias = numpy_helper.to_array(bias_tensor)
                    balanced_cur_bias = cur_bias * out_ch_scale_vector

                    raw_shape = tuple([i for i in bias_tensor.dims])
                    new_shape = np.shape(balanced_cur_bias)
                    assert new_shape == raw_shape
                    bias_tensor.ClearField("float_data")
                    bias_tensor.ClearField("int32_data")
                    bias_tensor.ClearField("int64_data")
                    bias_tensor.raw_data = balanced_cur_bias.tobytes()
                in_ch_scale_vector = np.expand_dims(in_ch_scale_vector, axis=(0,2,3))
                balanced_next_weight = next_weight * in_ch_scale_vector
                raw_shape = tuple([i for i in next_weight_tensor.dims])
                new_shape = np.shape(balanced_next_weight)
                assert new_shape == raw_shape, "{}, {}".format(raw_shape, new_shape)
                next_weight_tensor.ClearField("float_data")
                next_weight_tensor.ClearField("int32_data")
                next_weight_tensor.ClearField("int64_data")
                next_weight_tensor.raw_data = balanced_next_weight.tobytes()
                weight_name = cur_node.input[1]
                cur_weight_tensor = weight_name2tensor[weight_name]
                cur_weight = numpy_helper.to_array(cur_weight_tensor) # out_ch, in_ch, ker, ker

                weight_name = next_node.input[1]
                next_weight_tensor = weight_name2tensor[weight_name]
                next_weight = numpy_helper.to_array(next_weight_tensor) # out_ch, in_ch, ker, ker

                out_ch_mean = np.median(np.abs(cur_weight), axis=(1,2,3)) # 关注当前节点的输出
                in_ch_mean = np.median(np.abs(next_weight), axis=(0,2,3)) # 下一个节点的输入
                # print(node.name, "==updated==>", np.min(out_ch_mean), np.min(np.median(np.abs(next_weight), axis=(1,2,3))))
            else:  # Concat层的平衡
                # Concat前的输入，需要进行concat从而进行平衡
                assert len(next_convs) == 1
                next_node = next_convs[0]
                weight_name = next_node.input[1]
                next_weight_tensor = weight_name2tensor[weight_name]
                next_weight = numpy_helper.to_array(next_weight_tensor) # out_ch, in_ch, ker, ker
                start_ch = 0
                end_ch = 0
                in_ch_scale_vectors = []
                for cur_node in cur_convs:
                    weight_name = cur_node.input[1]
                    cur_weight_tensor = weight_name2tensor[weight_name]
                    cur_weight = numpy_helper.to_array(cur_weight_tensor) # out_ch, in_ch, ker, ker
                    # 第一步输出的均值平衡(而不是权重的均值，权重的per channel量化损失较少)
                    # out_ch_mean = fp32_outputs_means[cur_node.output[0]]

                    # 第二步,权重的整体平衡
                    start_ch = end_ch
                    end_ch = start_ch + np.shape(cur_weight)[0]
                    out_ch_weight_mean = np.median(np.abs(cur_weight), axis=(1,2,3)) # 关注当前节点的输出
                    in_ch_weight_mean = np.median(np.abs(next_weight), axis=(0,2,3))[start_ch:end_ch] # 下一个节点的输入
                    balanced_mean = np.sqrt(out_ch_weight_mean * in_ch_weight_mean)
                    out_ch_weight_scale_vector = balanced_mean/out_ch_weight_mean
                    in_ch_weight_scale_vector = balanced_mean/in_ch_weight_mean

                    # out_ch_weight_mean = np.mean(np.abs(cur_weight))
                    # in_ch_weight_mean = np.mean(np.abs(next_weight))
                    # balanced_weight_mean = np.sqrt(out_ch_weight_mean * in_ch_weight_mean)
                    # print("===balanced_weight_mean===", balanced_weight_mean)

                    # out_ch_scale_vector = np.mean(out_ch_mean)/out_ch_mean #  * balanced_weight_mean/out_ch_weight_mean
                    # in_ch_scale_vector = 1.0/out_ch_scale_vector

                    # out_ch_scale_vector = np.mean(out_ch_mean)/out_ch_mean * out_ch_weight_scale_vector
                    # in_ch_scale_vector = out_ch_mean/np.mean(out_ch_mean) * in_ch_weight_scale_vector
                    out_ch_scale_vector = out_ch_weight_scale_vector  #  np.mean(out_ch_mean)/out_ch_mean
                    in_ch_scale_vector = in_ch_weight_scale_vector  #  out_ch_mean/np.mean(out_ch_mean)

                    # if np.sum(out_ch_scale_vector < 1e-7) > 0 or np.sum(in_ch_scale_vector < 1e-7) > 0:
                    #     out_ch_scale_vector = np.ones_like(out_ch_scale_vector)
                    #     in_ch_scale_vector = np.ones_like(in_ch_scale_vector)
                    print("===out_ch_scale_vector===", out_ch_scale_vector)
                    # # c(ax + b) = ca ^ ( cs ^ + b / cs^)
                    # out_ch_mean = np.median(np.abs(cur_weight), axis=(1,2,3)) # 关注当前节点的输出
                    # # print(start_ch, end_ch, np.shape(cur_weight), np.shape(next_weight))
                    # in_ch_mean = np.median(np.abs(next_weight), axis=(0,2,3))[start_ch:end_ch] # 下一个节点的输入
                    # balanced_mean = np.sqrt(out_ch_mean * in_ch_mean)

                    # out_ch_scale_vector = balanced_mean/out_ch_mean
                    # in_ch_scale_vector = balanced_mean/in_ch_mean
                    # print(node.name, "====", np.min(out_ch_mean), np.min(in_ch_mean), np.min(balanced_mean))

                    # out_ch_scale_vector = np.mean(out_ch_mean)/out_ch_mean # 全部归一化到out_ch_mean 
                    # in_ch_scale_vector = 1.0/out_ch_scale_vector
                    # print(node.name, "===concat pre===>", np.mean(out_ch_weight_mean), np.mean(np.median(np.abs(next_weight), axis=(1,2,3))))

                    # print(out_ch_scale_vector)
                    # print(in_ch_scale_vector)
                    in_ch_scale_vectors.append(in_ch_scale_vector)
                    # 修改当前层权重
                    balanced_cur_weight = cur_weight * np.expand_dims(out_ch_scale_vector, axis=(1,2,3))
                    raw_shape = tuple([i for i in cur_weight_tensor.dims])
                    new_shape = np.shape(balanced_cur_weight)
                    assert new_shape == raw_shape
                    cur_weight_tensor.ClearField("float_data")
                    cur_weight_tensor.ClearField("int32_data")
                    cur_weight_tensor.ClearField("int64_data")
                    cur_weight_tensor.raw_data = balanced_cur_weight.tobytes()

                    cur_weight_tensor = weight_name2tensor[weight_name]
                    cur_weight = numpy_helper.to_array(cur_weight_tensor)
                    # print(node.name, "===concat updated===>", np.min(np.median(np.abs(cur_weight), axis=(1,2,3)) ))

                    if len(cur_node.input) == 3:            
                        bias_name = cur_node.input[2]
                        bias_tensor = weight_name2tensor[bias_name]
                        cur_bias = numpy_helper.to_array(bias_tensor)
                        balanced_cur_bias = cur_bias * out_ch_scale_vector

                        raw_shape = tuple([i for i in bias_tensor.dims])
                        new_shape = np.shape(balanced_cur_bias)
                        assert new_shape == raw_shape
                        bias_tensor.ClearField("float_data")
                        bias_tensor.ClearField("int32_data")
                        bias_tensor.ClearField("int64_data")
                        bias_tensor.raw_data = balanced_cur_bias.tobytes()

                # 下一层的权重
                in_ch_scale_vector = np.concatenate(in_ch_scale_vectors, axis=0)
                balanced_next_weight = next_weight * np.expand_dims(in_ch_scale_vector, axis=(0,2,3))
                raw_shape = tuple([i for i in next_weight_tensor.dims])
                new_shape = np.shape(balanced_next_weight)
                assert new_shape == raw_shape, "{}, {}".format(raw_shape, new_shape)
                next_weight_tensor.ClearField("float_data")
                next_weight_tensor.ClearField("int32_data")
                next_weight_tensor.ClearField("int64_data")
                next_weight_tensor.raw_data = balanced_next_weight.tobytes()
    return model

# print("*" * 100)
# for node_name, balances_nodes in  balances_convs.items():
#     cur_convs, next_convs = balances_nodes
#     if len(cur_convs) > 0:
#         print(node_name, [n.name for n in cur_convs], [n.name for n in next_convs]) 


# BUG: Concat附近的节点会被多次balance
if len(sys.argv) < 2:
    model = onnx.load("/Users/howellyang/Projects/15_GPU_Utils_Opt/RMTNet_release20220609_mm2conv.optimized.onnx")
else:
    model = onnx.load(sys.argv[1])



for i in range(10):
    model = balance_model(model)
    model_opt = onnxoptimizer.optimize(model)
    # model_simp, check = simplify(model_opt)
    model_opt = shape_inference.infer_shapes(model_opt)
    model = deepcopy(model_opt)

if len(sys.argv) < 3:
    onnx.save(model_opt, "/Users/howellyang/Projects/15_GPU_Utils_Opt/RMTNet_release20220609_mm2conv.optimized.balanced.onnx")
else:
    onnx.save(model_opt, sys.argv[2])
