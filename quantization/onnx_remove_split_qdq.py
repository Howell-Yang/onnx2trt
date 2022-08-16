
import onnx
import sys
import onnxoptimizer
from onnx import helper, shape_inference
from onnxsim import simplify
from onnx import numpy_helper
import numpy as np

onnx_model = onnx.load(sys.argv[1])
inits = onnx_model.graph.initializer
inits_names =[init.name for init in inits] + ["input.1"]

nodes_names_to_remove = []
for i in range(len(onnx_model.graph.node)):
    if onnx_model.graph.node[i].op_type == "Split":
        if "DequantizeLinear" in onnx_model.graph.node[i].input[0]: # 输入是DQ
            # 找到DQ
            for j in range(len(onnx_model.graph.node)):
                if onnx_model.graph.node[j].output[0] == onnx_model.graph.node[i].input[0]:
                    nodes_names_to_remove.append(onnx_model.graph.node[j].name)
                    dq_input = onnx_model.graph.node[j].input[0]
                    break

            # 找到Q
            for j in range(len(onnx_model.graph.node)):
                if onnx_model.graph.node[j].output[0] == dq_input:
                    nodes_names_to_remove.append(onnx_model.graph.node[j].name)
                    q_input = onnx_model.graph.node[j].input[0]
                    break

            # 改变Split的输入
            onnx_model.graph.node[i].input[0] = q_input


def remove_initializer_from_input(model):
    if model.ir_version < 4:
        print(
            "Model with ir_version below 4 requires to include initilizer in graph input"
        )
        return

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    return model

# 删除多余的节点
for rm_name in nodes_names_to_remove:
    for i in range(len(onnx_model.graph.node)):
        if onnx_model.graph.node[i].name == rm_name:
            old_node = onnx_model.graph.node[i]
            print("remove", old_node.name)
            onnx_model.graph.node.remove(old_node)  # 删除旧节点
            break

onnx_model = remove_initializer_from_input(onnx_model)
model_opt = onnxoptimizer.optimize(onnx_model)
# model_simp, check = simplify(model_opt)
model_simp = shape_inference.infer_shapes(model_opt)
onnx.save(model_simp, sys.argv[2])
