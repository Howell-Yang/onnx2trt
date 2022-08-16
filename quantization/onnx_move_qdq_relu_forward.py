# 对于relu节点的qdq,将其移动到qdq前面

import onnx
import sys
import onnxoptimizer
from onnx import helper, shape_inference
from onnxsim import simplify
from onnx import numpy_helper


def find_input_node_index(onnx_model, node_input):
    for i in range(len(onnx_model.graph.node)):
        if onnx_model.graph.node[i].output[0] == node_input:
            return i
    return None


def find_output_node_index(onnx_model, node_output):
    for i in range(len(onnx_model.graph.node)):
        if onnx_model.graph.node[i].input[0] == node_output:
            return i
    return None


def find_pre_relu(onnx_model, node_input):
    for i in range(len(onnx_model.graph.node)):
        for j in range(len(onnx_model.graph.node[i].output)):
            if onnx_model.graph.node[i].output[j] == node_input:
                if onnx_model.graph.node[i].op_type in [
                    "QuantizeLinear",
                    "DequantizeLinear",
                ]:
                    raise NameError("Relu的前置节点是QDQ-{}".format(node_input))
                return i, j
    return None, None


def find_post_dq(onnx_model, node_output):
    node_indexes = []
    node_indexes_i = []
    for i in range(len(onnx_model.graph.node)):
        for j in range(len(onnx_model.graph.node[i].input)):
            if onnx_model.graph.node[i].input[j] == node_output:
                node_indexes.append(i)
                node_indexes_i.append(j)
    return node_indexes, node_indexes_i


def find_post_relu(onnx_model, node_output, q_index):
    node_indexes = []
    node_indexes_i = []
    for i in range(len(onnx_model.graph.node)):
        if i == q_index:
            continue
        for j in range(len(onnx_model.graph.node[i].input)):
            if onnx_model.graph.node[i].input[j] == node_output:
                node_indexes.append(i)
                node_indexes_i.append(j)
    return node_indexes, node_indexes_i


# 将Relu节点后的QDQ移动到Relu前面

onnx_model = onnx.load(sys.argv[1])
inits = onnx_model.graph.initializer
init_names = [init.name for init in inits]  + ["input.1"]

scales_map = {}
weights_map = {}
for init in inits:
    if "PPQ_Variable" in init.name:
        W = numpy_helper.to_array(init)
        scales_map[init.name] = W
    else:
        W = numpy_helper.to_array(init)
        weights_map[init.name] = W

relu2qde = {}
for i in range(len(onnx_model.graph.node)):
    if onnx_model.graph.node[i].op_type == "QuantizeLinear":
        assert "PPQ_Variable" in onnx_model.graph.node[i].output[0]
        # 首先，当前节点的输入节点是否是Relu
        node_input = onnx_model.graph.node[i].input[0]
        input_node_index = find_input_node_index(onnx_model, node_input)
        if input_node_index is None:
            assert node_input in init_names, "input is not found in model - {}".format(node_input)
            continue

        node_output = onnx_model.graph.node[i].output[0]
        output_node_index = find_output_node_index(onnx_model, node_output)

        if onnx_model.graph.node[input_node_index].op_type == "Relu":
            # relu及其 qdq节点
            q_index = i
            # q_node = onnx_model.graph.node[q_index]
            relu_index = input_node_index
            # relu_node = onnx_model.graph.node[relu_index]
            dq_index = output_node_index
            # dq_node = onnx_model.graph.node[relu_index]

            # relu的前置节点
            pre_relu_index, pre_relu_index_i = find_pre_relu(
                onnx_model, onnx_model.graph.node[relu_index].input[0]
            )

            # dq的后置节点(可能有多个)
            post_dq_indexes, post_dq_indexes_i = find_post_dq(
                onnx_model, onnx_model.graph.node[dq_index].output[0]
            )

            # relu的后置节点(可能有多个)
            post_relu_indexes, post_relu_indexes_i = find_post_relu(
                onnx_model, onnx_model.graph.node[relu_index].output[0], q_index
            )

            # 首先q节点的输入，改为relu前置节点的输出
            onnx_model.graph.node[q_index].input[0] = onnx_model.graph.node[
                pre_relu_index
            ].output[pre_relu_index_i]

            # 然后，relu节点的输入，改为dq节点的输出
            onnx_model.graph.node[relu_index].input[0] = onnx_model.graph.node[
                dq_index
            ].output[0]

            # 再然后，dp的后置节点的输入，改为relu节点的输出
            for idx, idx_i in zip(post_dq_indexes, post_dq_indexes_i):
                onnx_model.graph.node[idx].input[idx_i] = onnx_model.graph.node[
                    relu_index
                ].output[0]

            # 最后，其它relu后置节点的输入，仍然是relu
            # for idx, idx_i in zip(post_relu_indexes, post_relu_indexes_i):
            #     onnx_model.graph.node[idx].input[idx_i] = onnx_model.graph.node[dq_index].output[0]

model_opt = onnxoptimizer.optimize(onnx_model)
# model_simp, check = simplify(model_opt)
model_simp = shape_inference.infer_shapes(model_opt)
onnx.save(model_simp, sys.argv[2])
