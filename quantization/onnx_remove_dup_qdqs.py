
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

scales_map = {}
weights_map = {}
for init in inits:
    if "PPQ_Variable" in init.name:
        W = numpy_helper.to_array(init)
        scales_map[init.name] = W
    else:
        W = numpy_helper.to_array(init)
        weights_map[init.name] = W


def find_node_input_name(onnx_model, onnx_v):
    graph = onnx_model.graph
    node = graph.node
    for i in range(len(node)):
        for o in node[i].output:
            if o == onnx_v:
                return node[i].name, node[i].op_type
    return None, None

def find_dq_node_output_node(onnx_model, q_node_o):
    graph = onnx_model.graph
    node = graph.node
    for i in range(len(node)):
        for o in node[i].input:
            if o == q_node_o: # 找到dq节点
                dq_index = i
                break
    dq_o = node[dq_index].output[0]
    for i in range(len(node)):
        for j, o in enumerate(node[i].input):
            if o == dq_o: # 找到dq后置节点
                dq_o_index = i
                dq_o_index_i = j
                break
    return dq_o_index, dq_o_index_i, dq_index


# 记录每个Relu的Q/DQ的名称
relu2qde = {}
for i in range(len(onnx_model.graph.node)):
    if onnx_model.graph.node[i].op_type == "QuantizeLinear":
        assert "PPQ_Variable" in onnx_model.graph.node[i].output[0]
        # 如果输入节点是Relu
        node_input = onnx_model.graph.node[i].input[0]
        node_name, node_type = find_node_input_name(onnx_model, node_input)
        if node_type is None:
            assert node_input in inits_names, "{} - {}".format(onnx_model.graph.node[i].name, node_input)
        if node_type == "Relu":
            if node_name not in relu2qde:
                relu2qde[node_name] = []
            relu2qde[node_name].append(i)  # 这里记录的是index


# 如果有多个, 则取均值
redirect_nodes = {}
remove_nodes = []

act_scale_map = {}
for i in range(len(onnx_model.graph.node)):
    if onnx_model.graph.node[i].op_type == "Relu":
        if onnx_model.graph.node[i].name in relu2qde:
            if len(relu2qde[onnx_model.graph.node[i].name]) > 1:
                print(onnx_model.graph.node[i].name)
                qdq_indexes = relu2qde[onnx_model.graph.node[i].name]

                # 取多个scale的均值
                q_vals = []
                dq_vals = []
                q_init_names = []
                dq_init_names = []
                for idx in qdq_indexes:
                    q_node_o = onnx_model.graph.node[idx].output[0]
                    dq_o_index, dq_o_index_i, dq_index = find_dq_node_output_node(onnx_model, q_node_o)
                    q_val = scales_map[onnx_model.graph.node[idx].input[1]]
                    dq_val = scales_map[onnx_model.graph.node[dq_index].input[1]]
                    q_init_names.append(onnx_model.graph.node[idx].input[1])
                    dq_init_names.append(onnx_model.graph.node[dq_index].input[1])
                    q_vals.append(q_val)
                    dq_vals.append(dq_val)

                # 给权重重新赋值
                for idx, init in enumerate(inits):
                    if init.name in q_init_names:
                        W_new = np.mean(q_vals, axis=0)
                        tensor = numpy_helper.from_array(W_new, init.name)
                        onnx_model.graph.initializer[idx].CopyFrom(tensor)
                    elif init.name in dq_init_names:
                        W_new = np.mean(dq_vals, axis=0)
                        tensor = numpy_helper.from_array(W_new, init.name)
                        onnx_model.graph.initializer[idx].CopyFrom(tensor)


                # 修改移除后的输入输出，并记录需要移除的点
                for idx in qdq_indexes[1:]:
                    remove_nodes.append(onnx_model.graph.node[idx].name)
                    # 找到后续的dq节点
                    q_node_o = onnx_model.graph.node[idx].output[0]
                    dq_o_index, dq_o_index_i, dq_index = find_dq_node_output_node(onnx_model, q_node_o)
                    remove_nodes.append(onnx_model.graph.node[dq_index].name)                    
                    onnx_model.graph.node[dq_o_index].input[dq_o_index_i] = onnx_model.graph.node[idx].input[0]
        else:
            print("Relu wo QDQ", onnx_model.graph.node[i].name)

# 删除多余的节点
for rm_name in remove_nodes:
    for i in range(len(onnx_model.graph.node)):
        if onnx_model.graph.node[i].name == rm_name:
            old_node = onnx_model.graph.node[i]
            print("remove", old_node.name)
            onnx_model.graph.node.remove(old_node)  # 删除旧节点
            break

model_opt = onnxoptimizer.optimize(onnx_model)
# model_simp, check = simplify(model_opt)
model_simp = shape_inference.infer_shapes(model_opt)
onnx.save(model_simp, sys.argv[2])
