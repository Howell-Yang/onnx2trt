from lib2to3.pytree import Node
import onnx
from onnx.external_data_helper import load_external_data_for_model



# 移除已训练好的QDQ 多任务模型中的特定分支的QDQ节点

# Step 01 找到从中心concat到输出的节点
# input_path = '/mapai/howellyang/code/onnx2trt/RMTNet_release20220609_mm2conv.optimized.onnx'
output_path = '/mapai/howellyang/code/onnx2trt/RMTNet_release20220609_mm2conv.optimized.train_hy.edge_seg_logits_head.onnx'
# input_names = ['onnx::Conv_9157', 'onnx::GlobalAveragePool_9176'] # ['input_0', 'input_1', 'input_2']
# output_names = ['edge_seg_logits'] # ['output_0', 'output_1']

# onnx.utils.extract_model(input_path, output_path, input_names, output_names)

# Step 02 找到这些节点中的输入和输出节点

# 移除特定的QDQ量化节点
sub_model_inouts = ["onnx::Mul_8478", "onnx::Add_8479", "input.4700", "input.4568", "input.4560", "x.239", \
    "input.439", "onnx::Add_11845", "onnx::Add_11836", "onnx::Add_11827", "input.4124", "onnx::Sigmoid_8131", \
        "onnx::Conv_8130", "input.4048", "input.4036", "x.123", "input.3524", ""]

# sub_model = onnx.load(output_path)
# for node in sub_model.graph.node:
#     for i in node.input:
#         sub_model_inouts.append(i)
#     for o in node.output:
#         sub_model_inouts.append(o)

# Step 03 移除这些QDQ节点，并修改输入输出
full_graph = onnx.load("/mapai/howellyang/code/onnx2trt/RMTNet_release20220609_mm2conv.optimized.train_hy.onnx")


nodes_to_remove = []
for node in full_graph.graph.node:
    if node.op_type == "QuantizeLinear":
        if node.input[0] in sub_model_inouts:
            nodes_to_remove.append(node)
            for dq_node in full_graph.graph.node:
                if dq_node.input[0] == node.output[0]:
                    # DQ节点
                    nodes_to_remove.append(dq_node)
                    for post_node in full_graph.graph.node:
                        for i, input in enumerate(post_node.input):
                            if input == dq_node.output[0]:
                                post_node.input[i] = node.input[0]

for node in nodes_to_remove:
    full_graph.graph.node.remove(node)

onnx.save(full_graph, "/mapai/howellyang/code/onnx2trt/RMTNet_release20220609_mm2conv.optimized.train_hy.edge_seg_logits_head_fp32.onnx")