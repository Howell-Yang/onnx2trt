# 找出所有A-->B之间的节点
from numpy import linspace
import onnx

# A = "8305"
# B = "8527"
# E = ["8302", "8305", "8530"]

# A = "8835"
# B = "8897"

A = "8772"
B = "8834"

Final_Nodes = [A, B]

input_path = "/apdcephfs/private_howellyang/onnx2trt/model.onnx"
output_path = "/apdcephfs/private_howellyang/onnx2trt/model_{}_to_{}.onnx".format(A, B)
calib_cache = "/apdcephfs/private_howellyang/onnx2trt/model.trt_int8_with_1578pics_calib_entropy.calib_cache"
input_names = [A]
output_names = [B]

onnx.utils.extract_model(input_path, output_path, input_names, output_names)

onnx_model = onnx.load(output_path)

for node in onnx_model.graph.node:
    Final_Nodes.extend(node.output)
    # if node.op_type in ["QuantizeLinear"]:
    #     act_name = node.input[0]
    #     scale_name = node.input[1]
    #     scale_value = scales_map[scale_name]
    #     if act_name in weights_map:  # 权重量化
    #         if act_name not in weights_scale_map:
    #             weights_scale_map[act_name] = []

    #         weights_scale_map[act_name].append(scale_value)
    #     else:  # act 量化
    #         if act_name not in acts_scale_map:
    #             acts_scale_map[act_name] = []

    #         acts_scale_map[act_name].append(scale_value)
print(Final_Nodes)

lines = []
with open(calib_cache) as fr:
    for i, line in enumerate(fr.readlines()):
        if i == 0:
            lines.append(line)
        else:
            node_name, hex_value = line.strip().split(":")
            if node_name in Final_Nodes:
                continue
            else:
                lines.append(line)

with open(calib_cache + "_remove_{}_to_{}.calib_cache".format(A, B), "w") as fw:
    fw.writelines(lines)
