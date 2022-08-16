# 过滤掉calib_cache中的sigmoid层
import sys
import onnx

onnx_model = onnx.load(sys.argv[1])


# Sigmoid\HardSigmoid
sigmoid_inputs = []
sigmoid_outputs = []

add_inputs = []
add_outputs = []

for node in onnx_model.graph.node:
    if node.op_type in ["HardSigmoid", "Sigmoid"]:
        input_name = node.input[0]
        sigmoid_inputs.append(input_name)
        output_name = node.output[0]
        sigmoid_outputs.append(output_name)
    elif node.op_type in ["Mul", "Add", "Concat"]:
        input_name = node.input[0]
        add_inputs.append(input_name)
        input_name = node.input[1]
        add_inputs.append(input_name)

        output_name = node.output[0]
        add_outputs.append(output_name)

# print(sigmoid_outputs)

# 过滤Sigmoid的输出
lines = []
total_sigmoids = 0
total_nodes = 0
with open(sys.argv[2]) as fr:
    for i, line in enumerate(fr.readlines()):
        if i == 0:
            lines.append(line)
        else:
            total_nodes += 1

            name, value = line.strip().split(":")
            name = name.strip(" ")
            if name in sigmoid_outputs or name in add_outputs or name in sigmoid_inputs or name in add_inputs:
                total_sigmoids += 1
                continue
            else:
                lines.append(line)


print("total nodes", total_nodes)
print("sigmoids ", total_sigmoids)
print("final nodes", len(lines) - 1)

with open(sys.argv[3], "w") as fw:
    fw.writelines(lines)