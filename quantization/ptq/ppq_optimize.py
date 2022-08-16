import onnx
from onnx import numpy_helper
import numpy as np
import json
import sys


def get_post_nodes(onnx_model, tensor_name):
    post_nodes = []
    for node in onnx_model.graph.node:
        for input_tensor in node.input:
            if input_tensor == tensor_name:
                post_nodes.append(node)
                break
    return post_nodes


def remove_qdq(onnx_model, node):
    nodes_to_remove = []
    assert node.op_type == "QuantizeLinear"
    nodes_to_remove.append(node)
    for dq_node in onnx_model.graph.node:
        if dq_node.input[0] == node.output[0]:
            assert dq_node.op_type == "DequantizeLinear"
            nodes_to_remove.append(dq_node)
            for post_node in onnx_model.graph.node:
                for i, input in enumerate(post_node.input):
                    if input == dq_node.output[0]:
                        post_node.input[i] = node.input[0]
    return nodes_to_remove


def create_act_initializer_tensor(
    name,
    tensor_array,
    data_type=onnx.TensorProto.FLOAT,
):

    # (TensorProto)
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=(),  # [1],
        vals=[tensor_array],
    )

    return initializer_tensor


def add_act_dqd_node(qdq_model, tensor_name, scale):
    flag_found = False
    for node in qdq_model.graph.node:
        for j in range(len(node.input)):
            if node.input[j] == tensor_name:
                flag_found = True
    if not flag_found:
        return None

    quant_node_name = tensor_name + "_QuantizeLinear"
    dequant_node_name = tensor_name + "_DequantizeLinear"
    q_input = tensor_name
    q_output = tensor_name + "_QuantizeLinear"
    dq_input = q_output
    dq_output = tensor_name + "_DequantizeLinear"

    scale_name = tensor_name + "_QuantizeScale"
    zp_name = tensor_name + "_QuantizeZp"
    qlinear_node = onnx.helper.make_node(
        "QuantizeLinear",
        [q_input, scale_name, zp_name],
        [q_output],
        quant_node_name,
    )
    dequant_node = onnx.helper.make_node(
        "DequantizeLinear",
        [dq_input, scale_name, zp_name],
        [dq_output],
        dequant_node_name,
    )

    for node in qdq_model.graph.node:
        for j in range(len(node.input)):
            if node.input[j] == tensor_name:
                node.input[j] = dq_output

    qdq_model.graph.node.extend([qlinear_node, dequant_node])

    scale_initializer_tensor = create_act_initializer_tensor(
        name=scale_name, tensor_array=scale, data_type=onnx.TensorProto.FLOAT
    )

    zp_initializer_tensor = create_act_initializer_tensor(
        name=zp_name, tensor_array=0, data_type=onnx.TensorProto.INT8
    )

    qdq_model.graph.initializer.append(scale_initializer_tensor)
    qdq_model.graph.initializer.append(zp_initializer_tensor)
    return qdq_model

# Step 01. Move QDQ forward
int8_model_path = sys.argv[1] # onnx.load("/mapai/howellyang/code/onnx2trt/RMTNet_release20220609_mm2conv.optimized.rm_inits.onnx.model_int8.onnx")
int8_model = onnx.load(int8_model_path)
weight_name2tensor = {}
for weight in int8_model.graph.initializer:
    weight_name2tensor[weight.name] = weight

nodes_to_remove = []
scale_map = {}
scale_map_final = {}
for node in int8_model.graph.node:
  output_tensor = node.output[0]
  post_nodes = get_post_nodes(int8_model, output_tensor)

  QDQ_count = 0
  for post_node in post_nodes:
    if post_node.op_type in ["QuantizeLinear"]:
      QDQ_count += 1

  # 第一种情况: 存在QDQ, 但是与后续节点个数不同
  # 第二种情况: 存在多于1个QDQ
  if node.op_type not in ["Concat"] and (QDQ_count > 0 and QDQ_count != len(post_nodes)) or QDQ_count > 1:
    scale_values = []
    for post_node in post_nodes:
      if post_node.op_type in ["QuantizeLinear"]:
        scale_name = post_node.input[1]
        scale_tensor = weight_name2tensor[scale_name]
        scale_value = numpy_helper.to_array(scale_tensor) # out_ch, in_ch, ker, ker
        scale_values.append(float(scale_value.ravel()))
        nodes_to_remove.extend(remove_qdq(int8_model, post_node))
    print(node.name, QDQ_count, len(post_nodes), scale_values)
    scale_map[node.output[0]] = np.mean(scale_values)
  elif QDQ_count == 1 and len(post_nodes) == 1:
    scale_values = []
    for post_node in post_nodes:
      if post_node.op_type in ["QuantizeLinear"]:
        scale_name = post_node.input[1]
        scale_tensor = weight_name2tensor[scale_name]
        scale_value = numpy_helper.to_array(scale_tensor) # out_ch, in_ch, ker, ker
        scale_values.append(float(scale_value.ravel()))
      assert len(scale_values) == 1
      scale_map_final[node.output[0]] = np.mean(scale_values)

for node in nodes_to_remove:
  int8_model.graph.node.remove(node)

for tensor_name, scale in scale_map.items():
  add_act_dqd_node(int8_model, tensor_name, scale)

onnx.save(int8_model, int8_model_path + ".opt_step1.onnx")

scale_map_final.update(scale_map)
with open(int8_model_path + ".opt_step1.scale_map.json", "w") as fw:
  json.dump(scale_map_final, fw, indent=4)


# Step 02 add QDQ node to model
def read_calib_cache(calib_cache):
    import struct
    scale_map = {}
    with open(calib_cache) as fr:
        for line in fr.readlines()[1:]:
            print(line.strip())
            name, value = line.strip().split(": ")
            name = name.strip(":")
            value = value.strip(":")
            if value.strip() == "0":
                val = 0.0
            else:
                val = struct.unpack("!f", bytes.fromhex(value.strip()))[0]
            scale_map[name] = val
    scale_map = {k: scale_map[k] for k in sorted(scale_map)}
    return scale_map


# int8_model = onnx.load(
#     "/mapai/howellyang/code/onnx2trt/RMTNet_release20220609_mm2conv.optimized.rm_inits.onnx.model_int8.opt_step1.onnx")

weight_name2tensor = {}
for weight in int8_model.graph.initializer:
    weight_name2tensor[weight.name] = weight

QDQ_scales = {}
for node in int8_model.graph.node:
    if node.op_type in ["QuantizeLinear"]:
        scale_name = node.input[1]
        scale_tensor = weight_name2tensor[scale_name]
        scale_value = numpy_helper.to_array(
            scale_tensor)  # out_ch, in_ch, ker, ker
        if np.size(scale_value) > 1:
            continue
        scale_value = float(scale_value.ravel())
        QDQ_scales[node.input[0]] = scale_value

calib_cache = "/mapai/howellyang/code/onnx2trt/RMTNet_release20220609_mm2conv.optimized.rm_inits.trt_int8_with_1687pics_calib_percentile595.calib_cache"
full_scale_map = read_calib_cache(calib_cache)

for tensor_name, scale in full_scale_map.items():
    if tensor_name in QDQ_scales:
        print(tensor_name, scale, QDQ_scales[tensor_name])
    else:
        print(tensor_name, "not exist")
        scale = max(scale, 1e-8)
        add_act_dqd_node(int8_model, tensor_name, scale)


onnx.save(int8_model, int8_model_path + ".opt_step2.onnx")

# print(QDQ_scales)
# print(full_scale_map)
