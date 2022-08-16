from unicodedata import name
import onnx
from onnx import numpy_helper
import numpy as np
from torch import init_num_threads
import json
import struct
import sys


from sklearn.metrics.pairwise import cosine_similarity

fp32_model = "/apdcephfs/private_howellyang/onnx2trt/model_T01/model.onnx"
int8_qat_model = sys.argv[1] # "/apdcephfs/private_howellyang/onnx2trt/model_T01/model.move_relu_qdq_forword.onnx"
onnx_updated_model_path = int8_qat_model.replace(".onnx", ".rm_dqd.weight_quantized.onnx")


# 第一步, 从int8-qat模型中取出所有的zero points和scales
print("[Step1] read scales values from model ")
onnx_model = onnx.load(int8_qat_model)
inits = onnx_model.graph.initializer
scales_map = {}
weights_map = {}
for init in inits:
    if "PPQ_Variable" in init.name:
        W = numpy_helper.to_array(init)
        scales_map[init.name] = W
    else:
        W = numpy_helper.to_array(init)
        weights_map[init.name] = W


# {'Relu', 'Mul', 'MaxPool', 'GlobalAveragePool', 'Conv', \
# 'QuantizeLinear', 'Resize', 'Add', 'Concat', 'HardSigmoid', 'DequantizeLinear', 'Sigmoid'}

# 第二步, 统计权重和输出的scale值
print("[Step2] Collect scales and average ")
acts_scale_map = {}
weights_scale_map = {}
for node in onnx_model.graph.node:
    if node.op_type in ["QuantizeLinear"]:
        act_name = node.input[0]
        scale_name = node.input[1]
        scale_value = scales_map[scale_name]
        if act_name in weights_map:  # 权重量化
            if act_name not in weights_scale_map:
                weights_scale_map[act_name] = []

            weights_scale_map[act_name].append(scale_value)
        else:  # act 量化
            if act_name not in acts_scale_map:
                acts_scale_map[act_name] = []

            acts_scale_map[act_name].append(scale_value)


for key, value in acts_scale_map.items():
    assert isinstance(value, list), " {} {}".format(key, value)
    assert isinstance(value[0], float) or np.size(value[0]) == 1, " {} {}".format(
        key, value
    )

    acts_scale_map[key] = float(np.median(value))
    # act_min_q = -128
    # act_max_q = 127
    # act_min = act_min_q * float(np.median(value))
    # act_max = act_max_q * float(np.median(value))
    # acts_scale_map[key] = max(abs(act_min), abs(act_max))
    # 这里是scale值 q = x/scale ---> -128, 127
    # 转换为min max值需要乘以128.0


for key, value in weights_scale_map.items():
    assert isinstance(value, list), " {} {}".format(key, value)
    weights_scale_map[key] = np.median(value, axis=0, keepdims=False)



# 第三步, 对权重部分，进行fakequant后，放回onnx模型中;
print("[Step3] Fake quant weights ")


def fake_quant(weight, scales):
    weight = np.array(weight)
    scales = np.array(scales)
    assert np.shape(weight)[0] == len(scales)
    # 权重量化在QAT中是-128, 127; 但是在直接转换中是-127,127
    quantized_weight = np.clip(np.round(weight / scales.reshape(-1, 1, 1, 1) + 0.0), -128, 127)
    # output = clamp(round(input / scale) + zeroPt)

    # 反量化
    weight_r = (quantized_weight.astype(np.float32) - 0.0) * scales.reshape(-1, 1, 1, 1)

    quant_output = np.reshape(weight, (1, -1))
    origin_output = np.reshape(weight_r, (1, -1))
    cos_sim = cosine_similarity(quant_output, origin_output)
    assert cos_sim > 0.99, " {} {} {}".format(
        cos_sim, scales.reshape((-1,))[:5], weight_r.reshape((-1,))[:5]
    )
    return weight_r


onnx_model = onnx.load(fp32_model)  # 主要目的是获取模型结构
inits = onnx_model.graph.initializer
for idx, init in enumerate(inits):
    if init.name in weights_scale_map:
        # 需要使用LSQ更新后的权重和scale
        W_new = fake_quant(weights_map[init.name], weights_scale_map[init.name])
        print(init.name, np.shape(W_new))
        tensor = numpy_helper.from_array(W_new, init.name)
        onnx_model.graph.initializer[idx].CopyFrom(tensor)
    # else:
    #     print(init.name)

onnx.save(onnx_model, onnx_updated_model_path)


acts_scale_map = {k: acts_scale_map[k] for k in sorted(acts_scale_map)}


# 第三步，对act部分，记录scale值, 生成calib.cache文件
print("[Step4] Dump act scales")
with open(onnx_updated_model_path + "_calib_cache.json", "w") as file:
    file.write(json.dumps(acts_scale_map, indent=4))  # use `json.loads` to do the reverse

# write plain text: tensorRT需要对结果做转换
# TRT-8400-EntropyCalibration2
# input.1: 3ca94044
# 9131: 3cf4f8d5
# 加密 hex(struct.unpack('<I', struct.pack('<f', f))[0])
# 解析 struct.unpack('!f', bytes.fromhex('41973333'))[0]
with open(onnx_updated_model_path + "_calib_cache.cache", "w") as file:
    file.write("TRT-8400-EntropyCalibration2\n")
    for key in sorted(acts_scale_map.keys()):
        scale = acts_scale_map[key]
        # if scale > 0.5:
        #     print("scale过大, 建议不量化:", key, scale, 128.0 * scale)
        #     continue
        scale_hex = hex(struct.unpack("<I", struct.pack("<f", scale))[0])
        s = key + ": " + str(scale_hex).lstrip("0x")
        file.write(s)
        file.write("\n")
