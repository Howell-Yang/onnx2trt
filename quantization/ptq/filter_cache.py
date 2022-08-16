from trt_utils import read_calib_cache
from trt_utils import write_cache_to_json
import onnx


# 过滤掉不输入onnx节点的量化值，以及数字过大的量化值
onnx_model = onnx.load("/mapai/howellyang/code/onnx2trt/RMTNet_release20220609_mm2conv.optimized.rm_inits.onnx")
onnx_output_names = []
for node in onnx_model.graph.node:
    for o in node.output:
        onnx_output_names.append(o)

scale_map = read_calib_cache("/mapai/howellyang/code/onnx2trt/RMTNet_release20220609_mm2conv.optimized.trt_int8_with_531pics_calib_percentile595.calib_cache")
calib_cache = "/mapai/howellyang/code/onnx2trt/RMTNet_release20220609_mm2conv.optimized.trt_int8_with_531pics_calib_percentile595.filtered.calib_cache"

scale_map = {k: scale_map[k] for k in sorted(scale_map)}
with open(calib_cache, "w") as file:
    file.write("TRT-8400-EntropyCalibration2\n")
    for key in sorted(scale_map.keys()):
        value = scale_map[key]
        scale = float(value)
        scale_hex = hex(struct.unpack("<I", struct.pack("<f", scale))[0])
        s = key + ": " + str(scale_hex).lstrip("0x")
        file.write(s)
        file.write("\n")