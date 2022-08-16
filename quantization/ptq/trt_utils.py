

import struct
from matplotlib import scale
import numpy as np

def read_calib_cache(calib_cache):
    scale_map = {}
    with open(calib_cache) as fr:
        for line in fr.readlines()[1:]:
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


def write_cache_to_json(scale_map, calib_cache):
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

def fake_quant_per_channel(weight, scale):
    # The scale is a vector of coefficients and must have the same size as the quantization axis. 
    # The quantization scale must consist of all positive float coefficients. 
    # The rounding method is rounding-to-nearest ties-to-even and clamping is in the range [-128, 127].
    # formula: clamp(round(input[k,c,r,s] / scale[k]))
    scale = np.reshape(scale, (-1, 1, 1, 1))
    int8_weight = np.clip(np.round(weight / scale), -128, 127)
    fake_quantized_weight = int8_weight * scale
    return fake_quantized_weight

if __name__ == "__main__":
    scale_map = read_calib_cache("/mapai/howellyang/code/onnx2trt/RMTNet_release20220609_mm2conv.optimized.balanced_v0.trt_int8_with_531pics_calib_percentile595.calib_cache")
    print(scale_map)