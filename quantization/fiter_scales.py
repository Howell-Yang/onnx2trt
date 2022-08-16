import struct

calib_cache = "/apdcephfs/private_howellyang/onnx2trt/model.trt_int8_with_1578pics_calib_entropy.calib_cache"

lines = []
with open(calib_cache) as fr:
    for i, line in enumerate(fr.readlines()):
        if i == 0:
            lines.append(line)
        else:
            name, value = line.strip().split(":")
            if value.strip() == "0":
                val = 0.0
            else:
                val = struct.unpack('!f', bytes.fromhex(value.strip()))[0]

            if val > 0.5:
                print(name, val)
            else:
                lines.append(line)

with open(calib_cache + "_filter_scale05.calib_cache", "w") as fw:
    fw.writelines(lines)