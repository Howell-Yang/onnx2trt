import sys
import struct
import json
import numpy as np

with open(sys.argv[1]) as fr:
    scale_map_ppq = json.load(fr)


with open(sys.argv[2]) as fr:
    scale_map_etp = json.load(fr)


# 第一步: etp里面，相同的值进行归类
same_value_map = {}
for name, value in scale_map_etp.items():
    value = np.round(value, 7)
    if value not in same_value_map:
        same_value_map[value] = []
    same_value_map[value].append(name)

# 第二步: 构建value map的映射 ----> 转换容易失败
same_name_map = {}
for value, names in same_value_map.items():
    if len(names) > 1:
        print(names)

exit(0)

scale_map_etp.update(scale_map_etp)


with open(sys.argv[3], "w") as file:
    file.write("TRT-8400-EntropyCalibration2\n")
    for key in sorted(scale_map_etp.keys()):
        scale = scale_map_etp[key]
        # if scale > 0.5:
        #     print("scale过大, 建议不量化:", key, scale, 128.0 * scale)
        #     continue
        if len(key) > 5:
            print(key, scale)
            continue
        scale_hex = hex(struct.unpack("<I", struct.pack("<f", scale))[0])
        s = key + ": " + str(scale_hex).lstrip("0x")
        file.write(s)
        file.write("\n")
