import sys
import struct
import json


with open(sys.argv[1]) as fr:
    scale_map = json.load(fr)

scale_map = {k: scale_map[k] for k in sorted(scale_map)}
with open(".".join(sys.argv[1].split(".")[:-1]) + ".cache", "w") as file:
    file.write("TRT-8400-EntropyCalibration2\n")
    for key in sorted(scale_map.keys()):
        value = scale_map[key]
        scale = float(value)
        scale_hex = hex(struct.unpack("<I", struct.pack("<f", scale))[0])
        s = key + ": " + str(scale_hex).lstrip("0x")
        file.write(s)
        file.write("\n")
