import sys
import struct
import json


scale_map = {}
with open(sys.argv[1]) as fr:
    for line in fr.readlines()[1:]:
        name, value = line.strip().split(":")
        if value.strip() == "0":
            val = 0.0
        else:
            val = struct.unpack("!f", bytes.fromhex(value.strip()))[0]

        scale_map[name] = val

scale_map = {k: scale_map[k] for k in sorted(scale_map)}

with open(".".join(sys.argv[1].split(".")[:-1]) + ".json", "w") as fw:
    json.dump(scale_map, fw, indent=4)
