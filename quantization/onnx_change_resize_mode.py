import onnx
import sys
import onnxoptimizer
from onnx import helper, shape_inference
from onnxsim import simplify
from onnx import numpy_helper
import numpy as np

onnx_model = onnx.load(sys.argv[1])

for node in onnx_model.graph.node:
    if node.op_type == "Resize":
        for attr in node.attribute:
            if (attr.name == "coordinate_transformation_mode"):
                attr.s = "half_pixel".encode("UTF-8")
            elif attr.name == "mode":
                attr.s = "linear".encode("UTF-8")
            elif attr.name == "nearest_mode":
                attr.s = "round_prefer_floor".encode("UTF-8")

model_opt = onnxoptimizer.optimize(onnx_model)
# model_simp, check = simplify(model_opt)
model_simp = shape_inference.infer_shapes(model_opt)
onnx.save(model_simp, sys.argv[2])
