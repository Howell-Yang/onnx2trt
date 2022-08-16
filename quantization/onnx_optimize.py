import onnx
import onnx
from onnx import helper, shape_inference
from onnx import TensorProto
from onnx.tools import update_model_dims
import onnxoptimizer
from onnxsim import simplify
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

# 获取想要的输出
input_names = ['input.1']
# output_names = ['1080']
# onnx.utils.extract_model(input_path, output_path, input_names, output_names)


# 推断shape
onnx_model = onnx.load(input_path)

model_opt = onnxoptimizer.optimize(onnx_model)
model_simp, check = simplify(model_opt)
model_simp = shape_inference.infer_shapes(model_simp)
onnx.save(model_simp, output_path)
