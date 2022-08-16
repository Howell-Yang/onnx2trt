import onnxoptimizer
from onnxsim import simplify
from onnx import shape_inference
from onnx import version_converter
import onnx
import sys


def remove_useless_constants(model):
    all_nodes = []
    for node in model.graph.node:
        all_nodes.append(node)

    input_name_to_nodes = {}
    for node in all_nodes:
        for input_name in node.input:
            if input_name not in input_name_to_nodes:
                input_name_to_nodes[input_name] = [node]
            else:
                input_name_to_nodes[input_name].append(node)

    unused_nodes = []
    for node in all_nodes:
        if node.op_type == "Constant" and node.output[0] not in input_name_to_nodes:
            unused_nodes.append(node)

    for node in unused_nodes:
        if node in model.graph.node:
            model.graph.node.remove(node)
    return model


def remove_initializers_from_inputs(model):
    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    unused_initializers = []
    for initializer in model.graph.initializer:
        # remove initializers from input
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

        post_nodes = OnnxModel.get_post_nodes(model, initializer.name)
        if len(post_nodes) == 0:
            unused_initializers.append(initializer)

    # remove unused initializers
    for initializer in unused_initializers:
        model.graph.initializer.remove(initializer)
    return model


def optimize_onnx_model(onnx_model):
    remove_useless_constants(onnx_model)
    remove_initializers_from_inputs(onnx_model)
    onnx_model = version_converter.convert_version(onnx_model, 13)
    onnx_model = onnxoptimizer.optimize(onnx_model)
    onnx_model, check = simplify(onnx_model)
    onnx_model = shape_inference.infer_shapes(onnx_model)
    return onnx_model


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    onnx_model = onnx.load(input_path)
    model_simp = optimize_onnx_model(onnx_model)
    onnx.save(model_simp, output_path)
