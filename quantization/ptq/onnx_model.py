import onnxoptimizer
from onnxsim import simplify
from onnx import shape_inference
from onnx import version_converter
from onnx import numpy_helper
import onnx
from copy import deepcopy
import numpy as np
from trt_utils import fake_quant_per_channel
import onnxruntime

class OnnxModel:
    def __init__(self, onnx_model_path: str):
        onnx_model = onnx.load(onnx_model_path)
        onnx_model = OnnxModel.optimize_model(onnx_model)
        self.fp32_model = onnx_model
        self.qdq_model = deepcopy(onnx_model)
        self.input_names = [input.name for input in onnx_model.graph.input]
        self.output_names = [output.name for output in onnx_model.graph.output]
        self.initializers = onnx_model.graph.initializer
        self.weight_names = [initializer.name for initializer in self.initializers]

        self.all_tensor_names = set()
        for node in self.fp32_model.graph.node:
            for output in node.output:
                self.all_tensor_names.add(output)

    @staticmethod
    def quantize_weights(onnx_model):
        for node in onnx_model.graph.node:
            if node.op_type in ["Conv", "Matmul"]:
                weight = None
                for init_node in onnx_model.graph.initializer:
                    if init_node.name == node.input[1]:
                        weight = numpy_helper.to_array(init_node)
                        break
                assert (
                    len(np.shape(weight)) == 4
                ), "Only Conv Weights Should Be Quantized"
                max_range = np.max(np.abs(weight), axis=(1, 2, 3))
                scale = max_range / 128.0
                scale = np.array(scale).astype(np.float32)
                quantized_weight = fake_quant_per_channel(weight, scale)
                quantized_weight = np.array(
                    quantized_weight).astype(np.float32)

                raw_shape = tuple([i for i in init_node.dims])
                new_shape = np.shape(quantized_weight)
                assert new_shape == raw_shape
                init_node.ClearField("float_data")
                init_node.ClearField("int32_data")
                init_node.ClearField("int64_data")
                init_node.raw_data = quantized_weight.tobytes()
        return onnx_model

    @staticmethod
    def quantize_weights_qdq(onnx_model):
        for node in onnx_model.graph.node:
            if node.op_type in ["Conv", "Matmul"]:
                weight = None
                for init_node in onnx_model.graph.initializer:
                    if init_node.name == node.input[1]:
                        weight = numpy_helper.to_array(init_node)
                        break
                assert (
                    len(np.shape(weight)) == 4
                ), "Only Conv Weights Should Be Quantized"
                max_range = np.max(np.abs(weight), axis=(1, 2, 3))
                scale = max_range / 128.0
                scale = np.array(scale).astype(np.float32)
                OnnxModel.

                # quantized_weight = fake_quant_per_channel(weight, scale)
                # quantized_weight = np.array(
                #     quantized_weight).astype(np.float32)

                # raw_shape = tuple([i for i in init_node.dims])
                # new_shape = np.shape(quantized_weight)
                # assert new_shape == raw_shape
                # init_node.ClearField("float_data")
                # init_node.ClearField("int32_data")
                # init_node.ClearField("int64_data")
                # init_node.raw_data = quantized_weight.tobytes()

        return onnx_model

    @staticmethod
    def add_act_dqd_node(qdq_model, tensor_name, scale):
        quant_node_name = tensor_name + "_QuantizeLinear"
        dequant_node_name = tensor_name + "_DequantizeLinear"
        q_input = tensor_name
        q_output = tensor_name + "_QuantizeLinear"
        dq_input = q_output
        dq_output = tensor_name + "_DequantizeLinear"

        scale_name = tensor_name + "_QuantizeScale"
        zp_name = tensor_name + "_QuantizeZp"
        qlinear_node = onnx.helper.make_node(
            "QuantizeLinear",
            [q_input, scale_name, zp_name],
            [q_output],
            quant_node_name,
        )
        dequant_node = onnx.helper.make_node(
            "DequantizeLinear",
            [dq_input, scale_name, zp_name],
            [dq_output],
            dequant_node_name,
        )

        for node in qdq_model.graph.node:
            for j in range(len(node.input)):
                if node.input[j] == tensor_name:
                    node.input[j] = dq_output

        qdq_model.graph.node.extend([qlinear_node, dequant_node])

        scale_initializer_tensor = onnx.helper.make_tensor(
            name=scale_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=(),
            vals=[scale],
        )
        zp_initializer_tensor = onnx.helper.make_tensor(
            name=zp_name,
            data_type=onnx.TensorProto.INT8,
            dims=(),
            vals=[0],
        )
        qdq_model.graph.initializer.append(scale_initializer_tensor)
        qdq_model.graph.initializer.append(zp_initializer_tensor)
        return dq_output

    @staticmethod
    def get_onnx_outputs(onnx_model, output_tensors, input_data):
        # remove original outputs
        dummy_onnx_model = deepcopy(onnx_model)
        outputs_list  = []
        for output in dummy_onnx_model.graph.output:
            outputs_list.append(output)
        for output in outputs_list:
            dummy_onnx_model.graph.output.remove(output)

        # add intermediant outputs
        for output in output_tensors:
            intermediate_layer_value_info = onnx.helper.ValueInfoProto()
            intermediate_layer_value_info.name = output
            dummy_onnx_model.graph.output.append(intermediate_layer_value_info)

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        sess = onnxruntime.InferenceSession(
            onnx._serialize(dummy_onnx_model),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            sess_options=options)

        input_name = sess.get_inputs()[0].name
        onnx_outpus_all = {}
        for sample in input_data:
            onnx_outputs = sess.run(output_tensors, {input_name: sample})
            for i, name in enumerate(output_tensors):
                if name not in onnx_outpus_all:
                    onnx_outpus_all[name] = []
                conv_output = onnx_outputs[i]
                median_value = np.median(conv_output, axis=(0,2,3))
                onnx_outpus_all[name].append(median_value)
        return onnx_outpus_all

    @staticmethod
    def optimize_model(onnx_model):
        OnnxModel.remove_useless_constants(onnx_model)
        OnnxModel.remove_initializers_from_inputs(onnx_model)
        onnx_model = version_converter.convert_version(onnx_model, 13)
        onnx_model = onnxoptimizer.optimize(onnx_model)
        onnx_model, check = simplify(onnx_model)
        onnx_model = shape_inference.infer_shapes(onnx_model)
        return onnx_model

    @staticmethod
    def get_post_nodes(onnx_model, tensor_name):
        post_nodes = []
        for node in onnx_model.graph.node:
            for input_tensor in node.input:
                if input_tensor == tensor_name:
                    post_nodes.append(node)
                    break
        return post_nodes

    @staticmethod
    def get_previous_nodes(onnx_model, tensor_name):
        previous_nodes = []
        for node in onnx_model.graph.node:
            for output_tensor in node.output:
                if output_tensor == tensor_name:
                    previous_nodes.append(node)
                    break
        return previous_nodes

    @staticmethod
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

    @staticmethod
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
