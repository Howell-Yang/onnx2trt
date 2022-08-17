import onnx
import numpy as np
from onnx import numpy_helper
from step02_onnx_simplify import get_post_nodes


def fuse_normalize_to_conv(onnx_model, means, scales, input_tensor_name=None):
    # Y = (x - means)/scales
    initializer_names = [
        initializer.name for initializer in onnx_model.graph.initializer
    ]
    inputs = [inp for inp in onnx_model.graph.input if inp not in initializer_names]
    if input_tensor_name is None:
        assert (
            len(inputs) == 1
        ), "if multiple input exists, please specify input_tensor_name"
        input_tensor_name = inputs[0]

    weight_name2tensor = {}
    for weight in onnx_model.graph.initializer:
        weight_name2tensor[weight.name] = weight

    # find post nodes
    post_nodes = get_post_nodes(onnx_model, input_tensor_name)
    for post_node in post_nodes:
        if post_node.op_type != "Conv":
            raise NameError(
                "the input tensor is used by non-Conv node, normalize process can't be fused"
            )

        paddings = [0]
        for attr in post_node.attribute:
            if attr.name == "pads":
                paddings = attr.ints
                break

        for pad in paddings:
            if pad != 0:
                raise NameError(
                    "the conv after input has padding, normalize process can't be fused"
                )

        group_num = 1
        for attr in post_node.attribute:
            if attr.name == "group":
                group_num = attr.i
        if group_num > 1:
            raise NameError(
                "the conv after input has group > 1, normalize process can't be fused"
            )

        # fuse normalize-conv
        assert (
            len(post_node.input) == 3
        ), " conv node must has bias for normalize fusion"
        weight_name = post_node.input[1]
        weight_tensor = weight_name2tensor[weight_name]
        weight_value = numpy_helper.to_array(weight_tensor)  # out_ch, in_ch, ker, ker

        bias_name = post_node.input[2]
        bias_tensor = weight_name2tensor[bias_name]
        bias_value = numpy_helper.to_array(bias_tensor)  # out_ch, in_ch, ker, ker

        assert (
            len(means) == len(scales) == np.shape(weight_value)[1]
        ), "mean and scale value mismatch the input channel num"

        means = np.reshape(np.array(means), (1, -1, 1, 1))
        scales = np.reshape(np.array(means), (1, -1, 1, 1))

        new_weight_value = np.array(weight_value / scales).astype(np.float32)
        raw_shape = tuple([i for i in weight_tensor.dims])
        new_shape = np.shape(new_weight_value)
        assert new_shape == raw_shape
        weight_tensor.ClearField("float_data")
        weight_tensor.ClearField("int32_data")
        weight_tensor.ClearField("int64_data")
        weight_tensor.raw_data = new_weight_value.tobytes()

        new_bias_value = np.array(
            bias_value - np.sum(weight_value * means / scales)
        ).astype(np.float32)
        raw_shape = tuple([i for i in bias_tensor.dims])
        new_shape = np.shape(new_bias_value)
        assert new_shape == raw_shape
        bias_tensor.ClearField("float_data")
        bias_tensor.ClearField("int32_data")
        bias_tensor.ClearField("int64_data")
        bias_tensor.raw_data = new_bias_value.tobytes()

    return onnx_model


if __name__ == "__main__":
    import sys

    onnx_model = onnx.load(sys.argv[1])
    means = eval(sys.argv[2])
    scales = eval(sys.argv[3])
    onnx_model = fuse_normalize_to_conv(onnx_model, means, scales)
