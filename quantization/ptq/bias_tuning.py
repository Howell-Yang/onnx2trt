# coding: utf-8
import json
from tqdm import tqdm
import numpy as np
import sys
import onnx
import onnxruntime
import torch
import glob
import onnxoptimizer
import os
from scipy.optimize import leastsq
from sklearn.metrics.pairwise import cosine_similarity
from copy import deepcopy
import time
from onnx import helper, shape_inference
from onnx import numpy_helper
from onnx import version_converter
from onnxsim.onnx_simplifier import simplify
from numpy import polyfit, poly1d
import random
print("start import packages")
print("import finished")


def read_image_v1(path):
    from PIL import Image
    from torchvision import transforms
    _img_transforms = transforms.Compose(
        [
            transforms.Resize((384, 768)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    img = Image.open(path).convert("RGB")
    img = _img_transforms(img)
    img = img.unsqueeze(0)
    return np.array(img)


def read_image_v2(path):
    import cv2
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    input_w = 960
    input_h = 480
    mean = np.array(mean)
    std = np.array(std)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_w, input_h))
    img = img.astype(np.float32)
    for i in range(3):
        img[..., i] = (img[..., i] - mean[i]) / std[i]
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return np.ascontiguousarray(img, dtype=np.float32)


def read_image_v3(path):
    import cv2
    mean_val = [103.53, 116.28, 123.675]
    std_val = [57.375, 57.12, 58.395]
    input_size = [768, 448]

    # img = np.random.randint(255, size=input_size + [3]).astype(np.uint8)
    img_raw = cv2.imread(path)
    img = cv2.resize(
        img_raw, (input_size[0], input_size[1])).astype(np.float32)
    img -= mean_val
    img /= std_val
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img = np.expand_dims(img, axis=0)

    img = np.ascontiguousarray(img, dtype=np.float32)
    # img_tensor = torch.from_numpy(img)
    return img


if sys.argv[3] == "v1":
    read_image = read_image_v1
elif sys.argv[3] == "v2":
    read_image = read_image_v2
elif sys.argv[3] == "v3":
    read_image = read_image_v3
else:
    print("please enter raed image type")
    exit(0)


def convert_any_to_numpy(x, accepet_none=True):
    if x is None and not accepet_none:
        raise ValueError("Trying to convert an empty value.")
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, int) or isinstance(x, float):
        return np.array(
            [
                x,
            ]
        )
    elif isinstance(x, torch.Tensor):
        if x.numel() == 0 and accepet_none:
            return None
        if x.numel() == 0 and not accepet_none:
            raise ValueError("Trying to convert an empty value.")
        if x.numel() == 1:
            return convert_any_to_numpy(x.detach().cpu().item())
        if x.numel() > 1:
            return x.detach().cpu().numpy()
    elif isinstance(x, list) or isinstance(x, tuple):
        return np.array(x)
    else:
        raise TypeError(
            "input value can not be converted as numpy type."
        )


print("reading calibration data")

calibration_files = glob.glob(
    os.path.join(
        "/mapai/howellyang/code/road-service/road_service/calib_images/", "*.jpg")
)
random.seed(103600)
random.shuffle(calibration_files)
calibration_files = calibration_files[:100]
input_data = [convert_any_to_numpy(read_image(path))
              for path in calibration_files]


class OnnxOptimizer:
    def __init__(self):
        self.debug = False
        onnx_model_path = sys.argv[1]
        onnx_model = onnx.load(onnx_model_path)
        onnx_model = self.add_input_from_initializer(onnx_model)
        onnx_model = shape_inference.infer_shapes(onnx_model)
        onnx_model = version_converter.convert_version(onnx_model, 13)
        onnx_model = onnxoptimizer.optimize(onnx_model)
        onnx_model, check = simplify(onnx_model)
        onnx_model = shape_inference.infer_shapes(onnx_model)
        onnx_model = self.remove_initializer_from_input(onnx_model)
        self.fp32_model = deepcopy(onnx_model)

        onnx_model_path = sys.argv[2]
        onnx_model = onnx.load(onnx_model_path)
        onnx_model = self.add_input_from_initializer(onnx_model)
        onnx_model = shape_inference.infer_shapes(onnx_model)
        onnx_model = version_converter.convert_version(onnx_model, 13)
        onnx_model = onnxoptimizer.optimize(onnx_model)
        # onnx_model, check = simplify(onnx_model)
        onnx_model = shape_inference.infer_shapes(onnx_model)
        onnx_model = self.remove_initializer_from_input(onnx_model)
        self.int8_model = deepcopy(onnx_model)

    def get_post_nodes(self, onnx_model, tensor_name):
        post_nodes = []
        for node in onnx_model.graph.node:
            for input_tensor in node.input:
                if input_tensor == tensor_name:
                    post_nodes.append(node)
                    break
        return post_nodes

    def remove_initializer_from_input(self, model):
        if model.ir_version < 4:
            print(
                "Model with ir_version below 4 requires to include initilizer in graph input"
            )
            return

        # ununsed constance nodes
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

        # remove inits in inputs
        inputs = model.graph.input
        name_to_input = {}
        for input in inputs:
            name_to_input[input.name] = input

        unused_initializers = []
        for initializer in model.graph.initializer:
            if initializer.name in name_to_input:
                inputs.remove(name_to_input[initializer.name])

            post_nodes = self.get_post_nodes(model, initializer.name)
            if len(post_nodes) == 0:
                unused_initializers.append(initializer)

        for initializer in unused_initializers:
            model.graph.initializer.remove(initializer)

        return model

    def add_input_from_initializer(self, model):
        """
        Currently onnx.shape_inference doesn't use the shape of initializers, so add
        that info explicitly as ValueInfoProtos.
        Mutates the model.
        Args:
            model: The ModelProto to update.
        """
        # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
        if model.ir_version < 4:
            return

        def add_const_value_infos_to_graph(graph):
            inputs = {i.name for i in graph.input}
            existing_info = {vi.name: vi for vi in graph.input}
            for init in graph.initializer:
                # Check it really is a constant, not an input
                if init.name in inputs:
                    continue

                # The details we want to add
                elem_type = init.data_type
                shape = init.dims

                # Get existing or create new value info for this constant
                vi = existing_info.get(init.name)
                if vi is None:
                    vi = graph.input.add()
                    vi.name = init.name

                # Even though it would be weird, we will not overwrite info even if it doesn't match
                tt = vi.type.tensor_type
                if tt.elem_type == onnx.TensorProto.UNDEFINED:
                    tt.elem_type = elem_type
                if not tt.HasField("shape"):
                    # Ensure we set an empty list if the const is scalar (zero dims)
                    tt.shape.dim.extend([])
                    for dim in shape:
                        tt.shape.dim.add().dim_value = dim

            # Handle subgraphs
            for node in graph.node:
                for attr in node.attribute:
                    # Ref attrs refer to other attrs, so we don't need to do anything
                    if attr.ref_attr_name != "":
                        continue

                    if attr.type == onnx.AttributeProto.GRAPH:
                        add_const_value_infos_to_graph(attr.g)
                    if attr.type == onnx.AttributeProto.GRAPHS:
                        for g in attr.graphs:
                            add_const_value_infos_to_graph(g)

        add_const_value_infos_to_graph(model.graph)
        return model

    def get_fp32_outputs(self, fp32_model, tensor_name=None):
        # remove origin outputs
        onnx_model = deepcopy(fp32_model)
        original_outputs = [output for output in onnx_model.graph.output]
        for o in original_outputs:
            onnx_model.graph.output.remove(o)

        if tensor_name is None:
            # add all output to graph output
            output_names = []
            for node in onnx_model.graph.node:
                for o in node.output:
                    intermediate_layer_value_info = onnx.helper.ValueInfoProto()
                    intermediate_layer_value_info.name = o
                    onnx_model.graph.output.append(
                        intermediate_layer_value_info)
                    output_names.append(o)
        else:
            intermediate_layer_value_info = onnx.helper.ValueInfoProto()
            intermediate_layer_value_info.name = tensor_name
            onnx_model.graph.output.append(intermediate_layer_value_info)
            output_names = [tensor_name]

        # run onnx infer
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        sess = onnxruntime.InferenceSession(
            onnx._serialize(onnx_model),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"], sess_options=options)

        input_name = sess.get_inputs()[0].name
        samples = [convert_any_to_numpy(sample) for sample in input_data]

        fp32_outputs = {output_name: [] for output_name in output_names}
        for sample in tqdm(samples):
            onnx_outputs = sess.run(output_names, {input_name: sample})
            for i, output_name in enumerate(output_names):
                fp32_outputs[output_name].append(onnx_outputs[i])
        return {output_name: np.array(fp32_outputs[output_name]) for output_name in output_names}


    def compare_onnx_onnx(self, fp32_model, int8_model, input_data):
        fp32_outputs = self.infer_onnx(fp32_model, input_data)
        int8_outputs = self.infer_onnx(int8_model, input_data)
        for name, fp32_outs in fp32_outputs.items():
            int8_outs = int8_outputs[name]
            abs_diffs = []
            relative_diffs = []
            cosine_sims = []
            for fp32_out, int8_out in zip(fp32_outs, int8_outs):
                abs_diff = np.mean(np.abs(fp32_out - int8_out))
                relative_diff = np.mean(
                    np.abs(fp32_out - int8_out)/(np.abs(fp32_out) + 1e-7))
                fp32_out = np.reshape(fp32_out, (1, -1))
                int8_out = np.reshape(int8_out, (1, -1))
                cos_sim = cosine_similarity(fp32_out, int8_out)
                abs_diffs.append(abs_diff)
                relative_diffs.append(relative_diff)
                cosine_sims.append(cos_sim)
            print("{}: abs_diff = {:6.4f}, rel_diff = {:6.4f}, cos_sim = {:6.4f}".format(
                name, np.mean(abs_diffs), np.mean(relative_diffs), 100.0 * np.mean(cosine_sims)))

    def infer_onnx(self, onnx_model, input_data):
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        sess = onnxruntime.InferenceSession(
            onnx._serialize(onnx_model), providers=["CUDAExecutionProvider", "CPUExecutionProvider"], sess_options=options
        )
        input_name = sess.get_inputs()[0].name
        onnx_output_names = [output.name for output in sess.get_outputs()]
        samples = [convert_any_to_numpy(sample) for sample in input_data]
        onnx_outputs_all = {name: [] for name in onnx_output_names}
        for sample in tqdm(samples, desc="onnx infer"):
            onnx_outputs = sess.run(onnx_output_names, {input_name: sample})
            for i, name in enumerate(onnx_output_names):
                onnx_outputs_all[name].append(onnx_outputs[i])
        return onnx_outputs_all

    def process(self, conv_name):
        out_tensor = ""
        for conv_node in self.fp32_model.graph.node:
            if conv_node.name == conv_name:
                out_tensor = conv_node.output[0]
                break

        # Step 01. 或者这个Conv的fp32输出
        fp32_outputs = self.get_fp32_outputs(self.fp32_model, out_tensor)

        # Step 02. 获取这个Conv的int8输出
        int8_outputs = self.get_fp32_outputs(self.int8_model, out_tensor)

        # Step 03. 进行scale和bias --> 优化最终输出
        print(np.shape(fp32_outputs[out_tensor])) # (1, 32, 96, 192)
        scales = []
        biases = []
        for channel in range(np.shape(fp32_outputs[out_tensor])[2]):
            Xi = np.reshape(int8_outputs[out_tensor][:, 0, channel, :, :], (-1,))
            Yi = np.reshape(fp32_outputs[out_tensor][:, 0, channel, :, :], (-1,))
            coeff = polyfit(Xi, Yi, 1)
            k = coeff[0]
            b = coeff[1]
            print(np.mean(Xi), np.mean(Yi), k, b, np.sum(Xi > 0), np.sum(Yi > 0))
            scales.append(k)
            biases.append(b)
        print("Final Scales:", scales)
        print("Final Bias:", biases)
        # Step 04. 计算bias前后的指标
        abs_diffs = []
        relative_diffs = []
        cosine_sims = []
        for fp32_out, int8_out in zip(fp32_outputs[out_tensor], int8_outputs[out_tensor]):
            # (100, 1, 32, 384, 768)
            abs_diff = np.mean(np.abs(fp32_out - int8_out))
            relative_diff = np.mean(
                np.abs(fp32_out - int8_out)/(np.abs(fp32_out) + 1e-7))
            fp32_out = np.reshape(fp32_out, (1, -1))
            int8_out = np.reshape(int8_out, (1, -1))
            cos_sim = cosine_similarity(fp32_out, int8_out)
            abs_diffs.append(abs_diff)
            relative_diffs.append(relative_diff)
            cosine_sims.append(cos_sim)

        print("before tuning: {}: abs_diff = {:6.4f}, rel_diff = {:6.4f}, cos_sim = {:6.4f}".format(
            out_tensor, np.mean(abs_diffs), np.mean(relative_diffs), 100.0 * np.mean(cosine_sims)))

        abs_diffs = []
        relative_diffs = []
        cosine_sims = []

        scales = np.reshape(scales, (1, -1, 1, 1))
        biases = np.reshape(biases, (1, -1, 1, 1))
        for fp32_out, int8_out in zip(fp32_outputs[out_tensor], int8_outputs[out_tensor]):
            # (1, 32, 384, 768)
            int8_out = int8_out * np.array(scales) + np.array(biases)

            abs_diff = np.mean(np.abs(fp32_out - int8_out))
            relative_diff = np.mean(
                np.abs(fp32_out - int8_out)/(np.abs(fp32_out) + 1e-7))
            fp32_out = np.reshape(fp32_out, (1, -1))
            int8_out = np.reshape(int8_out, (1, -1))
            cos_sim = cosine_similarity(fp32_out, int8_out)
            abs_diffs.append(abs_diff)
            relative_diffs.append(relative_diff)
            cosine_sims.append(cos_sim)

        print("after tuning: {}: abs_diff = {:6.4f}, rel_diff = {:6.4f}, cos_sim = {:6.4f}".format(
            out_tensor, np.mean(abs_diffs), np.mean(relative_diffs), 100.0 * np.mean(cosine_sims)))

        # # Step 05. 修改onnx模型中conv的bias
        # weight_name2tensor = {}
        # for weight in self.int8_model.graph.initializer:
        #     weight_name2tensor[weight.name] = weight

        # bias_name = conv_node.input[2]
        # bias_tensor = weight_name2tensor[bias_name]
        # bias_weight = numpy_helper.to_array(bias_tensor) # out_ch, in_ch, ker, ker
        # new_bias_weight = bias_weight + final_bias

        # bias_tensor.ClearField("float_data")
        # bias_tensor.ClearField("int32_data")
        # bias_tensor.ClearField("int64_data")
        # bias_tensor.raw_data = new_bias_weight.tobytes()

        # # Step 06. 移除这个Conv的输出量化节点 以及 权重量化节点
        # nodes_to_remove = []
        # for node in self.int8_model.graph.node:
        #     if node.op_type == "QuantizeLinear":
        #         if node.input[0] == out_tensor:
        #             nodes_to_remove.append(node)
        #             for dq_node in self.int8_model.graph.node:
        #                 if dq_node.input[0] == node.output[0]:
        #                     nodes_to_remove.append(dq_node)
        #                     for post_node in self.int8_model.graph.node:
        #                         for i, input in enumerate(post_node.input):
        #                             if input == dq_node.output[0]:
        #                                 post_node.input[i] = node.input[0]
        #         elif node.input[0] == conv_node.input[1]:
        #             nodes_to_remove.append(node)
        #             for dq_node in self.int8_model.graph.node:
        #                 if dq_node.input[0] == node.output[0]:
        #                     nodes_to_remove.append(dq_node)
        #                     for post_node in self.int8_model.graph.node:
        #                         for i, input in enumerate(post_node.input):
        #                             if input == dq_node.output[0]:
        #                                 post_node.input[i] = node.input[0]
        # for node in nodes_to_remove:
        #     self.int8_model.graph.node.remove(node)

        # # Step 07. 对比fp32模型和int8模型
        # self.compare_onnx_onnx(self.fp32_model, self.int8_model, input_data)
        return self.int8_model


onnx_opt = OnnxOptimizer()
int8_model = onnx_opt.process('Conv_2576')
onnx.save(int8_model, sys.argv[4])
