# coding: utf-8
print("start import packages")
import random
from omegaconf import EnumNode
from onnxsim.onnx_simplifier import simplify
from onnx import version_converter
from onnx import numpy_helper
from onnx import helper, shape_inference
import time
from copy import deepcopy
from sklearn.metrics.pairwise import cosine_similarity
import os
import onnxoptimizer
import glob
import torch
import onnxruntime
import onnx
import sys
import numpy as np
from tqdm import tqdm
import json
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
    os.path.join("/mapai/howellyang/code/road-service/road_service/calib_images/", "*.jpg")
)
random.seed(103600)
random.shuffle(calibration_files)
calibration_files = calibration_files[:100]
input_data = [convert_any_to_numpy(read_image(path))
              for path in calibration_files]

class OnnxOptimizer:
    def __init__(self, onnx_model_path=sys.argv[1]):
        self.debug = False
        onnx_model = onnx.load(onnx_model_path)
        onnx_model = self.add_input_from_initializer(onnx_model)
        onnx_model = shape_inference.infer_shapes(onnx_model)
        onnx_model = version_converter.convert_version(onnx_model, 13)
        onnx_model = onnxoptimizer.optimize(onnx_model)
        onnx_model, check = simplify(onnx_model)
        onnx_model = shape_inference.infer_shapes(onnx_model)
        onnx_model = self.remove_initializer_from_input(onnx_model)
        self.fp32_model = deepcopy(onnx_model)
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


    def get_fp32_outputs(self, fp32_model):
        # remove origin outputs
        onnx_model = deepcopy(fp32_model)
        original_outputs = [output for output in onnx_model.graph.output]
        for o in original_outputs:
            onnx_model.graph.output.remove(o)

        # add all output to graph output
        output_names = []
        for node in onnx_model.graph.node:
            for o in node.output:
                intermediate_layer_value_info = onnx.helper.ValueInfoProto()
                intermediate_layer_value_info.name = o
                onnx_model.graph.output.append(intermediate_layer_value_info)
                output_names.append(o)

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
            onnx_outputs = sess.run(output_names, {input_name: sample })
            for i, output_name in enumerate(output_names):
                fp32_outputs[output_name].append(onnx_outputs[i])
        return fp32_outputs


    def fake_quant(self, weight, scales):
        weight = np.array(weight)
        scales = np.array(scales)
        assert np.shape(weight)[0] == len(scales)
        quantized_weight = np.clip(np.round(weight / scales.reshape(-1, 1, 1, 1) + 0.0), -127, 127)
        weight_r = (quantized_weight.astype(np.float32) - 0.0) * scales.reshape(-1, 1, 1, 1)
        quant_output = np.reshape(weight, (1, -1))
        origin_output = np.reshape(weight_r, (1, -1))
        cos_sim = cosine_similarity(quant_output, origin_output)
        assert cos_sim > 0.99, " {} {} {}".format(
            cos_sim, scales.reshape((-1,))[:5], weight_r.reshape((-1,))[:5]
        )
        return weight_r


    def quantize_weights(self, onnx_model):
        # 将所有权重全部提前量化，并加入到已量化的节点中
        weights_scale_map = {}
        weights_map = {}
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
                max_range = np.percentile(np.abs(weight), 100.0, axis=(1, 2, 3))
                scale = max_range / 128.0
                scale = np.array(scale).astype(np.float32)
                weights_scale_map[init_node.name] = scale
                weights_map[init_node.name] = weight

        inits = onnx_model.graph.initializer
        for idx, init in enumerate(inits):
            if init.name in weights_scale_map:
                # 需要使用LSQ更新后的权重和scale
                W_new = self.fake_quant(weights_map[init.name], weights_scale_map[init.name])
                tensor = numpy_helper.from_array(W_new, init.name)
                onnx_model.graph.initializer[idx].CopyFrom(tensor)
        return onnx_model


    def caculate_scale(self, fp32_outputs):
        act_scale_map = {}
        count = 0
        total = len(fp32_outputs)
        for name, value in fp32_outputs.items():
            count += 1
            best_error = float("inf")
            best_scale = 1e-7
            best_cosine = -1.0
            num_search = 100
            best_percentile = -1.0
            best_relative_error = float("inf")
            best_abs_error = float("inf")
            for idx in range(num_search + 1):
                # 0.99 -- 1.00
                # print(np.min(value), np.max(value))
                percentile = (0.975 + idx/num_search * 0.025) * 100.0
                max_range = np.percentile(np.abs(value), percentile)
                scale = max_range / 128.0
                if scale < 1e-5:
                    continue
                quantized_act = np.clip(np.round(value / scale + 0.0), -128.0, 128.0)
                fake_quantized_value = (quantized_act.astype(np.float32) - 0.0) * scale
                abs_error = np.mean(np.abs(fake_quantized_value - value))
                relative_error = np.mean(np.abs(fake_quantized_value - value)/np.abs(value + 1e-8))
                cosines = []
                for i in range(len(value)):
                    cos_sim = cosine_similarity(np.reshape(value[i], (1, -1)), np.reshape(fake_quantized_value[i], (1, -1)))
                    cosines.append(cos_sim)
                cos_sim = np.mean(cosines)
                error = relative_error + 1 - cos_sim
                if error < best_error:
                    best_error = error
                    best_relative_error = relative_error
                    best_abs_error = abs_error
                    best_cosine = cos_sim
                    best_scale = scale
                    best_percentile = percentile
            if best_relative_error > 0.20 or best_cosine < 0.95:
                print("*" * 20, end = "\t")
                # 特殊的case不量化
            else:
                act_scale_map[name] = best_scale
            print("{}/{}, {}: abs_diff = {:6.4f}, rel_diff = {:6.4f}, cos_sim = {:6.4f}".format(count, total, name, best_abs_error, best_relative_error, 100.0* best_cosine))
        return act_scale_map


    # 添加act量化节点
    def add_act_dqd_node(self, onnx_model, tensor_name, scale):
        qdq_model = deepcopy(onnx_model)
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
            dims=(),  # [1],
            vals=[scale],
        )

        zp_initializer_tensor = onnx.helper.make_tensor(
            name=zp_name,
            data_type=onnx.TensorProto.INT8,
            dims=(),  # [1],
            vals=[0],
        )

        qdq_model.graph.initializer.append(scale_initializer_tensor)
        qdq_model.graph.initializer.append(zp_initializer_tensor)
        return qdq_model


    def add_qdq_nodes(self, onnx_model, act_scale_map):
        for tensor_name, scale in act_scale_map.items():
            onnx_model = self.add_act_dqd_node(onnx_model, tensor_name, scale)

        return onnx_model


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
                relative_diff = np.mean(np.abs(fp32_out - int8_out)/(np.abs(fp32_out)  + 1e-7))
                fp32_out = np.reshape(fp32_out, (1, -1))
                int8_out = np.reshape(int8_out, (1, -1))
                cos_sim = cosine_similarity(fp32_out, int8_out)
                abs_diffs.append(abs_diff)
                relative_diffs.append(relative_diff)
                cosine_sims.append(cos_sim)

            print("{}: abs_diff = {:6.4f}, rel_diff = {:6.4f}, cos_sim = {:6.4f}".format(name, np.mean(abs_diffs), np.mean(relative_diffs), 100.0* np.mean(cosine_sims)))


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
        onnx_outputs_all  = {name : [] for name in onnx_output_names}
        for sample in tqdm(samples, desc="onnx infer"):
            onnx_outputs = sess.run(onnx_output_names, {input_name: sample})
            for i, name in enumerate(onnx_output_names):
                onnx_outputs_all[name].append(onnx_outputs[i])
        return onnx_outputs_all

    def process(self): 
        # step 01: get fp32 outputs
        self.fp32_outputs = self.get_fp32_outputs(self.fp32_model)
        # for name, value in self.fp32_outputs.items():
        #     save_path = os.path.join("/mapai/howellyang/code/onnx2trt/fp32_outputs", name + ".npy")
        #     np.save(save_path, np.array(value))


        # Step 02: quantize weights
        self.int8_model = self.quantize_weights(self.int8_model)


        # Step 03: caculate act scales
        act_scale_map = self.caculate_scale(self.fp32_outputs)
        with open("./scale_map.json", "w") as fw:
            json.dump(act_scale_map, fw)
        # Step 04: create int8 onnx model
        self.int8_model = self.add_qdq_nodes(self.int8_model, act_scale_map)


        # # Step 05: compare fp32 and int8 outputs
        self.compare_onnx_onnx(self.fp32_model, self.int8_model, input_data)


        return self.int8_model
        # self.fp32_model = deepcopy(onnx_model)
        # self.input_names = [input.name for input in onnx_model.graph.input]
        # self.inits_nodes = onnx_model.graph.initializer
        # self.inits_names = [init.name for init in onnx_model.graph.initializer]
        # self.output_names = [output.name for output in onnx_model.graph.output]

        # self.constant_names = self.get_constant_names(self.fp32_model)
        # self.next_to_quant = []
        # self.already_quantized = [None]  # 记录已经量化过的tensor名称
        # self.quantized_cosines = [1.0]  # 记录量化过程的每个节点对应的cosine值
        # self.next_to_quant.extend(self.input_names)

        # self.concat_outputs_with_shared_input = (
        #     self.get_concat_outputs_with_shared_input(self.fp32_model)
        # )  # 不同的Concat节点，存在共享的输入，Concat输出不能量化

        # self.no_need_to_quantize = self.input_no_quantization_tensors(
        #     self.fp32_model
        # )  # 不需要量化的tensor名称

        # self.total_nodes_num = (
        #     len(self.fp32_model.graph.node)
        #     # + len(self.inits_names)
        #     - len(self.no_need_to_quantize)
        # )
        # self.quantized_nodes_num = 0
        # print("reading input data")
        # self.calibration_data = calibration_data

        # print("getting fp32 outputs")
        # if not self.debug:
        #     self.fp32_outputs = self.infer_with_onnx(
        #         self.fp32_model, self.calibration_data
        #     )

        # # 记录scale值
        # self.scale_map = {}

        # self.qdq_model = self.quantize_weights(deepcopy(self.fp32_model))

        # self.each_node_time = None



#         # # move other constant nodes to initializers ------->simplify过程会把这些都干掉
#         # constant_nodes = []
#         # for node in model.graph.node:
#         #     if node.op_type == "Constant":
#         #         constant_nodes.append(node)

#         # for node in constant_nodes:
#         #     attr = None
#         #     for attr in node.attribute:
#         #         if attr.name == "value":
#         #             break
#         #     # 根据类型不同, 获取不同的value
#         #     # FLOAT = 1;
#         #     # INT = 2;
#         #     # STRING = 3;
#         #     # FLOATS = 6;
#         #     # INTS = 7;
#         #     # STRINGS = 8;
#         #     if attr.type == onnx.AttributeProto.AttributeType.FLOAT:
#         #         initializer_tensor = onnx.helper.make_tensor(
#         #             name=node.output[0],
#         #             data_type=attr.type,
#         #             dims=(),
#         #             vals=attr.f,
#         #         )
#         #     elif attr.type == onnx.AttributeProto.AttributeType.INT:
#         #         initializer_tensor = onnx.helper.make_tensor(
#         #             name=node.output[0],
#         #             data_type=attr.type,
#         #             dims=(),
#         #             vals=attr.i,
#         #         )
#         #     elif (
#         #         attr.type == onnx.AttributeProto.AttributeType.STRING
#         #         or attr.type == onnx.AttributeProto.AttributeType.STRINGS
#         #     ):
#         #         print(node.name, "attr type not float or int", attr.type)
#         #         raise NameError(
#         #             "Constant to Initializer not Supported: String or Strings"
#         #         )
#         #     elif attr.type == onnx.AttributeProto.AttributeType.FLOATS:
#         #         initializer_tensor = onnx.helper.make_tensor(
#         #             name=node.output[0],
#         #             data_type=attr.type,
#         #             dims=(len(attr.floats)),
#         #             vals=attr.floats,
#         #         )
#         #     elif attr.type == onnx.AttributeProto.AttributeType.INTS:
#         #         initializer_tensor = onnx.helper.make_tensor(
#         #             name=node.output[0],
#         #             data_type=attr.type,
#         #             dims=(len(attr.ints)),
#         #             vals=attr.ints,
#         #         )
#         #     else:
#         #         print(node.name, "attr type not float or int", attr.type)
#         #         raise NameError("Constant to Initializer not Supported: Unknown")
#         #     # 移除constant nodes
#         #     if node in model.graph.node:
#         #         print("remove node from graph", node.name)
#         #         model.graph.node.remove(node)
#         #     # 添加initi
#         #     model.graph.initializer.append(initializer_tensor)

#         # remove inits in inputs
#         inputs = model.graph.input
#         name_to_input = {}
#         for input in inputs:
#             name_to_input[input.name] = input

#         unused_initializers = []
#         for initializer in model.graph.initializer:
#             if initializer.name in name_to_input:
#                 inputs.remove(name_to_input[initializer.name])

#             post_nodes = self.get_post_nodes(model, initializer.name)
#             if len(post_nodes) == 0:
#                 unused_initializers.append(initializer)

#         for initializer in unused_initializers:
#             model.graph.initializer.remove(initializer)

#         return model

    

#     def cosine_compute(self, fp32_outputs, qdq_outputs):
#         sims = {}
#         for i in range(len(qdq_outputs)):
#             for output_name, _ in qdq_outputs[i].items():
#                 trt_output = np.reshape(qdq_outputs[i][output_name], (1, -1))
#                 trt_fp32_output = np.reshape(
#                     fp32_outputs[i][output_name], (1, -1))
#                 cos_sim = cosine_similarity(trt_output, trt_fp32_output)
#                 mse = mean_squared_error(trt_output, trt_fp32_output, squared=False)
#                 if output_name not in sims:
#                     sims[output_name] = []
#                 sims[output_name].append(cos_sim.ravel() - mse.ravel())

#         mean_sims = []
#         for key, value in sims.items():
#             print("\t\t\t", key, np.mean(value), np.min(value))
#             mean_sims.append(np.mean(value))
#         return np.mean(mean_sims)

#     # 设置act量化的scale和zp
#     def create_act_initializer_tensor(
#         self,
#         name,
#         tensor_array,
#         data_type=onnx.TensorProto.FLOAT,
#     ):

#         # (TensorProto)
#         initializer_tensor = onnx.helper.make_tensor(
#             name=name,
#             data_type=data_type,
#             dims=(),  # [1],
#             vals=[tensor_array],
#         )

#         return initializer_tensor

#     # 设置weight量化的scale和zp
#     def create_weight_initializer_tensor(
#         self,
#         name,
#         tensor_array,
#         data_type=onnx.TensorProto.FLOAT,
#     ):

#         # (TensorProto)
#         initializer_tensor = onnx.helper.make_tensor(
#             name=name,
#             data_type=data_type,
#             dims=np.shape(tensor_array),
#             vals=tensor_array.flatten(),
#         )
#         return initializer_tensor

#     # 添加act量化节点
#     def add_act_dqd_node(self, onnx_model, tensor_name, scale):
#         qdq_model = deepcopy(onnx_model)
#         quant_node_name = tensor_name + "_QuantizeLinear"
#         dequant_node_name = tensor_name + "_DequantizeLinear"
#         q_input = tensor_name
#         q_output = tensor_name + "_QuantizeLinear"
#         dq_input = q_output
#         dq_output = tensor_name + "_DequantizeLinear"

#         scale_name = tensor_name + "_QuantizeScale"
#         zp_name = tensor_name + "_QuantizeZp"
#         qlinear_node = onnx.helper.make_node(
#             "QuantizeLinear",
#             [q_input, scale_name, zp_name],
#             [q_output],
#             quant_node_name,
#         )
#         dequant_node = onnx.helper.make_node(
#             "DequantizeLinear",
#             [dq_input, scale_name, zp_name],
#             [dq_output],
#             dequant_node_name,
#         )

#         for node in qdq_model.graph.node:
#             for j in range(len(node.input)):
#                 if node.input[j] == tensor_name:
#                     node.input[j] = dq_output

#         qdq_model.graph.node.extend([qlinear_node, dequant_node])

#         scale_initializer_tensor = self.create_act_initializer_tensor(
#             name=scale_name, tensor_array=scale, data_type=onnx.TensorProto.FLOAT
#         )

#         zp_initializer_tensor = self.create_act_initializer_tensor(
#             name=zp_name, tensor_array=0, data_type=onnx.TensorProto.INT8
#         )

#         qdq_model.graph.initializer.append(scale_initializer_tensor)
#         qdq_model.graph.initializer.append(zp_initializer_tensor)
#         return qdq_model

#     # 添加weight量化节点
#     def add_weight_dqd_node(self, onnx_model, tensor_name, scale):
#         qdq_model = deepcopy(onnx_model)
#         quant_node_name = tensor_name + "_QuantizeLinear"
#         dequant_node_name = tensor_name + "_DequantizeLinear"
#         q_input = tensor_name
#         q_output = tensor_name + "_QuantizeLinear"
#         dq_input = q_output
#         dq_output = tensor_name + "_DequantizeLinear"

#         scale_name = tensor_name + "_QuantizeScale"
#         zp_name = tensor_name + "_QuantizeZp"
#         qlinear_node = onnx.helper.make_node(
#             op_type="QuantizeLinear",
#             inputs=[q_input, scale_name, zp_name],
#             outputs=[q_output],
#             name=quant_node_name,
#             axis=0,
#         )

#         dequant_node = onnx.helper.make_node(
#             op_type="DequantizeLinear",
#             inputs=[dq_input, scale_name, zp_name],
#             outputs=[dq_output],
#             name=dequant_node_name,
#             axis=0,
#         )
#         for node in qdq_model.graph.node:
#             for j in range(len(node.input)):
#                 if node.input[j] == tensor_name:
#                     node.input[j] = dq_output

#         qdq_model.graph.node.extend([qlinear_node, dequant_node])

#         scale_initializer_tensor = self.create_weight_initializer_tensor(
#             name=scale_name, tensor_array=scale, data_type=onnx.TensorProto.FLOAT
#         )

#         zp_initializer_tensor = self.create_weight_initializer_tensor(
#             name=zp_name,
#             tensor_array=np.zeros_like(scale, dtype=np.int8),
#             data_type=onnx.TensorProto.INT8,
#         )

#         qdq_model.graph.initializer.append(scale_initializer_tensor)
#         qdq_model.graph.initializer.append(zp_initializer_tensor)
#         return qdq_model

#     def get_fp32_outputs(self, output_v):
#         # 使用fp32模型，获取当前节点的fp32输出
#         if output_v in self.inits_names:  # 当前节点是权重
#             weight = None
#             for init_node in self.inits_nodes:
#                 if init_node.name == output_v:
#                     weight = numpy_helper.to_array(init_node)
#                     break
#             assert (
#                 len(np.shape(weight)) == 4
#             ), "Only Conv Weights Should Be Quantized"
#             cur_node_output = weight
#         elif output_v in self.input_names:
#             cur_node_output = self.calibration_data
#         else:
#             cur_node_output = self.infer_with_onnx(
#                 self.fp32_model, self.calibration_data, output_node=output_v
#             )
#             cur_node_output = [op[output_v] for op in cur_node_output]
#         return cur_node_output

#     def get_post_nodes(self, onnx_model, tensor_name):
#         post_nodes = []
#         for node in onnx_model.graph.node:
#             for input_tensor in node.input:
#                 if input_tensor == tensor_name:
#                     post_nodes.append(node)
#                     break
#         return post_nodes

#     def get_pre_nodes(self, onnx_model, tensor_name):
#         pre_nodes = []
#         for node in onnx_model.graph.node:
#             for output_tensor in node.output:
#                 if output_tensor == tensor_name:
#                     pre_nodes.append(node)
#                     break
#         return pre_nodes

#     def add_tensor_to_next_to_quant(self, output):
#         # 已经准备量化的，不再量化
#         if output in self.next_to_quant:
#             return

#         # 输出节点不再量化
#         if output in self.output_names:
#             return

#         # 已经量化的不再量化
#         if output in self.already_quantized:
#             return

#         self.next_to_quant.append(output)

#     # 添加act量化节点
#     def create_act_search_model(self, onnx_model, tensor_name):
#         qdq_model = deepcopy(onnx_model)
#         quant_node_name = tensor_name + "_QuantizeLinear"
#         dequant_node_name = tensor_name + "_DequantizeLinear"
#         q_input = tensor_name
#         q_output = tensor_name + "_QuantizeLinear"
#         dq_input = q_output
#         dq_output = tensor_name + "_DequantizeLinear"

#         scale_name = tensor_name + "_QuantizeScale"
#         zp_name = tensor_name + "_QuantizeZp"
#         qlinear_node = onnx.helper.make_node(
#             "QuantizeLinear",
#             [q_input, scale_name, zp_name],
#             [q_output],
#             quant_node_name,
#         )
#         dequant_node = onnx.helper.make_node(
#             "DequantizeLinear",
#             [dq_input, scale_name, zp_name],
#             [dq_output],
#             dequant_node_name,
#         )

#         for node in qdq_model.graph.node:
#             for j in range(len(node.input)):
#                 if node.input[j] == tensor_name:
#                     node.input[j] = dq_output

#         qdq_model.graph.node.extend([qlinear_node, dequant_node])
#         zp_initializer_tensor = self.create_act_initializer_tensor(
#             name=zp_name, tensor_array=0, data_type=onnx.TensorProto.INT8
#         )

#         qdq_model.graph.initializer.append(zp_initializer_tensor)

#         scale_initializer_tensor = onnx.helper.make_tensor_value_info(
#             scale_name,
#             onnx.TensorProto.FLOAT,
#             shape=(),
#         )
#         qdq_model.graph.input.append(scale_initializer_tensor)
#         return qdq_model, scale_name

#     def create_weight_search_model(self, onnx_model, tensor_name, num_channels):
#         qdq_model = deepcopy(onnx_model)
#         quant_node_name = tensor_name + "_QuantizeLinear"
#         dequant_node_name = tensor_name + "_DequantizeLinear"
#         q_input = tensor_name
#         q_output = tensor_name + "_QuantizeLinear"
#         dq_input = q_output
#         dq_output = tensor_name + "_DequantizeLinear"

#         scale_name = tensor_name + "_QuantizeScale"
#         zp_name = tensor_name + "_QuantizeZp"
#         qlinear_node = onnx.helper.make_node(
#             op_type="QuantizeLinear",
#             inputs=[q_input, scale_name, zp_name],
#             outputs=[q_output],
#             name=quant_node_name,
#             axis=0,
#         )

#         dequant_node = onnx.helper.make_node(
#             op_type="DequantizeLinear",
#             inputs=[dq_input, scale_name, zp_name],
#             outputs=[dq_output],
#             name=dequant_node_name,
#             axis=0,
#         )
#         for node in qdq_model.graph.node:
#             for j in range(len(node.input)):
#                 if node.input[j] == tensor_name:
#                     node.input[j] = dq_output

#         qdq_model.graph.node.extend([qlinear_node, dequant_node])
#         zp_initializer_tensor = self.create_weight_initializer_tensor(
#             name=zp_name,
#             tensor_array=np.zeros((num_channels,), dtype=np.int8),
#             data_type=onnx.TensorProto.INT8,
#         )

#         qdq_model.graph.initializer.append(zp_initializer_tensor)

#         scale_initializer_tensor = onnx.helper.make_tensor_value_info(
#             scale_name,
#             onnx.TensorProto.FLOAT,
#             shape=(num_channels,),
#         )
#         qdq_model.graph.input.append(scale_initializer_tensor)
#         return qdq_model, scale_name

#     def search_scale(self,
#                      output_v,
#                      qdq_model,
#                      scale_name,
#                      cur_node_fp32_output
#                      ):

#         # 创建inference session
#         last_cosine = self.quantized_cosines[-1]
#         sess = onnxruntime.InferenceSession(
#             onnx._serialize(qdq_model),
#             providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

#         # 初始化搜索范围
#         searched_scales = []
#         searched_performences = []
#         search_scales = [100, 99.995, 99.95, 99.5, 99.0]
#         best_cosine = -100.0
#         best_scale = None
#         for percentile in search_scales:
#             if percentile in searched_scales:
#                 continue

#             if output_v in self.inits_names:
#                 max_range = np.percentile(
#                     np.abs(cur_node_fp32_output), percentile, axis=(1, 2, 3))
#                 scale = max_range / 128.0
#                 scale = np.array(scale).astype(np.float32)
#             else:
#                 max_range = np.percentile(
#                     np.abs(cur_node_fp32_output), percentile)
#                 scale = max_range / 128.0
#                 scale = np.array((scale)).astype(np.float32)

#             if not self.debug:

#                 input_name = sess.get_inputs()[0].name
#                 samples = [convert_any_to_numpy(sample)
#                            for sample in input_data]
#                 qdq_outputs = []
#                 for sample in samples:
#                     onnx_outputs = sess.run(
#                         self.output_names, {input_name: sample, scale_name: scale})
#                     onnx_outputs_dict = {
#                         self.output_names[i]: onnx_outputs[i]
#                         for i in range(len(self.output_names))
#                     }
#                     qdq_outputs.append(deepcopy(onnx_outputs_dict))
#                 qdq_cosine = self.cosine_compute(
#                     self.fp32_outputs, qdq_outputs)
#             else:
#                 qdq_cosine = 0.9999

#             print("     [search]", output_v, percentile, qdq_cosine)
#             searched_scales.append(percentile)
#             searched_performences.append(qdq_cosine)
#             if best_cosine < qdq_cosine:
#                 best_scale = scale
#                 best_cosine = qdq_cosine

#             # 相比于上一个模型，下降幅度极小，则退出
#             if best_cosine/last_cosine >= 0.9995:
#                 return best_cosine, best_scale

#         # 进一步进行插值搜索
#         for _ in range(5):
#             search_idx = np.argsort(searched_performences)
#             search_scales = [
#                 (searched_scales[search_idx[-2]] + searched_scales[search_idx[-1]]) / 2.0,
#                 (searched_scales[search_idx[-3]] + searched_scales[search_idx[-2]]) / 2.0,
#                 (searched_scales[search_idx[-4]] + searched_scales[search_idx[-3]]) / 2.0,
#                 (searched_scales[search_idx[-5]] + searched_scales[search_idx[-4]]) / 2.0,
#             ]
#             for percentile in search_scales:
#                 if percentile in searched_scales:
#                     continue

#                 if output_v in self.inits_names:
#                     max_range = np.percentile(
#                         np.abs(cur_node_fp32_output), percentile, axis=(1, 2, 3))
#                     scale = max_range / 128.0
#                     scale = np.array(scale).astype(np.float32)
#                 else:
#                     max_range = np.percentile(
#                         np.abs(cur_node_fp32_output), percentile)
#                     scale = max_range / 128.0
#                     scale = np.array((scale)).astype(np.float32)

#                 if not self.debug:

#                     input_name = sess.get_inputs()[0].name
#                     samples = [convert_any_to_numpy(sample)
#                             for sample in input_data]
#                     qdq_outputs = []
#                     for sample in samples:
#                         onnx_outputs = sess.run(
#                             self.output_names, {input_name: sample, scale_name: scale})
#                         onnx_outputs_dict = {
#                             self.output_names[i]: onnx_outputs[i]
#                             for i in range(len(self.output_names))
#                         }
#                         qdq_outputs.append(deepcopy(onnx_outputs_dict))
#                     qdq_cosine = self.cosine_compute(
#                         self.fp32_outputs, qdq_outputs)
#                 else:
#                     qdq_cosine = 0.9999

#                 print("     [search]", output_v, percentile, qdq_cosine)
#                 searched_scales.append(percentile)
#                 searched_performences.append(qdq_cosine)
#                 if best_cosine < qdq_cosine:
#                     best_scale = scale
#                     best_cosine = qdq_cosine

#                 # 相比于上一个模型，下降幅度极小，则退出
#                 if best_cosine/last_cosine >= 0.9995:
#                     return best_cosine, best_scale
#         return best_cosine, best_scale

#     def process(self):
#         while len(self.next_to_quant) > 0:
#             start_time = time.time()
#             output_v = self.next_to_quant[-1]
#             self.next_to_quant = self.next_to_quant[:-1]
#             print("【Step 1】searching for ", output_v)

#             # 第一步, 创建用于搜索值域范围的模型
#             if self.debug:
#                 cur_node_fp32_output = [1, 1, 1]
#             else:
#                 cur_node_fp32_output = self.get_fp32_outputs(output_v)

#             if output_v in self.inits_names:
#                 num_channels = np.shape(cur_node_fp32_output)[0]
#                 qdq_model, scale_name = self.create_weight_search_model(
#                     self.qdq_model, output_v, num_channels)
#             else:
#                 qdq_model, scale_name = self.create_act_search_model(
#                     self.qdq_model, output_v)

#             # 第二步，进行值域搜索
#             best_cosine, best_scale = self.search_scale(
#                 output_v,
#                 qdq_model,
#                 scale_name,
#                 cur_node_fp32_output,  # 用于获取scale值
#             )
#             if output_v not in self.inits_names:
#                 self.scale_map[output_v] = best_scale
#             print("【Step 2】quantized finished=] ", output_v, best_cosine)
#             if output_v in self.inits_names:
#                 self.qdq_model = self.add_weight_dqd_node(
#                     self.qdq_model, output_v, best_scale)
#             else:
#                 self.qdq_model = self.add_act_dqd_node(
#                     self.qdq_model, output_v, best_scale)
#             self.already_quantized.append(output_v)
#             self.quantized_cosines.append(best_cosine)

#             # 第三步，更新下一步量化的节点
#             post_nodes = self.get_post_nodes(self.fp32_model, output_v)
#             print("【Step 3】查找下一步量化的节点:", [pnd.name for pnd in post_nodes])
#             while len(post_nodes) > 0:
#                 post_node = post_nodes[-1]
#                 post_nodes = post_nodes[:-1]
#                 print("\t 【节点{}的前置{}输入已完成量化, 尝试量化这个节点".format(
#                     post_node.name, output_v))

#                 # 单输入，无权重，单输出的节点，且没有权重
#                 if post_node.op_type in [
#                     "Clip",
#                     "Relu",
#                     "Sigmoid",
#                     "HardSigmoid",
#                     "LeakyRelu",
#                     "GlobalAveragePool",
#                 ]:
#                     if (
#                         post_node.output[0] in self.no_need_to_quantize
#                     ):  # Conv-Relu/Conv-Concat的传递
#                         self.already_quantized.append(post_node.output[0])
#                         self.quantized_cosines.append(
#                             self.quantized_cosines[-1]
#                             if len(self.quantized_cosines)
#                             else 1.0
#                         )
#                         next_post_nodes = self.get_post_nodes(
#                             self.fp32_model, post_node.output[0]
#                         )
#                         for next_post_node in next_post_nodes:
#                             if next_post_node not in post_nodes:
#                                 post_nodes.append(next_post_node)
#                     else:  # 纯Relu
#                         self.add_tensor_to_next_to_quant(post_node.output[0])

#                 # 单输入，无权重，单输出，向后传递的节点
#                 elif post_node.op_type in [
#                     "Reshape",
#                     "Transpose",
#                     "MaxPool",
#                     "Resize",
#                     "InstanceNormalization",
#                 ]:
#                     if (
#                         post_node.output[0] in self.no_need_to_quantize
#                     ):  # Conv-Relu/Conv-Concat的传递
#                         self.already_quantized.append(post_node.output[0])
#                         self.quantized_cosines.append(
#                             self.quantized_cosines[-1]
#                             if len(self.quantized_cosines)
#                             else 1.0
#                         )
#                         next_post_nodes = self.get_post_nodes(
#                             self.fp32_model, post_node.output[0]
#                         )
#                         for next_post_node in next_post_nodes:
#                             if next_post_node not in post_nodes:
#                                 post_nodes.append(next_post_node)
#                     else:  # 假设前置节点已量化
#                         self.add_tensor_to_next_to_quant(post_node.output[0])

#                 # Conv层: Conv-Relu需要合并
#                 elif post_node.op_type in ["Conv"]:
#                     # 权重是否已经量化
#                     if post_node.input[1] in self.already_quantized:
#                         if (
#                             post_node.output[0] in self.no_need_to_quantize
#                         ):  # Conv-Relu/Conv-Concat的传递
#                             self.already_quantized.append(post_node.output[0])
#                             self.quantized_cosines.append(
#                                 self.quantized_cosines[-1]
#                                 if len(self.quantized_cosines)
#                                 else 1.0
#                             )
#                             next_post_nodes = self.get_post_nodes(
#                                 self.fp32_model, post_node.output[0]
#                             )
#                             for next_post_node in next_post_nodes:
#                                 if next_post_node not in post_nodes:
#                                     post_nodes.append(next_post_node)
#                         else:  # 纯Conv
#                             self.add_tensor_to_next_to_quant(
#                                 post_node.output[0])
#                     else:
#                         self.add_tensor_to_next_to_quant(post_node.input[1])

#                 # Add层: Add-Relu需要合并
#                 elif post_node.op_type in ["Add"]:
#                     # 输入是否均已量化
#                     all_inputs_quantized = True
#                     for input in post_node.input:
#                         if input not in self.already_quantized:
#                             all_inputs_quantized = False

#                     if all_inputs_quantized:
#                         if (
#                             post_node.output[0] in self.no_need_to_quantize
#                         ):  # Conv-Relu/Conv-Concat的传递
#                             self.already_quantized.append(post_node.output[0])
#                             self.quantized_cosines.append(
#                                 self.quantized_cosines[-1]
#                                 if len(self.quantized_cosines)
#                                 else 1.0
#                             )
#                             next_post_nodes = self.get_post_nodes(
#                                 self.fp32_model, post_node.output[0]
#                             )
#                             for next_post_node in next_post_nodes:
#                                 if next_post_node not in post_nodes:
#                                     post_nodes.append(next_post_node)
#                         else:
#                             self.add_tensor_to_next_to_quant(
#                                 post_node.output[0])
#                     else:
#                         # Add层还有输入未量化
#                         pass

#                 # 多输入，无权重，单输出，且输出需要量化的节点
#                 elif post_node.op_type in ["Mul", "Concat"]:
#                     # 输入是否均已量化
#                     all_inputs_quantized = True
#                     for input in post_node.input:
#                         print(
#                             "【节点{}的前置输入{} 是否完成量化{}".format(
#                                 post_node.name, input, input in self.already_quantized
#                             )
#                         )
#                         if input not in self.already_quantized:
#                             all_inputs_quantized = False

#                     if all_inputs_quantized:
#                         # 输出量化
#                         if (
#                             post_node.output[0] in self.no_need_to_quantize
#                         ):  # Conv-Relu/Conv-Concat的传递
#                             self.already_quantized.append(post_node.output[0])
#                             self.quantized_cosines.append(
#                                 self.quantized_cosines[-1]
#                                 if len(self.quantized_cosines)
#                                 else 1.0
#                             )
#                             next_post_nodes = self.get_post_nodes(
#                                 self.fp32_model, post_node.output[0]
#                             )
#                             for next_post_node in next_post_nodes:
#                                 if next_post_node not in post_nodes:
#                                     post_nodes.append(next_post_node)
#                         else:
#                             self.add_tensor_to_next_to_quant(
#                                 post_node.output[0])
#                     else:
#                         # Concat层还有输入未量化
#                         pass

#                 # Resize层，其它输入需要量化
#                 # elif post_node.op_type in ["Resize", "InstanceNormalization"]:
#                 #     # 权重是否已经量化
#                 #     weight_already_quantized = True
#                 #     for weight in post_node.input[1:]:
#                 #         if weight not in self.already_quantized:
#                 #             weight_already_quantized = False
#                 #             self.add_tensor_to_next_to_quant(weight)
#                 #         else:
#                 #             pass

#                 #     if weight_already_quantized:
#                 #         if (
#                 #             post_node.output[0] in self.no_need_to_quantize
#                 #         ):
#                 #             self.already_quantized.append(post_node.output[0])
#                 #             self.quantized_cosines.append(
#                 #                 self.quantized_cosines[-1]
#                 #                 if len(self.quantized_cosines)
#                 #                 else 1.0
#                 #             )
#                 #             next_post_nodes = self.get_post_nodes(
#                 #                 self.fp32_model, post_node.output[0]
#                 #             )
#                 #             for next_post_node in next_post_nodes:
#                 #                 if next_post_node not in post_nodes:
#                 #                     post_nodes.append(next_post_node)
#                 #         else:  # 纯Relu
#                 #             self.add_tensor_to_next_to_quant(post_node.output[0])
#                 #     else:
#                 #         #  还有权重未量化
#                 #         pass

#                 elif post_node.op_type in ["Split"]:
#                     for output in post_node.output:
#                         if output in self.no_need_to_quantize:
#                             self.already_quantized.append(output)
#                             self.quantized_cosines.append(
#                                 self.quantized_cosines[-1]
#                                 if len(self.quantized_cosines)
#                                 else 1.0
#                             )
#                             next_post_nodes = self.get_post_nodes(
#                                 self.fp32_model, post_node.output[0]
#                             )
#                             for next_post_node in next_post_nodes:
#                                 if next_post_node not in post_nodes:
#                                     post_nodes.append(next_post_node)
#                         else:
#                             self.add_tensor_to_next_to_quant(
#                                 post_node.output[0])
#                             # raise NameError("Split Outputs should not be quanted")
#                 else:
#                     raise NameError("未知节点: {}".format(post_node.name))
#                 print("\t 待判断的节点:", [n.name for n in post_nodes])
#                 print("\t 已量化的节点:", self.already_quantized[-10:])
#                 print("\t 下一步量化的节点:", self.next_to_quant)

#             cost_time = (time.time() - start_time) / 60.0
#             self.quantized_nodes_num += 1
#             if self.each_node_time is None:
#                 self.each_node_time = cost_time
#             else:
#                 self.each_node_time = self.each_node_time * 0.95 + 0.05 * cost_time

#             total_time = (
#                 self.each_node_time *
#                 self.total_nodes_num
#             )
#             rest_time = self.each_node_time * (self.total_nodes_num - self.quantized_nodes_num)
#             print(
#                 "【==】[{}/{}-{:6.2f}/{:6.2f} min]search scale finished for {}".format(
#                     self.quantized_nodes_num,
#                     self.total_nodes_num,
#                     rest_time,
#                     total_time,
#                     output_v,
#                 )
#             )
#             print("="*50)

#         import json
#         onnx.save(self.qdq_model, sys.argv[2])
#         with open(sys.argv[2] + "_scale_map.json", "w") as fw:
#             json.dump(self.scale_map, fw)


onnx_opt = OnnxOptimizer()
int8_model = onnx_opt.process()
onnx.save(int8_model, sys.argv[2])

