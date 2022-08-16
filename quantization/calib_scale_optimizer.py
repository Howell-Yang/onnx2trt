from asyncore import read
from mimetypes import init
import numpy as np
import sys
import onnx
import onnxruntime
import torch
import glob
import numpy as np
import onnxoptimizer
from torchvision import transforms
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity
import onnx
from copy import deepcopy
import time
from onnx import helper, shape_inference
from onnx import numpy_helper
from onnx import version_converter
import cv2

print("import finished")


print("reading calibration data")

def read_image_v1(path):
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
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    input_w = 960
    input_h = 480

    # for onnx inference
    mean = np.array(mean)
    std = np.array(std)

    # Load by OpenCV
    img = cv2.imread(path)
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (input_w, input_h))

    img = img.astype(np.float32)

    # Norm
    for i in range(3):
        img[..., i] = (img[..., i] - mean[i]) / std[i]

    # hwc -> nchw
    h, w, c = img.shape
    img = img.reshape((1, c, h ,w))

    return np.array(img)


def read_image_v3(path):
    mean_val = [103.53, 116.28, 123.675]
    std_val = [57.375, 57.12, 58.395]
    input_size = [768, 448]
    img_raw = cv2.imread(path)
    img = cv2.resize(img_raw, (input_size[0], input_size[1])).astype(np.float32)
    img -= mean_val
    img /= std_val
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img = np.expand_dims(img, axis=0)

    img = np.ascontiguousarray(img, dtype=np.float32)
    assert np.max(img) > 0
    return img


def convert_any_to_numpy(x, accepet_none: bool = True) -> np.ndarray:
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
            f"input value {x}({type(x)}) can not be converted as numpy type."
        )


calibration_files = glob.glob(
    os.path.join("/apdcephfs/private_howellyang/data/Calib1k5/", "*.jpg")
)[:10]
input_data = [convert_any_to_numpy(read_image(path)) for path in calibration_files]


options = onnxruntime.SessionOptions()
options.intra_op_num_threads = 1


class OnnxOptimizer:
    def __init__(self, onnx_model_path=sys.argv[1], calibration_data=input_data):
        print("loading model")
        onnx_model = onnx.load(onnx_model_path)
        onnx_model = version_converter.convert_version(onnx_model, 13)
        onnx_model = onnxoptimizer.optimize(onnx_model)
        onnx_model = self.remove_initializer_from_input(onnx_model)
        onnx_model = shape_inference.infer_shapes(onnx_model)
        self.fp32_model = deepcopy(onnx_model)
        self.qdq_model = deepcopy(self.fp32_model)

        self.input_names = [input.name for input in onnx_model.graph.input]
        self.inits_nodes = onnx_model.graph.initializer
        self.inits_names = [init.name for init in onnx_model.graph.initializer]
        self.output_names = [output.name for output in onnx_model.graph.output]

        self.constant_names = self.get_constant_names(self.fp32_model)
        self.next_to_quant = []
        self.already_quantized = []  # 记录已经量化过的tensor名称
        self.quantized_cosines = []
        self.prepare_constants(self.constant_names, self.already_quantized, self.next_to_quant)
        self.next_to_quant.extend(self.input_names)

        self.concat_outputs_with_shared_input = (
            self.get_concat_outputs_with_shared_input(self.fp32_model)
        )  # 不同的Concat节点，存在共享的输入，Concat输出不能量化

        self.no_need_to_quantize = self.input_no_quantization_tensors(
            self.fp32_model
        )  # 不需要量化的tensor名称

        self.total_nodes_num = (
            len(self.fp32_model.graph.node)
            + len(self.inits_names)
            - len(self.no_need_to_quantize)
        )
        print("reading input data")
        self.calibration_data = calibration_data

        print("getting fp32 outputs")
        # self.fp32_outputs = self.infer_with_onnx(self.fp32_model, self.calibration_data)

        # 记录scale值
        self.scale_map = {}

    def prepare_constants(self, constant_names, already_quantized, next_to_quant):
        for constant_name in constant_names:
            post_nodes = self.get_post_nodes(self.fp32_model, constant_name)
            # 如果constant是Resize层的输入，则需要量化
            if len(post_nodes) == 1 and post_nodes[0].op_type == "Resize":
                next_to_quant.append(constant_name)
            
            # 如果Constant是其它层的输入，则不需要量化
            else:
                already_quantized.append(constant_name)

    def get_concat_outputs_with_shared_input(self, onnx_model):
        ret = []
        # 记录Concat的Inputs到outputs的映射
        concat_input2output = {}
        for node in onnx_model.graph.node:
            if node.op_type in ["Concat"]:
                for input in node.input:
                    if input not in concat_input2output:
                        concat_input2output[input] = []
                    concat_input2output[input].append(node.output[0])

        # 如果有两个及以上的concat用了这个节点，则所有concat的输出都不能被量化
        for _, values in concat_input2output.items():
            if len(values) > 0:
                ret.extend(values)
            # 输入节点其实也不能被量化(不过这点在其他逻辑中已经实现过了)

        return ret

    def get_constant_names(self, fp32_model):
        ret = []
        for node in fp32_model.graph.node:
            if node.op_type == "Constant":
                ret.append(node.output[0])
        return ret

    def input_no_quantization_tensors(self, onnx_model):
        ret_tensors = []
        for node in onnx_model.graph.node:
            if node.op_type in ["Concat"]:
                # Concat的输入，不量化
                for input in node.input:
                    ret_tensors.append(input)

                # Concat的输出 ----> 两个Concat层存在公共输入，则这两个Concat的输出均不能量化
                if node.output[0] in self.concat_outputs_with_shared_input:
                    ret_tensors.append(node.output[0])

            # Split的输出，不量化(同一块内存，不可以量化)
            elif node.op_type in ["Split"]:
                for output in node.output:
                    ret_tensors.append(output)

            # Conv-Relu中Conv的输出，不量化
            elif node.op_type in ["Conv"]:
                post_nodes = self.get_post_nodes(self.fp32_model, node.output[0])
                if len(post_nodes) == 1 and post_nodes[0].op_type in ["Relu", "Clip"]:
                    ret_tensors.append(node.output[0])
            # Add-Relu中Conv的输出，不量化
            elif node.op_type in ["Add"]:
                post_nodes = self.get_post_nodes(self.fp32_model, node.output[0])
                if len(post_nodes) == 1 and post_nodes[0].op_type in ["Relu", "Clip"]:
                    ret_tensors.append(node.output[0])

        return ret_tensors

    def add_input_from_initializer(self, model: onnx.ModelProto):
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

        def add_const_value_infos_to_graph(graph: onnx.GraphProto):
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

    def infer_with_onnx(self, onnx_model="", SAMPLES=None, output_node=None):
        if isinstance(onnx_model, str):
            sess = onnxruntime.InferenceSession(
                onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
        else:
            if output_node is not None:
                intermediate_layer_value_info = onnx.helper.ValueInfoProto()
                intermediate_layer_value_info.name = output_node
                onnx_model.graph.output.append(intermediate_layer_value_info)

            sess = onnxruntime.InferenceSession(
                onnx._serialize(onnx_model),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

        input_name = sess.get_inputs()[0].name
        if output_node is None:
            onnx_output_names = [output.name for output in sess.get_outputs()]
        else:
            onnx_output_names = [output_node]
        samples = [convert_any_to_numpy(sample) for sample in SAMPLES]

        onnx_outpus_all = []
        # for sample in tqdm(samples, desc="Onnx is running..."):
        for sample in samples:
            onnx_outputs = sess.run(onnx_output_names, {input_name: sample})
            onnx_outputs_dict = {
                onnx_output_names[i]: onnx_outputs[i]
                for i in range(len(onnx_output_names))
            }
            onnx_outpus_all.append(deepcopy(onnx_outputs_dict))
        return onnx_outpus_all

    def cosine_compute(self, fp32_outputs, qdq_outputs):
        sims = {}
        for i in range(len(qdq_outputs)):
            for output_name, _ in qdq_outputs[i].items():
                trt_output = np.reshape(qdq_outputs[i][output_name], (1, -1))
                trt_fp32_output = np.reshape(fp32_outputs[i][output_name], (1, -1))
                cos_sim = cosine_similarity(trt_output, trt_fp32_output)
                if output_name not in sims:
                    sims[output_name] = []
                sims[output_name].append(cos_sim.ravel())

        mean_sims = []
        for key, value in sims.items():
            # print(key, np.mean(value), np.min(value))
            mean_sims.append(np.mean(value))
        return np.mean(mean_sims)

    # 设置act量化的scale和zp
    def create_act_initializer_tensor(
        self,
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT,
    ) -> onnx.TensorProto:

        # (TensorProto)
        initializer_tensor = onnx.helper.make_tensor(
            name=name,
            data_type=data_type,
            dims=(),  # [1],
            vals=[tensor_array],
        )

        return initializer_tensor

    # 设置weight量化的scale和zp
    def create_weight_initializer_tensor(
        self,
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT,
    ) -> onnx.TensorProto:

        # (TensorProto)
        initializer_tensor = onnx.helper.make_tensor(
            name=name,
            data_type=data_type,
            dims=np.shape(tensor_array),
            vals=tensor_array.flatten(),
        )
        return initializer_tensor

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

        scale_initializer_tensor = self.create_act_initializer_tensor(
            name=scale_name, tensor_array=scale, data_type=onnx.TensorProto.FLOAT
        )

        zp_initializer_tensor = self.create_act_initializer_tensor(
            name=zp_name, tensor_array=0, data_type=onnx.TensorProto.INT8
        )

        qdq_model.graph.initializer.append(scale_initializer_tensor)
        qdq_model.graph.initializer.append(zp_initializer_tensor)
        return qdq_model

    # 添加weight量化节点
    def add_weight_dqd_node(self, onnx_model, tensor_name, scale):
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
            op_type="QuantizeLinear",
            inputs=[q_input, scale_name, zp_name],
            outputs=[q_output],
            name=quant_node_name,
            axis=0,
        )

        dequant_node = onnx.helper.make_node(
            op_type="DequantizeLinear",
            inputs=[dq_input, scale_name, zp_name],
            outputs=[dq_output],
            name=dequant_node_name,
            axis=0,
        )

        # op_proto = helper.make_node(
        #     op_type=operation.type,
        #     inputs=[_.name for _ in operation.inputs],
        #     outputs=[_.name for _ in operation.outputs],
        #     name=operation.name,
        #     **attributes)

        for node in qdq_model.graph.node:
            for j in range(len(node.input)):
                if node.input[j] == tensor_name:
                    node.input[j] = dq_output

        qdq_model.graph.node.extend([qlinear_node, dequant_node])

        scale_initializer_tensor = self.create_weight_initializer_tensor(
            name=scale_name, tensor_array=scale, data_type=onnx.TensorProto.FLOAT
        )

        zp_initializer_tensor = self.create_weight_initializer_tensor(
            name=zp_name,
            tensor_array=np.zeros_like(scale, dtype=np.int8),
            data_type=onnx.TensorProto.INT8,
        )

        qdq_model.graph.initializer.append(scale_initializer_tensor)
        qdq_model.graph.initializer.append(zp_initializer_tensor)
        return qdq_model

    def get_fp32_outputs(self, output_v):
        # 使用fp32模型，获取当前节点的fp32输出
        if output_v == self.input_names:
            cur_node_output = self.calibration_data
        else:
            cur_node_output = self.infer_with_onnx(
                self.fp32_model, self.calibration_data, output_node=output_v
            )
            cur_node_output = [op[output_v] for op in cur_node_output]
        return cur_node_output

    def get_post_nodes(self, onnx_model, tensor_name):
        post_nodes = []
        for node in onnx_model.graph.node:
            for input_tensor in node.input:
                if input_tensor == tensor_name:
                    post_nodes.append(node)
                    break
        return post_nodes

    # def add_node_to_next_to_quant(self, post_node):
    #     for input in post_node.input:
    #         if input in self.constant_names:
    #             continue
    #         elif input in self.no_need_to_quantize:
    #             continue
    #         elif input in self.already_quantized:
    #             print("input of ", post_node.name , input , "not ready")
    #             return

    #     for output in post_node.output:
    #         self.add_tensor_to_next_to_quant(output)  # Add的输出 -- Relu的输出
    #     return

    def add_tensor_to_next_to_quant(self, output):
        # 已经准备量化的，不再量化
        if output in self.next_to_quant:
            return

        # 输出节点不再量化
        if output in self.output_names:
            return

        # 已经量化的不再量化
        if output in self.already_quantized:
            return

        self.next_to_quant.append(output)

    def search_scale(
        self,
        output_v,
        cur_node_fp32_output,
        scales,
        performences,
        search_scales=[100, 99.995, 99.95, 99.5, 99.0],
        best_cosine=-100.0,
        best_scale=None,
        best_model=None,
    ):
        for percentile in search_scales:
            if percentile in scales:
                continue
            if output_v in self.inits_names:  #  当前节点是权重
                weight = None
                for init_node in self.inits_nodes:
                    if init_node.name == output_v:
                        weight = numpy_helper.to_array(init_node)
                        break
                assert (
                    len(np.shape(weight)) == 4
                ), "Only Conv Weights Should Be Quantized"
                max_range = np.percentile(np.abs(weight), percentile, axis=(1, 2, 3))
                scale = max_range / 128.0
                print("Scale Shape:", np.shape(scale))
                qdq_model = self.add_weight_dqd_node(self.qdq_model, output_v, scale)
            else:
                # 进行act量化
                max_range = (
                    100.0
                    / 99.95
                    * np.percentile(np.abs(cur_node_fp32_output), percentile)
                )
                scale = max_range / 128.0
                qdq_model = self.add_act_dqd_node(self.qdq_model, output_v, scale)

            # qdq_outputs = self.infer_with_onnx(qdq_model, input_data)
            # qdq_cosine = self.cosine_compute(self.fp32_outputs, qdq_outputs)
            qdq_cosine = 0.9999
            print("     [search]", output_v, percentile, qdq_cosine)
            scales.append(percentile)
            performences.append(qdq_cosine)
            if best_cosine < qdq_cosine:
                best_scale = scale
                best_cosine = qdq_cosine
                best_model = qdq_model
        return best_cosine, best_model, best_scale

    def process(self):
        start_time = time.time()
        while len(self.next_to_quant) > 0:
            output_v = self.next_to_quant[-1]
            self.next_to_quant = self.next_to_quant[:-1]
            cur_node_fp32_output = [1, 1, 1]  # self.get_fp32_outputs(output_v)
            scales = []
            performences = []
            print("searching for ", output_v)
            # 第一轮搜索
            best_cosine, best_model, best_scale = self.search_scale(
                output_v,
                cur_node_fp32_output,
                scales,
                performences,
                search_scales=[100],  # 99.995, 99.95, 99.5, 99.0],
            )
            if best_cosine < 0.990:
                best_cosine, best_model, best_scale = self.search_scale(
                    output_v,
                    cur_node_fp32_output,
                    scales,
                    performences,
                    search_scales=[100, 97.5, 95],
                    best_cosine=best_cosine,
                    best_scale=best_scale,
                    best_model=best_model,
                )
                for rp in range(10):
                    search_idx = np.argsort(performences)
                    search_scales = [
                        (scales[search_idx[-3]] + scales[search_idx[-2]]) / 2.0,
                        (scales[search_idx[-2]] + scales[search_idx[-1]]) / 2.0,
                    ]
                    best_cosine, best_model, best_scale = self.search_scale(
                        output_v,
                        cur_node_fp32_output,
                        scales,
                        performences,
                        search_scales=search_scales,
                        best_cosine=best_cosine,
                        best_scale=best_scale,
                        best_model=best_model,
                    )

            if output_v not in self.inits_names:
                self.scale_map[output_v] = best_scale
            cost_time = (time.time() - start_time) / 60.0
            total_time = (
                cost_time / (1e-5 + len(self.already_quantized)) * self.total_nodes_num
            )
            rest_time = total_time - cost_time
            print(
                "[{}/{}-{:6.2f}/{:6.2f} min]search scale finished for {}".format(
                    len(self.already_quantized),
                    self.total_nodes_num,
                    cost_time,
                    rest_time,
                    output_v,
                )
            )

            print("[=quantized finished=] ", output_v, best_cosine)
            if (
                len(self.quantized_cosines) > 0
                and (self.quantized_cosines[-1] - best_cosine) > 0.005
            ):
                if output_v in self.inits_names:
                    self.qdq_model = deepcopy(best_model)
                    self.already_quantized.append(output_v)
                    self.quantized_cosines.append(best_cosine)
                    print(
                        "[ERROR] weight quant generate large quantize error, please consider LSQ methods"
                    )
                    print("        tensor name ", output_v)
                    print(
                        "        cosine drop: {} --> {}".format(
                            self.quantized_cosines[-2], self.quantized_cosines[-1]
                        )
                    )
                else:
                    # 如果当前节点量化带来的损失过大，则当前节点不进行量化
                    self.scale_map.pop(output_v)
                    self.no_need_to_quantize.append(output_v)
                    print(
                        "[WARN] act quant generate large quantize error, using fp32/fp16 for precision purpose"
                    )
                    print("        tensor name ", output_v)
                    print(
                        "        cosine drop: {} --> {}".format(
                            self.quantized_cosines[-2], self.quantized_cosines[-1]
                        )
                    )
            else:  # 当前节点可量化
                self.qdq_model = deepcopy(best_model)
                self.already_quantized.append(output_v)
                self.quantized_cosines.append(best_cosine)

            # 找到后续节点 post_nodes
            post_nodes = self.get_post_nodes(self.fp32_model, output_v)
            print("Post Nodes:", [pnd.name for pnd in post_nodes])

            # 从后续节点的权重和输出中，找可以开始量化的tensor
            while len(post_nodes) > 0:
                post_node = post_nodes[-1]
                post_nodes = post_nodes[:-1]
                print("【节点{}的前置{}输入已完成量化, 尝试量化这个节点".format(post_node.name, output_v))

                # 判断这个节点的输入和权重是否都已经量化完成

                # 单输入，无权重，单输出的节点，且没有权重
                if post_node.op_type in [
                    "Clip",
                    "Relu",
                    "Sigmoid",
                    "HardSigmoid",
                    "LeakyRelu",
                    "GlobalAveragePool",
                ]:
                    if (
                        post_node.output[0] in self.no_need_to_quantize
                    ):  # Conv-Relu/Conv-Concat的传递
                        self.already_quantized.append(post_node.output[0])
                        self.quantized_cosines.append(
                            self.quantized_cosines[-1]
                            if len(self.quantized_cosines)
                            else 1.0
                        )
                        next_post_nodes = self.get_post_nodes(
                            self.fp32_model, post_node.output[0]
                        )
                        for next_post_node in next_post_nodes:
                            if next_post_node not in post_nodes:
                                post_nodes.append(next_post_node)
                    else:  # 纯Relu
                        self.add_tensor_to_next_to_quant(post_node.output[0])

                # 单输入，无权重，单输出，向后传递的节点
                elif post_node.op_type in ["Reshape", "Transpose", "MaxPool"]:
                    if (
                        post_node.output[0] in self.no_need_to_quantize
                    ):  # Conv-Relu/Conv-Concat的传递
                        self.already_quantized.append(post_node.output[0])
                        self.quantized_cosines.append(
                            self.quantized_cosines[-1]
                            if len(self.quantized_cosines)
                            else 1.0
                        )
                        next_post_nodes = self.get_post_nodes(
                            self.fp32_model, post_node.output[0]
                        )
                        for next_post_node in next_post_nodes:
                            if next_post_node not in post_nodes:
                                post_nodes.append(next_post_node)
                    else:  # 假设前置节点已量化
                        self.add_tensor_to_next_to_quant(post_node.output[0])

                # Conv层: Conv-Relu需要合并
                elif post_node.op_type in ["Conv"]:
                    # 权重是否已经量化
                    if post_node.input[1] in self.already_quantized:
                        if (
                            post_node.output[0] in self.no_need_to_quantize
                        ):  # Conv-Relu/Conv-Concat的传递
                            self.already_quantized.append(post_node.output[0])
                            self.quantized_cosines.append(
                                self.quantized_cosines[-1]
                                if len(self.quantized_cosines)
                                else 1.0
                            )
                            next_post_nodes = self.get_post_nodes(
                                self.fp32_model, post_node.output[0]
                            )
                            for next_post_node in next_post_nodes:
                                if next_post_node not in post_nodes:
                                    post_nodes.append(next_post_node)
                        else:  # 纯Conv
                            self.add_tensor_to_next_to_quant(post_node.output[0])
                    else:
                        self.add_tensor_to_next_to_quant(post_node.input[1])

                # Add层: Add-Relu需要合并
                elif post_node.op_type in ["Add"]:
                    # 输入是否均已量化
                    all_inputs_quantized = True
                    for input in post_node.input:
                        if input not in self.already_quantized:
                            all_inputs_quantized = False

                    if all_inputs_quantized:
                        if (
                            post_node.output[0] in self.no_need_to_quantize
                        ):  # Conv-Relu/Conv-Concat的传递
                            self.already_quantized.append(post_node.output[0])
                            self.quantized_cosines.append(
                                self.quantized_cosines[-1]
                                if len(self.quantized_cosines)
                                else 1.0
                            )
                            next_post_nodes = self.get_post_nodes(
                                self.fp32_model, post_node.output[0]
                            )
                            for next_post_node in next_post_nodes:
                                if next_post_node not in post_nodes:
                                    post_nodes.append(next_post_node)
                        else:
                            self.add_tensor_to_next_to_quant(post_node.output[0])
                    else:
                        # Add层还有输入未量化
                        pass

                # 多输入，无权重，单输出，且输出需要量化的节点
                elif post_node.op_type in ["Mul", "Concat"]:
                    # 输入是否均已量化
                    all_inputs_quantized = True
                    for input in post_node.input:
                        print(
                            "【节点{}的前置输入{} 是否完成量化{}".format(
                                post_node.name, input, input in self.already_quantized
                            )
                        )
                        if input not in self.already_quantized:
                            all_inputs_quantized = False

                    if all_inputs_quantized:
                        # 输出量化
                        if (
                            post_node.output[0] in self.no_need_to_quantize
                        ):  # Conv-Relu/Conv-Concat的传递
                            self.already_quantized.append(post_node.output[0])
                            self.quantized_cosines.append(
                                self.quantized_cosines[-1]
                                if len(self.quantized_cosines)
                                else 1.0
                            )
                            next_post_nodes = self.get_post_nodes(
                                self.fp32_model, post_node.output[0]
                            )
                            for next_post_node in next_post_nodes:
                                if next_post_node not in post_nodes:
                                    post_nodes.append(next_post_node)
                        else:
                            self.add_tensor_to_next_to_quant(post_node.output[0])
                    else:
                        # Concat层还有输入未量化
                        pass

                # Resize层，其它输入需要量化
                elif post_node.op_type in ["Resize", "InstanceNormalization"]:
                    # 权重是否已经量化
                    weight_already_quantized = True
                    for weight in post_node.input[1:]:
                        if weight not in self.already_quantized:
                            weight_already_quantized = False
                            self.add_tensor_to_next_to_quant(weight)
                        else:
                            pass

                    if weight_already_quantized:
                        if (
                            post_node.output[0] in self.no_need_to_quantize
                        ):
                            self.already_quantized.append(post_node.output[0])
                            self.quantized_cosines.append(
                                self.quantized_cosines[-1]
                                if len(self.quantized_cosines)
                                else 1.0
                            )
                            next_post_nodes = self.get_post_nodes(
                                self.fp32_model, post_node.output[0]
                            )
                            for next_post_node in next_post_nodes:
                                if next_post_node not in post_nodes:
                                    post_nodes.append(next_post_node)
                        else:  # 纯Relu
                            self.add_tensor_to_next_to_quant(post_node.output[0])
                    else:
                        #  还有权重未量化
                        pass

                elif post_node.op_type in ["Split"]:
                    for output in post_node.output:
                        if output in self.no_need_to_quantize:  # Split的输出都不需要量化
                            self.already_quantized.append(output)
                            self.quantized_cosines.append(
                                self.quantized_cosines[-1]
                                if len(self.quantized_cosines)
                                else 1.0
                            )
                            next_post_nodes = self.get_post_nodes(
                                self.fp32_model, post_node.output[0]
                            )
                            for next_post_node in next_post_nodes:
                                if next_post_node not in post_nodes:
                                    post_nodes.append(next_post_node)
                        else:
                            raise NameError("Split Outputs should not be quanted")
                else:
                    raise NameError("未知节点: {}".format(post_node.name))
                print("待判断的节点:", [n.name for n in post_nodes])
                print("已量化的节点:", self.already_quantized[-10:])
                print("下一步量化的节点:", self.next_to_quant)

        import json

        onnx.save(self.qdq_model, sys.argv[2])
        with open(sys.argv[2] + "_scale_map.json", "w") as fw:
            json.dump(self.scale_map, fw)


onnx_opt = OnnxOptimizer()
onnx_opt.process()
