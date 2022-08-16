from asyncore import read
from pickle import NONE
from matplotlib.pyplot import sca
import numpy as np
import sys
import onnx
import onnxruntime
import torch
from tqdm import tqdm
import glob
import cv2
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
from queue import Queue
from onnx import numpy_helper
from onnx import version_converter

print("import finished")


print("reading calibration data")


def read_image(path):
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


class OnnxOptimizer:
    def __init__(self, onnx_model_path=sys.argv[1], calibration_data=input_data):
        print("loading model")
        onnx_model = onnx.load(onnx_model_path)
        onnx_model = version_converter.convert_version(onnx_model, 13)
        onnx_model = self.remove_initializer_from_input(onnx_model)
        onnx_model = onnxoptimizer.optimize(onnx_model)
        onnx_model = shape_inference.infer_shapes(onnx_model)
        self.fp32_model = onnx_model
        self.qdq_model = deepcopy(self.fp32_model)

        self.input_names = [input.name for input in onnx_model.graph.input]
        self.inits_nodes = onnx_model.graph.initializer
        self.inits_names = [init.name for init in onnx_model.graph.initializer]
        self.output_names = [output.name for output in onnx_model.graph.output]

        self.next_to_quant = [self.input_names[0]]
        self.total_nodes_num = len(self.fp32_model.graph.node) + len(self.inits_names)
        self.already_quantized = []  # 记录已经量化过的tensor名称
        self.quantized_cosines = []
        self.no_need_to_quantize = self.input_no_quantization_tensors(
            self.fp32_model
        )  # 不需要量化的tensor名称

        print("reading input data")
        self.calibration_data = calibration_data

        print("getting fp32 outputs")
        self.fp32_outputs = self.infer_with_onnx(self.fp32_model, self.calibration_data)

        # 记录scale值
        self.scale_map = {}

    def input_no_quantization_tensors(self, onnx_model):
        ret_tensors = []
        for node in onnx_model.graph.node:
            # Concat的输入，不量化
            if node.op_type in ["Concat", "Split"]:
                for input in node.input:
                    ret_tensors.append(input)
            # # Split的输入，不量化
            # elif node.op_type in ["Split"]:
            #     for output in node.output:
            #         ret_tensors.append(output)
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

    def remove_initializer_from_input(self, model):
        if model.ir_version < 4:
            print(
                "Model with ir_version below 4 requires to include initilizer in graph input"
            )
            return

        inputs = model.graph.input
        name_to_input = {}
        for input in inputs:
            name_to_input[input.name] = input

        for initializer in model.graph.initializer:
            if initializer.name in name_to_input:
                inputs.remove(name_to_input[initializer.name])

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

    def add_node_to_next_to_quant(self, post_node):
        for output in post_node.output:
            self.add_tensor_to_next_to_quant(output)

    def add_tensor_to_next_to_quant(self, output):
        # 输出节点不再量化
        if output in self.output_names:
            return
        # 不需要量化的不再量化
        if output in self.no_need_to_quantize:
            return
        # 已经量化的不再量化
        if output in self.already_quantized:
            return
        # 已经准备量化的，不再量化
        if output in self.next_to_quant:
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

            qdq_outputs = self.infer_with_onnx(qdq_model, input_data)
            qdq_cosine = self.cosine_compute(self.fp32_outputs, qdq_outputs)
            print("     [search]", output_v, percentile, qdq_cosine)
            scales.append(percentile)
            performences.append(qdq_cosine)
            if best_cosine < qdq_cosine:
                best_scale = scale
                best_cosine = qdq_cosine
                best_model = qdq_model

            if best_cosine > 0.9999:
                break
        return best_cosine, best_model, best_scale

    def process(self):
        start_time = time.time()
        while len(self.next_to_quant) > 0:
            output_v = self.next_to_quant[0]
            self.next_to_quant = self.next_to_quant[1:]
            cur_node_fp32_output = self.get_fp32_outputs(output_v)
            scales = []
            performences = []
            print("searching for ", output_v)
            # 第一轮搜索
            best_cosine, best_model, best_scale = self.search_scale(
                output_v,
                cur_node_fp32_output,
                scales,
                performences,
                search_scales=[100, 99.995, 99.95, 99.5, 99.0],
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
                post_node = post_nodes[0]
                post_nodes = post_nodes[1:]

                # 首先判断当前节点是否已经可以开始进行量化
                ready = True
                for input in post_node.input:
                    if input in self.inits_names:
                        continue
                    elif (input not in self.no_need_to_quantize) and (
                        input not in self.already_quantized
                    ):
                        ready = False
                if not ready:
                    continue

                # 某些类型节点的输出，不需要进行量化，可以直接向后传递
                if post_node.op_type in ["Reshape", "Transpose", "Resize", "MaxPool"]:
                    for output in post_node.output:
                        self.no_need_to_quantize.append(output)
                        # 对每一个输出，找到其后续节点
                        next_post_nodes = self.get_post_nodes(self.fp32_model, output)
                        post_nodes.extend(next_post_nodes)
                    continue

                # Conv-Relu向后传递
                # Conv-Relu中Conv的输出，不量化
                elif post_node.op_type in ["Conv", "Add"]:
                    next_post_nodes = self.get_post_nodes(
                        self.fp32_model, post_node.output[0]
                    )
                    if len(next_post_nodes) == 1 and next_post_nodes[0].op_type in [
                        "Relu",
                        "Clip",
                    ]:
                        post_nodes.extend(next_post_nodes)
                        # continue #

                # 开始判断如何量化
                if post_node.op_type in ["Conv"]:
                    self.add_tensor_to_next_to_quant(post_node.input[1])
                    self.add_node_to_next_to_quant(post_node)
                else:
                    self.add_node_to_next_to_quant(post_node)
                print("already_quantized: ", self.already_quantized)
                print("no_need_to_quantize: ", self.no_need_to_quantize)
                print("next_to_quant: ", self.next_to_quant)

        import json

        onnx.save(self.qdq_model, sys.argv[2])
        with open(sys.argv[2] + "_scale_map.json", "w") as fw:
            json.dump(self.scale_map, fw)


onnx_opt = OnnxOptimizer()
onnx_opt.process()
