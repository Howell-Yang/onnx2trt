from copy import deepcopy
from matplotlib.pyplot import axis
from trt_utils import read_calib_cache
from data_loader import DataLoader
from onnx_model import OnnxModel
from queue import Queue
import numpy as np
from onnx import numpy_helper
import onnx
from time import time
from sklearn.metrics.pairwise import cosine_similarity

class QuantizaitonFilter:
    def __init__(self, model_path: str, calib_path: str):
        self.model_path = model_path
        self.data_loader = DataLoader()
        self.model = OnnxModel(model_path)
        self.calib_path = calib_path
        OnnxModel.quantize_weights(self.model.qdq_model)
        self.fp32_weight_name2tensor = {}
        for weight in self.model.fp32_model.graph.initializer:
            self.fp32_weight_name2tensor[weight.name] = weight

        self.int8_weight_name2tensor = {}
        for weight in self.model.qdq_model.graph.initializer:
            self.int8_weight_name2tensor[weight.name] = weight

    def is_conv_output(self, onnx_model, tensor_name):
        pre_node = None
        for node in onnx_model.graph.node:
            for output in node.output:
                if output == tensor_name:
                    pre_node = node
                    break

        if pre_node is None:
            return False

        if pre_node.op_type == "Conv":
            return True
        elif pre_node.op_type == "Relu":
            return self.is_conv_output(onnx_model, pre_node.input[0])
        elif pre_node.op_type == "Concat":
            ret = False
            for input in pre_node.input[0]:
                ret |= self.is_conv_output(onnx_model, input)
            return ret            

    def get_conv_tensors(self, onnx_model, act_scale_map):
        conv_out_scale_map = {}
        conv_out_to_bias = {}
        for node in onnx_model.graph.node:
            for tensor_name in node.output:
                if tensor_name not in act_scale_map:
                    continue
                else:
                    if node.op_type == "Conv":
                        if len(node.input) >= 3:  # with bias
                            conv_out_scale_map[tensor_name] = act_scale_map[tensor_name]
                            conv_out_to_bias[tensor_name] = node.input[2]
                    elif node.op_type == "Relu":
                        pre_nodes = OnnxModel.get_previous_nodes(onnx_model, node.input[0])
                        assert len(pre_nodes) == 1, "Relu should only have one input"
                        if pre_nodes[0].op_type == "Conv" and len(pre_nodes[0].input) >= 3:
                            conv_out_scale_map[node.input[0]] = act_scale_map[tensor_name]
                            conv_out_to_bias[node.input[0]] = pre_nodes[0].input[2]
                    elif node.op_type == "Concat":
                        for input in node.input:
                            pre_nodes = OnnxModel.get_previous_nodes(onnx_model, input)
                            assert len(pre_nodes) == 1, "each input shold corespond to one node"
                            if pre_nodes[0].op_type == "Conv" and len(pre_nodes[0].input) >= 3:
                                conv_out_scale_map[input] = act_scale_map[tensor_name]
                                conv_out_to_bias[input] = pre_nodes[0].input[2]
                            elif pre_nodes[0].op_type == "Relu":
                                nodes_before_relu = OnnxModel.get_previous_nodes(onnx_model, pre_nodes[0].input[0])
                                assert len(nodes_before_relu) == 1, "Relu should only have one input"
                                if nodes_before_relu[0].op_type == "Conv" and len(nodes_before_relu[0].input) >= 3:
                                    conv_out_scale_map[pre_nodes[0].input[0]] = act_scale_map[tensor_name]
                                    conv_out_to_bias[pre_nodes[0].input[0]] = nodes_before_relu[0].input[2]
        return conv_out_scale_map, conv_out_to_bias

    def eval_quantize(self, fp32_output, int8_output):
        sims = []
        diffs = []
        rel_diffs = []
        for fp32, int8 in zip(fp32_output, int8_output):
            fp32 = np.reshape(fp32, (1, -1))
            int8 = np.reshape(int8, (1, -1))
            sim = cosine_similarity(fp32, int8)
            diff = np.abs(fp32 - int8)
            rel_diff = diff / (np.abs(fp32) + 1e-8)
            sims.append(sim)
            diffs.append(np.median(diff))
            rel_diffs.append(np.median(rel_diff))
        return np.mean(sims), np.mean(diffs), np.mean(rel_diffs)

    def process(self):
        # Step 01. read input data
        input_data = self.data_loader.get_numpy_data(image_num=100)

        # Step 02. read calibration cache
        act_scale_map = read_calib_cache(self.calib_path)
        act_scale_map = {name: value for name, value in act_scale_map.items() if name in self.model.all_tensor_names}
        qdq_model = deepcopy(self.model.qdq_model)
        for tensor_name, scale_value in act_scale_map.items():
            OnnxModel.add_act_dqd_node(qdq_model, tensor_name, scale_value)
        onnx.save(qdq_model, self.model_path + "_qdq100.onnx")

        # Step 03. caculate snrs
        # fp32_outputs = OnnxModel.get_onnx_outputs(self.model.fp32_model, list(act_scale_map.keys()), input_data)
        # snrs = {}
        # for name, fp32_output in fp32_outputs.items():
        #     snrs[name] = self.caculate_snr(fp32_outputs[name], act_scale_map[name])

        return self.model.fp32_model, self.model.qdq_model


if __name__ == "__main__":
    import sys
    onnx_path = "/mapai/howellyang/code/onnx2trt/RMTNet_release20220609_mm2conv.optimized.rm_inits.onnx"
    calib_path = "/mapai/howellyang/code/onnx2trt/RMTNet_release20220609_mm2conv.optimized.trt_int8_with_531pics_calib_percentile595.calib_cache"
    BS = QuantizaitonFilter(onnx_path, calib_path)
    onnx_model, qdq_model = BS.process()
    # onnx.save(onnx_model,  "/mapai/howellyang/code/onnx2trt/RMTNet_release20220609_mm2conv.optimized.bias_correction_v1.onnx")
    # onnx.save(qdq_model,  "/mapai/howellyang/code/onnx2trt/RMTNet_release20220609_mm2conv.optimized.bias_correction_v1.qdq.onnx")