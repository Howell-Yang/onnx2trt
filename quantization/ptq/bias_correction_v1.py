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

class BiasCorrection:
    def __init__(self, model_path: str, calib_path: str):
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
        conv_out_scale_map, conv_out_to_bias = self.get_conv_tensors(self.model.fp32_model, act_scale_map)
        fp32_outputs = OnnxModel.get_onnx_outputs(self.model.fp32_model, list(conv_out_scale_map.keys()), input_data)

        # Step 03. bias correction
        next_to_process = Queue()
        for name in self.model.input_names:
            if name not in self.model.weight_names:
                next_to_process.put(name)
        already_processed = set()
        for name in self.model.weight_names:
            already_processed.add(name)

        start_time = time()
        total_num = len(conv_out_scale_map)
        cur_num = 0
        while not next_to_process.empty():  # width first algorithm
            tensor_name = next_to_process.get()
            already_processed.add(tensor_name)
            if tensor_name in conv_out_scale_map:
                fp32_output = fp32_outputs[tensor_name]
                int8_output = OnnxModel.get_onnx_outputs(self.model.qdq_model, [tensor_name], input_data)[tensor_name]
                conv_bias_name = conv_out_to_bias[tensor_name]
                fp32_channel_wise_mean = np.mean(fp32_output, axis=0)
                int8_channel_wise_mean = np.mean(int8_output, axis=0)
                bias_correction = fp32_channel_wise_mean - int8_channel_wise_mean
                fp32_bias_tensor = self.fp32_weight_name2tensor[conv_bias_name]
                fp32_bias_value = numpy_helper.to_array(fp32_bias_tensor) # out_ch, in_ch, ker, ker
                fp32_bias_value = np.array(fp32_bias_value + bias_correction).astype(np.float32)
                fp32_bias_tensor.ClearField("float_data")
                fp32_bias_tensor.ClearField("int32_data")
                fp32_bias_tensor.ClearField("int64_data")
                fp32_bias_tensor.raw_data = fp32_bias_value.tobytes()

                int8_bias_tensor = self.int8_weight_name2tensor[conv_bias_name]
                int8_bias_value = numpy_helper.to_array(int8_bias_tensor) # out_ch, in_ch, ker, ker
                int8_bias_value = np.array(int8_bias_value + bias_correction).astype(np.float32)
                int8_bias_tensor.ClearField("float_data")
                int8_bias_tensor.ClearField("int32_data")
                int8_bias_tensor.ClearField("int64_data")
                int8_bias_tensor.raw_data = int8_bias_value.tobytes()

                scale_value = conv_out_scale_map[tensor_name]

                sim, diff, rel_diff = self.eval_quantize(fp32_output, int8_output)

                cur_num += 1
                cost_time = (time() - start_time)/60.0
                totol_time = cost_time/cur_num * total_num
                tta_time = cost_time/cur_num * (total_num - cur_num)
                print("====={:6.2f} minutes used;  {:6.2f} minutes left; totally takes {:6.2f} minutes; =====".format(cost_time, tta_time, totol_time))
                print("{}: {}".format(tensor_name, bias_correction[:5]))
                print("{} {}: sim = {}, abs_diff = {}, rel_diff = {}".format(tensor_name, scale_value, sim, diff, rel_diff))
                OnnxModel.add_act_dqd_node(self.model.qdq_model, tensor_name, scale_value)

            post_nodes = OnnxModel.get_post_nodes(self.model.fp32_model, tensor_name)
            for node in post_nodes:
                node_is_ready = True
                for input in node.input:
                    if input not in already_processed:
                        node_is_ready = False

                if node_is_ready:
                    for output in node.output:
                        if output not in already_processed:
                            next_to_process.put(output)

        return self.model.fp32_model, self.model.qdq_model


if __name__ == "__main__":
    import sys
    onnx_path = "/mapai/howellyang/code/onnx2trt/RMTNet_release20220609_mm2conv.optimized.rm_inits.onnx"
    calib_path = "/mapai/howellyang/code/onnx2trt/RMTNet_release20220609_mm2conv.optimized.trt_int8_with_531pics_calib_percentile595.calib_cache"
    BS = BiasCorrection(onnx_path, calib_path)
    onnx_model, qdq_model = BS.process()
    onnx.save(onnx_model,  "/mapai/howellyang/code/onnx2trt/RMTNet_release20220609_mm2conv.optimized.bias_correction_v1.onnx")
    onnx.save(qdq_model,  "/mapai/howellyang/code/onnx2trt/RMTNet_release20220609_mm2conv.optimized.bias_correction_v1.qdq.onnx")