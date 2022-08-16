# coding: utf-8
import abc
import json
import numpy as np
import tensorrt as trt
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # fix init error of cuda
import os
import onnx
import struct
from onnxruntime.quantization.calibrate import (
    CalibrationDataReader,
    MinMaxCalibrater,
    EntropyCalibrater,
    PercentileCalibrater,
)

# 使用onnx的quantize tools生成每个节点的scales和zero point
# 并转换为tensorRT可用的calibration cache file
# 后续需要用tensorrt模型转换工具生成trt engine
class ONNXDataReader(CalibrationDataReader):
    def __init__(self, input_name, image_stream, max_iter_num=None):
        super(ONNXDataReader).__init__()
        self.input_name = input_name
        self.image_stream = image_stream
        self.max_iter_num = max_iter_num
        self.iter_num = 0

    def get_next(self) -> dict:
        self.iter_num += 1
        if self.iter_num > self.max_iter_num:
            return None
        batch = self.image_stream.next_batch()
        if not batch.size:
            return None
        """generate the input data dict for ONNXinferenceSession run"""
        return {
            self.input_name: batch,
            # "image_shape": np.asarray([[self.image_stream.WIDTH, self.image_stream.HEIGHT]], dtype=np.float32),
        }


class ONNXCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, input_layers, stream, cache_file, calib_algo, onnx_model_path):
        super(ONNXCalibrator, self).__init__()
        self.input_layers = input_layers

        # 数据读取的类, 等同于图片处理的回调
        self.stream = stream

        # 分配GPU
        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)

        # cache路径
        self.cache_file = cache_file

        # 重置校准集
        self.stream.reset()

        # 使用onnx的calibrator来统计每个节点的dynamic range
        calibrator = self.create_calibrator(calib_algo, onnx_model_path)
        # calibrator.set_execution_providers(
        #     ["CUDAExecutionProvider", "CPUExecutionProvider"]
        # )
        calibrator.set_execution_providers(
            ["CPUExecutionProvider"]
        )
        each_iter_num = 1
        for i in range(self.stream.max_batches // each_iter_num):
            data_reader = ONNXDataReader(
                self.input_layers[0], self.stream, each_iter_num
            )
            calibrator.collect_data(data_reader)
        self.write_calibration_table(calibrator.compute_range(), self.cache_file)

    @staticmethod
    def write_calibration_table(calibration_cache, save_path):
        """
        Helper function to write calibration table to files.
        """
        with open(save_path + "_calib_cache.json", "w") as file:
            file.write(
                json.dumps(calibration_cache)
            )  # use `json.loads` to do the reverse

        # write plain text: tensorRT需要对结果做转换
        # TRT-8400-EntropyCalibration2
        # input.1: 3ca94044
        # 9131: 3cf4f8d5
        # 加密 hex(struct.unpack('<I', struct.pack('<f', f))[0])
        # 解析 struct.unpack('!f', bytes.fromhex('41973333'))[0]
        with open(save_path, "w") as file:
            file.write("TRT-8400-EntropyCalibration2\n")
            for key in sorted(calibration_cache.keys()):
                value = calibration_cache[key]
                scale = max(abs(value[0]), abs(value[1]))
                scale_hex = hex(struct.unpack("<I", struct.pack("<f", scale))[0])
                s = key + ": " + str(scale_hex).lstrip("0x")
                file.write(s)
                file.write("\n")

    @staticmethod
    def create_calibrator(calib_algo, onnx_model_path):
        augmented_model_path = onnx_model_path.replace(".onnx", "_calib.onnx")
        if calib_algo == "ONNXMinMax":
            # default settings for min-max algorithm
            # symmetric = True  # tensorRT使用的是对称量化
            # moving_average = True
            # averaging_constant = 0.01
            return MinMaxCalibrater(
                onnx_model_path,
                op_types_to_calibrate=[],
                augmented_model_path=augmented_model_path,
                # use_external_data_format=False,
                # symmetric=symmetric,
                # moving_average=moving_average,
                # averaging_constant=averaging_constant,
            )
        elif calib_algo == "ONNXEntropy":
            # default settings for entropy algorithm
            # num_bins = 128
            num_quantized_bins = 128
            # symmetric = True
            return EntropyCalibrater(
                onnx_model_path,
                op_types_to_calibrate=[],
                augmented_model_path=augmented_model_path,
                # use_external_data_format=False,
                # symmetric=symmetric,
                # num_bins=num_bins,
                num_quantized_bins=num_quantized_bins,
            )
        elif calib_algo == "ONNXPercentile":
            # default settings for percentile algorithm
            num_quantized_bins = 2048
            percentile = 99.95
            # symmetric = True
            return PercentileCalibrater(
                onnx_model_path,
                op_types_to_calibrate=[],
                augmented_model_path=augmented_model_path,
                # use_external_data_format=False,
                # symmetric=symmetric,
                num_quantized_bins=num_quantized_bins,
                percentile=percentile,
            )

        raise ValueError("Unsupported calibration method {}".format(calib_algo))

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, names):
        try:
            batch = self.stream.next_batch()
            if not batch.size:
                return None
            cuda.memcpy_htod(self.d_input, batch)
            return [int(self.d_input)]
        except StopIteration:
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        else:
            return None

    def write_calibration_cache(self, cache):
        # cache = ctypes.c_char_p(int(ptr))
        with open(self.cache_file, "wb") as f:
            f.write(cache)
