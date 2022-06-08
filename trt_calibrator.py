from __future__ import print_function
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # fix init error of cuda

# __all__ = [
#     "TRTPercentileCalibrator",
#     "TRTEntropyCalibrator",
#     "TRTMinMaxCalibrator",
# ]


class TRTEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, input_layers, stream, cache_file):
        super(TRTEntropyCalibrator, self).__init__()
        self.input_layers = input_layers

        # 数据读取的类, 等同于图片处理的回调
        self.stream = stream

        # 分配GPU
        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)

        # cache路径
        self.cache_file = cache_file

        # 重置校准集
        self.stream.reset()

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


class TRTMinMaxCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, input_layers, stream, cache_file):
        super(TRTMinMaxCalibrator, self).__init__()
        self.input_layers = input_layers

        # 数据读取的类, 等同于图片处理的回调
        self.stream = stream

        # 分配GP
        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)

        # cache路径
        self.cache_file = cache_file

        # 重置校准集
        self.stream.reset()

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


class TRTPercentileCalibrator(trt.IInt8LegacyCalibrator):
    def __init__(
        self, input_layers, stream, cache_file, quantile=0.9995, regression_cutoff=1.0
    ):
        super(TRTPercentileCalibrator, self).__init__()
        self.input_layers = input_layers
        self.stream = stream
        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
        self.cache_file = cache_file
        self.stream.reset()
        self.quantile = quantile
        self.regression_cutoff = regression_cutoff

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

    def get_quantile(self):
        return self.quantile

    def get_regression_cutoff(self):
        return self.regression_cutoff

    def read_histogram_cache(self, length):
        return None

    def write_histogram_cache(self, ptr, length):
        return None
