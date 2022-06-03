from __future__ import print_function
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # fix init error of cuda
from PIL import Image
from torchvision import transforms
import numpy as np
import glob

__all__ = [
    "ImageBatchStream",
    "TRTPercentileCalibrator",
    "TRTEntropyCalibrator",
    "TRTMinMaxCalibrator",
]


class ImageBatchStream:
    def __init__(
        self,
        calibration_files,
        WIDTH,
        HEIGHT,
        CHANNEL=3,
        batch_size=1,
        pixel_type="RGB",
        means=(0.0, 0.0, 0.0),
        stds=(1.0, 1.0, 1.0),
        channel_order="NCHW",
    ):
        self.batch_size = batch_size
        self.max_batches = (len(calibration_files) // batch_size) + (
            1 if (len(calibration_files) % batch_size) else 0
        )
        self.files = calibration_files

        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.CHANNEL = CHANNEL
        self.channel_order = channel_order

        self.means = means
        self.stds = stds

        self.pixel_type = pixel_type
        self.calibration_data = np.zeros(
            (batch_size, CHANNEL, HEIGHT, WIDTH), dtype=np.float32
        )
        self.batch = 0

    @staticmethod
    def read_image(
        path, WIDTH, HEIGHT, CHANNEL, means, scales, channel_order, pixel_type
    ):
        img = Image.open(path).convert("RGB").resize((WIDTH, HEIGHT), Image.BICUBIC)
        img = np.array(img, dtype=np.float32, order="C")

        # RGB vs BGR
        if channel_order == "BGR":
            img = img[:, :, ::-1]

        if np.max(means) < 1.0 and np.max(scales) < 1.0:
            means = np.array(means) * 255.0
            scales = np.array(scales) * 255.0

        for i in range(CHANNEL):
            img[:, :, i] = (img[:, :, i] - means[i]) / scales[i]

        # NCHW vs NHWC
        if pixel_type == "NCHW":
            img = img.transpose(2, 0, 1)  # HWC --> CHW

        img = np.expand_dims(img, axis=0)
        return img

    def reset(self):
        self.batch = 0

    def next_batch(self):
        if self.batch < self.max_batches:
            imgs = []
            files_for_batch = self.files[
                self.batch_size * self.batch : self.batch_size * (self.batch + 1)
            ]

            for f in files_for_batch:
                img = ImageBatchStream.read_image(
                    f,
                    self.WIDTH,
                    self.HEIGHT,
                    self.CHANNEL,
                    self.means,
                    self.stds,
                    self.channel_order,
                    self.pixel_type,
                )
                imgs.append(img)
            for i in range(len(imgs)):
                self.calibration_data[i] = imgs[i]
            self.batch += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])


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
