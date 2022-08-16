from __future__ import print_function
import os
import pycuda.autoinit # fix init error of cuda
import tensorrt as trt
import pycuda.driver as cuda
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import glob


class PythonEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """ Int8 양자화를 위해 Calibrate를 진행하는 클래스 """

    def __init__(self, input_layers, stream, cache_file):
        super(PythonEntropyCalibrator, self).__init__()

        # Tensor RT에 지정될 Input 레이어 이름 설정
        self.input_layers = input_layers

        # Calib 이미지 배치 스트림 저장
        self.stream = stream

        # 데이터 GPU 할당
        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)

        # 캐시파일 경로 저장
        self.cache_file = cache_file

        # 현재 스트림 리셋
        stream.reset()

    def get_batch_size(self):
        """ 배치사이즈 반환 메소드 """
        return self.stream.batch_size

    def get_batch(self, names):
        try:
            print("=====get_batch====", flush=True)
            batch = self.stream.next_batch()
            if not batch.size:
                print("xxxxxx batch.size = None xxxxxxxx", flush=True)
                return None

            cuda.memcpy_htod(self.d_input, batch)
            return [int(self.d_input)]

        except StopIteration:
            print("xxxxxx Finished get_batch xxxxxxxx", flush=True)
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        # cache = ctypes.c_char_p(int(ptr))
        with open(self.cache_file, 'wb') as f:
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

class ImageBatchStream():
    """ 양자화 과정에서 사용되는 이미지 배치 스트림 """

    def __init__(self, batch_size, calibration_files, WIDTH, HEIGHT, CHANNEL):
        # 배치사이즈 결정
        self.batch_size = batch_size
        self.max_batches = (len(calibration_files) // batch_size) + \
            (1 if (len(calibration_files) % batch_size)
             else 0)

        # 파일 목록을 변수에 저장
        self.files = calibration_files

        # 변수 초기화
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.CHANNEL = CHANNEL
        self.calibration_data = np.zeros(
            (batch_size, CHANNEL, HEIGHT, WIDTH), dtype=np.float32)
        self.batch = 0
        print("===ImageBatchStream====", calibration_files)
        print("===max_batches====", batch_size, self.max_batches)

    @staticmethod
    def read_image(path, WIDTH, HEIGHT, CHANNEL):
        _img_transforms = transforms.Compose(
            [
                transforms.Resize((384, 768)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        img = Image.open(path).convert("RGB")
        img_w, img_h = img.size[0], img.size[1]
        img = _img_transforms(img)
        img = img.unsqueeze(0)
        return np.array(img)

    def reset(self):
        self.batch = 0

    def next_batch(self):
        if self.batch < self.max_batches:
            imgs = []
            files_for_batch = self.files[self.batch_size * self.batch:
                                         self.batch_size * (self.batch + 1)]

            for f in files_for_batch:
                img = ImageBatchStream.read_image(
                    f, self.WIDTH, self.HEIGHT, self.CHANNEL)
                imgs.append(img)
                print("[ImageBatchStream] Processing ", f, np.min(img), np.max(img) ,flush=True)
            for i in range(len(imgs)):
                self.calibration_data[i] = imgs[i]
            self.batch += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            print("===ERROR====")
            return np.array([])


class MyLogger(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self)

    def log(self, severity, msg):
        print("[TRT]-{}".format(severity), msg)


if __name__ == "__main__":
    import os
    calibration_files = glob.glob(os.path.join("/apdcephfs/private_howellyang/data/Calib1k5/", '*.jpg'))

    # 测试不同的校准算法, 并计算int8模型与fp16模型之间的相似度
    onnx_path = "/apdcephfs/private_howellyang/road_service_app/road-service/road_service/engine/mod_road_multi_tasks/model/RMTNet_release20220609_v2.opt.onnx"
    batchstream = ImageBatchStream(1, calibration_files, 768, 384, 3)

    # ======================================== 1 ===================================
    trt_path = onnx_path.replace(".onnx", ".trt_int8_with_{}pics_calib_percentile395.trtmodel".format(len(calibration_files)))
    calib_save_path = onnx_path.replace(".onnx", ".trt_int8_with_{}pics_calib_percentile395.calib_cache".format(len(calibration_files)))
    calib = TRTPercentileCalibrator(["input.1"], batchstream, calib_save_path)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    TRT_LOGGER = trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network((EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(onnx_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                exit(-1)
        print('Completed parsing of ONNX file')

        print('Network inputs:')
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            print(tensor.name, trt.nptype(tensor.dtype), tensor.shape)

        config = builder.create_builder_config()
        config.max_workspace_size = 163840 << 20 # 16GB
        print(dir(config))
        config.avg_timing_iterations = 5
        if calib:
            # config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calib
        else:
            builder.fp16_mode = True

        print('Building an engine from file {}; this may take a while...'.format(onnx_path))
        engine = builder.build_engine(network, config)
        print("Completed creating Engine. Writing file to: {}".format(onnx_path + "trt_int8_calib"))
        with open(trt_path, "wb") as f:
            f.write(engine.serialize())

    # ======================================== 2 ===================================
    import os
    calibration_files = glob.glob(os.path.join("/apdcephfs/private_howellyang/data/Calib1k5/", '*.jpg'))
    trt_path = onnx_path.replace(".onnx", ".trt_int8_with_{}pics_calib_entropy.trtmodel".format(len(calibration_files)))
    calib_save_path = onnx_path.replace(".onnx", ".trt_int8_with_{}pics_calib_entropy.calib_cache".format(len(calibration_files)))
    calib = PythonEntropyCalibrator(["input.1"], batchstream, calib_save_path)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    TRT_LOGGER = trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network((EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(onnx_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                exit(-1)
        print('Completed parsing of ONNX file')

        print('Network inputs:')
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            print(tensor.name, trt.nptype(tensor.dtype), tensor.shape)

        config = builder.create_builder_config()
        config.max_workspace_size = 163840 << 20 # 16GB
        print(dir(config))
        config.avg_timing_iterations = 5
        if calib:
            # config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calib
        else:
            builder.fp16_mode = True

        print('Building an engine from file {}; this may take a while...'.format(onnx_path))
        engine = builder.build_engine(network, config)
        print("Completed creating Engine. Writing file to: {}".format(onnx_path + "trt_int8_calib"))
        with open(trt_path, "wb") as f:
            f.write(engine.serialize())

    # ======================================== 3 ===================================
    trt_path = onnx_path.replace(".onnx", ".trt_int8_with_{}pics_calib_percentile295.trtmodel".format(len(calibration_files)))
    calib_save_path = onnx_path.replace(".onnx", ".trt_int8_with_{}pics_calib_percentile295.calib_cache".format(len(calibration_files)))
    calib = TRTPercentileCalibrator(["input.1"], batchstream, calib_save_path, quantile=0.995)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    TRT_LOGGER = trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network((EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(onnx_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                exit(-1)
        print('Completed parsing of ONNX file')

        print('Network inputs:')
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            print(tensor.name, trt.nptype(tensor.dtype), tensor.shape)

        config = builder.create_builder_config()
        config.max_workspace_size = 163840 << 20 # 16GB
        print(dir(config))
        config.avg_timing_iterations = 5
        if calib:
            # config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calib
        else:
            builder.fp16_mode = True

        print('Building an engine from file {}; this may take a while...'.format(onnx_path))
        engine = builder.build_engine(network, config)
        print("Completed creating Engine. Writing file to: {}".format(onnx_path + "trt_int8_calib"))
        with open(trt_path, "wb") as f:
            f.write(engine.serialize())

    # ======================================== 4 ===================================
    trt_path = onnx_path.replace(".onnx", ".trt_int8_with_{}pics_calib_percentile495.trtmodel".format(len(calibration_files)))
    calib_save_path = onnx_path.replace(".onnx", ".trt_int8_with_{}pics_calib_percentile495.calib_cache".format(len(calibration_files)))
    calib = TRTPercentileCalibrator(["input.1"], batchstream, calib_save_path, quantile=0.99995)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    TRT_LOGGER = trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network((EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(onnx_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                exit(-1)
        print('Completed parsing of ONNX file')

        print('Network inputs:')
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            print(tensor.name, trt.nptype(tensor.dtype), tensor.shape)

        config = builder.create_builder_config()
        config.max_workspace_size = 163840 << 20 # 16GB
        print(dir(config))
        config.avg_timing_iterations = 5
        if calib:
            # config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calib
        else:
            builder.fp16_mode = True

        print('Building an engine from file {}; this may take a while...'.format(onnx_path))
        engine = builder.build_engine(network, config)
        print("Completed creating Engine. Writing file to: {}".format(onnx_path + "trt_int8_calib"))
        with open(trt_path, "wb") as f:
            f.write(engine.serialize())
