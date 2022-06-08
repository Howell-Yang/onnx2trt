# coding: utf-8

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # fix init error of cuda
from google.protobuf.json_format import MessageToDict
import onnx
import onnxruntime
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from trt_calibrator import (
    TRTEntropyCalibrator,
    TRTMinMaxCalibrator,
    TRTPercentileCalibrator,
)
from onnx_calibrator import ONNXCalibrator
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

def create_image_stream(
    calib_dir, input_shapes, means, stds, pixel_type, channel_order
):
    calibration_files = glob.glob(os.path.join(calib_dir, "*jpg"))
    calibration_files += glob.glob(os.path.join(calib_dir, "*png"))
    channel = input_shapes[1]
    height = input_shapes[2]
    width = input_shapes[3]
    means = np.array(list(map(float, means.split(","))))
    stds = np.array(list(map(float, stds.split(","))))
    image_stream = ImageBatchStream(
        calibration_files,
        width,
        height,
        pixel_type=pixel_type,
        means=means,
        stds=stds,
        channel_order=channel_order,
    )
    return image_stream


def create_calibrator(image_stream, input_names, trt_calib_cache, calib_algo, onnx_model_path = None):
    CALIB_ALGO_MAP = {
        "TRTEntropy": TRTEntropyCalibrator,
        "TRTMinMax": TRTMinMaxCalibrator,
        "TRTPercentile": TRTPercentileCalibrator,
    }
    if calib_algo in CALIB_ALGO_MAP:
        CalibratorType = CALIB_ALGO_MAP[calib_algo]
        calibrator = CalibratorType(input_names, image_stream, trt_calib_cache)
    else:
        assert onnx_model_path is not None, "onnx model path must be provided for Onnx Calibrator"
        CalibratorType = ONNXCalibrator(input_names, image_stream, trt_calib_cache, calib_algo, onnx_model_path)
    return calibrator


def create_tensorrt_engine(onnx_path, engine_type, calibrator=None):
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # EXPLICIT_PRECISION = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)
    TRT_LOGGER = trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        (EXPLICIT_BATCH)
    ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        print("[ONNX2TRT] INFO: Beginning ONNX file parsing")
        with open(onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                print("[ONNX2TRT] ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                exit(-1)

        print("[ONNX2TRT] INFO: Network Configuration")
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            print(
                "[ONNX2TRT] INFO: Input{}".format(i),
                tensor.name,
                trt.nptype(tensor.dtype),
                tensor.shape,
            )

        config = builder.create_builder_config()
        config.max_workspace_size = 163840 << 20  # 16GB
        config.avg_timing_iterations = 5

        if engine_type == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
            builder.fp16_mode = True
        elif engine_type == "int8":
            assert calibrator is not None, "with int8 mode, calibrator must be set"
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calibrator
        elif engine_type == "best":
            assert calibrator is not None, "with int8 mode, calibrator must be set"
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calibrator
        else:
            print(
                "[ONNX2TRT] WARNING: using FP32 mode directly, add fp16 precision may increate performance"
            )

        print(
            "[ONNX2TRT] INFO: Building an engine from file {}; this may take a while...".format(
                onnx_path
            )
        )
        engine = builder.build_engine(network, config)
    return engine


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    output_names = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        if not isinstance(binding, str):
            binding = engine.get_binding_name(binding)

        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            output_names.append(binding)
    return inputs, outputs, bindings, stream, output_names


def do_inference_v2(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]


def evaluate_engine(onnx_path, engine, image_stream):
    onnx_model = onnx.load_model(onnx_path)
    sess = onnxruntime.InferenceSession(onnx_model.SerializeToString())
    inputs_memo, outputs_memo, bindings, stream, trt_output_names = allocate_buffers(
        engine
    )
    context = engine.create_execution_context()

    # sess.set_providers(['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    onnx_output_names = [output.name for output in sess.get_outputs()]
    image_stream.reset()
    image_data = image_stream.next_batch()
    cos_similarity = []
    infer_time = []
    while len(image_data) > 0:
        # onnx infer
        onnx_outputs = sess.run(onnx_output_names, {input_name: image_data})
        onnx_outputs = {
            onnx_output_names[i]: onnx_outputs[i] for i in range(len(onnx_output_names))
        }

        # trt infer
        start = time.time()
        np.copyto(inputs_memo[0].host, image_data)
        trt_outputs = do_inference_v2(
            context,
            bindings=bindings,
            inputs=inputs_memo,
            outputs=outputs_memo,
            stream=stream,
        )
        trt_outputs = {
            trt_output_names[i]: trt_outputs[i] for i in range(len(trt_output_names))
        }
        end = time.time()

        cos_sims = []
        for name, value in trt_outputs.items():
            trt_output = np.reshape(value, (1, -1))
            onnx_output = np.reshape(onnx_outputs[name], (1, -1))
            cos_sim = cosine_similarity(trt_output, onnx_output)
            cos_sims.append(cos_sim)
        # cosine, runtime
        infer_time.append(end - start)
        cos_similarity.append(np.mean(cos_sims))
        image_data = image_stream.next_batch()

    return np.mean(cos_similarity), np.mean(infer_time)

