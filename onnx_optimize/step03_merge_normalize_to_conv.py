import onnx


def merge_normalize_to_conv(onnx_model):
    # similar to conv-bn merge
    