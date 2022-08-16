import onnx


def extract_sub_graph(input_path, output_path, input_names=None, output_names=None):
    onnx.utils.extract_model(input_path, output_path, input_names, output_names)


if __name__ == "__main__":
    import sys

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    input_names = ["input.1"]
    output_names = ["1080"]
    extract_sub_graph(input_path, output_path, input_names, output_names)
