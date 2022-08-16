import torch


def export_torch_to_onnx(model, output_path, output_names=None, input_shape=(320, 192)):
    dummy_input = torch.autograd.Variable(
        torch.randn(1, 3, input_shape[1], input_shape[0])
    )  # N, C, H, W
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=True,
        keep_initializers_as_inputs=False,
        do_constant_folding=True,
        training=False,
        opset_version=11,
        output_names=output_names,
    )


if __name__ == "__main__":
    # load your model
    config = "exp01_baseline.yml"
    model_path = "epoch_300.pth"
    output_path = "model.onnx"
    model = build_model(config)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load(state_dict)
    # set input shape
    input_shape = (320, 192)  # W H
    # set output names, which will be used in model deployment
    output_names = ["s8_cls", "s8_reg", "s16_cls", "s16_reg"]

    export_torch_to_onnx(model, output_path, output_names, input_shape)
