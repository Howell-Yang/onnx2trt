import torch
# 加载你的模型
config = "exp01_baseline.yml"
model_path = "epoch_300.pth"
output_path = "model.onnx"
model = build_model(config)
state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
model.load(state_dict)


# 设置输入的大小
input_shape = (320, 192)  # W H
dummy_input = torch.autograd.Variable(torch.randn(1, 3, input_shape[1],input_shape[0]))  # N, C, H, W

# 设置输出节点名称，便于后续部署
output_names = ["s8_cls", "s8_reg", "s16_cls", "s16_reg"]
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