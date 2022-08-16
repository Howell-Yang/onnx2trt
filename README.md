# ONNX2TRT: 端上模型部署[整理中]


## 1. 概述

模型的压缩(蒸馏、剪枝、量化)和部署，是模型在自动驾驶和物联网产业落地中的重要步骤。端上的设备

在实际工作过程中，我们遇到了很多的困难: 文档缺失、依赖库冲突、算子不支持、精度差、速度慢等。

因此，我将我在实际工作过程中的一些经验，整理成文档记录在这里，供其它开发者参考。同时，我会将过程中用到的一些脚本，整理成一些独立的工具脚本，方便大家使用。

<br>

## 2. 模型部署流程

模型部署的一般步骤为:
- 模型导出onnx
- onnx模型结构优化
- 模型量化，构建tensorRT的engine
- tensorRT模型部署
- 精度和速度测试
- 问题排查与分析

接下来，我将给出相关的工具，并对其中的关键步骤进行详细说明；

<br>

### 2.1 模型导出
onnx是一种模型表示方式，能够将不同框架下的模型，统一表示为同一种形式；因此，尝尝被用来作为模型转换的中间节点；目前，tensorRT已经支持了直接用torch转成tensorRT的engine；但是其它的SDK框架，如MNN、TNN、Paddle-Lite、OpenVino等仍然只支持onnx格式的模型转换；并且，onnx本身也是一种很好用的模型框架，可以很方便地在上面做开发；

```
import torch
# 加载你的模型
model = build_model(config.model)
checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
load_model_weight(model, checkpoint)

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

```

其它常用框架也基本都有导出为onnx模型的方式，可以通过搜索引擎很容易得到相关结果，再次不作过多介绍。

导出模型为onnx以后，如果不需要做模型量化，可以直接将onnx模型转换为所需的格式后进行模型部署；如果想快速完成部署，可以使用在线模型转换的工具来完成模型转换 https://convertmodel.com/；

<br>

### 2.2 onnx模型结构优化 ###

onnx模型结构优化，一方面是为后续的模型量化做准备；另一方面是减少了输入和输出部分的计算，这部分计算对云端的算力而言可能是无关紧要的，但是对端上的微弱算力而言，这部分计算能省则省吧。


simplify


optimize

预处理融合

sigmoid移除


### 3.1 量化

*3.1.1 量化的理论基础* 


*3.1.2 量化的计算过程*  


*3.1.3 常用的量化工具箱* 

*3.1.4 PTQ量化*

- 简单量化
- balance vector(weight equalization)
- bias correction


*3.1.5 QAT量化*

- QDQ模式介绍
- QDQ流程优化


3.2 剪枝



3.3 蒸馏


## 4. 参考
1. tiny-tensorRT: https://github.com/zerollzeng/tiny-tensorrt
2. micronet: https://github.com/666DZY666/micronet
3. ppq: https://github.com/openppl-public/ppq
4. onnx-runtime quantization: https://onnxruntime.ai/docs/performance/quantization.html
5. polygraphy: https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy