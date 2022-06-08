# onnx2trt

onnx2trt是用于进行tensorRT的int8模型量化的工具; 在进行int8模型量化时，某些int8 tensorRT模型的精度会出现一定程度的下降。而当前tensorRT默认使用的校准算法是Entropy, 为此特意开发onnx2trt工具来优化量化模型的精度。


## 安装
python36 (py37会遇到pycuda安装的问题)

pip install nvidia-pyindex
pip install nvidia-tensorrt
pip install pycuda
pip install sympy



## 使用





## tensorRT量化存在的问题

1. 大模型的量化误差累积
   在进行模型的量化校准时，通常的做法是先用fp32模型进行一遍infer，然后统计每个节点的动态范围。这样的做法简单快捷，做一遍infer即可得到整个模型所有节点的动态范围。
   但是，当层数较多时，量化的误差会不断累积；距离模型输入越远，这种量化误差越大。


2. 量化后阈值偏移
   当某个节点的输出数量比较小时，节点输出的cosine相似度已经很高，但是却出现了阈值偏移；
   例如：fp32_out = [-6.223839, 3.5978181, -2.4270086], int8_out = [-2.37992859, 1.80094731, -1.93005347]
   如果这个输出后接的是softmax结构的话，这种阈值偏移对最终精度的影响会比较小；
   但是如果这里输出的是分数score的话，就会带来一些不利于实际部署的结果：例如recall降低，precision升高的变化；这时需要重新调整阈值来维持precision或者recall不变，以保证模型部署的效果。


