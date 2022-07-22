# onnx2trt  


【DEPRECATED】 开发过程中，发现了一个整体思路与我这个repo类似，但功能更完善，且实现了一些高级算法的repo，建议使用这个repo来进行模型量化和部署； https://github.com/openppl-public/ppq



onnx2trt是用于进行tensorRT的int8模型量化的工具; 在进行int8模型量化时，某些int8 tensorRT模型的精度会出现一定程度的下降。而当前tensorRT默认使用的校准算法是Entropy, 为此特意开发onnx2trt工具来优化量化模型的精度。



## 安装
python36 (py37会遇到pycuda安装的问题)

pip install nvidia-pyindex 
pip install nvidia-tensorrt  
pip install pycuda  
pip install sympy  



## tensorRT量化存在的问题

1. 大模型的量化误差累积
   在进行模型的量化校准时，通常的做法是先用fp32模型进行一遍infer，然后统计每个节点的动态范围。这样的做法简单快捷，做一遍infer即可得到整个模型所有节点的动态范围。
   但是，当层数较多时，量化的误差会不断累积；距离模型输入越远，这种量化误差越大。


2. 量化后阈值偏移
   当某个节点的输出数量比较小时，节点输出的cosine相似度已经很高，但是却出现了阈值偏移；
   例如：fp32_out = [-6.223839, 3.5978181, -2.4270086], int8_out = [-2.37992859, 1.80094731, -1.93005347]
   如果这个输出后接的是softmax结构的话，这种阈值偏移对最终精度的影响会比较小；
   但是如果这里输出的是分数score的话，就会带来一些不利于实际部署的结果：例如recall降低，precision升高的变化；这时需要重新调整阈值来维持precision或者recall不变，以保证模型部署的效果。


TODO:  
- [ ] QDQ量化工具: 使用QDQ方式进行tensorRT的模型量化.  
- [ ] 量化精度损失分析工具:    
   - tensorRT自带量化分析工具polygraph: https://zhuanlan.zhihu.com/p/535021438 
   - 给定每个节点的量化scale值，计算每一层的量化前后的cosine值.   
   - 给定每个节点的量化scale值，计算这一层量化对最终输出的莲花cosine值.   
- [ ] 自定义scale计算工具/自定义calibrator:     
   - 用于trt exec生成trt engine(隐式设置精度). 
   - 用于QDQ生成trt engine(显式设置精度). 

