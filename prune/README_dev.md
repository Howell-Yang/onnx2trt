1. 剪枝概述




2. 常用工具

- nni: 基于torch.fx，在某些复杂网络如nanodet上失效；
- pytorch原生工具: 难以获取网络的拓扑结构，对于包含Concat等结构的网络剪枝困难；
   - torch.fx工具虽然提供了相关的功能，但是对于其中的shape计算、特殊算子等支持并不友好(其实是我太懒，不想新学一种表示)
   - 更简单的办法是，通过onnx作为中间层，使用onnx simpliier工具优化后，读取模型的拓扑结构；
   - 根据onnx模型中的拓扑结构，来进行结构化的剪枝；保存模型权重，输出模型结构；
   - 最后进行模型的finetune，输出最终的模型；
- 问题: onnx会修改权重的名称，无法根据名称找到对应关系；
- 解决方案:
   - 通过计算相似度等方式找到权重对应关系 ----> 设置training为True后，权重名称也得以与state_dict中一致
   - 通过设置export的training=True来避免Conv-BN融合
- 问题: 对于Add、Concat、Channel Shuffle结构，需要进行针对性的识别和处理 ----> 复杂度较高
- 解决方案: 
   - 使用torch的hook机制，对特定类型的节点进行hook，从而获得每个op的mask
   - 然后根据mask的结果，对权重进行prune(整个流程可能跟nni中的torch.fx是一致的)
   - Add、Concat模块没办法直接进行Hook，需要使用自定义的module实现后才能完成hook功能
- 为了便于后续的研究，将采用复杂的方案来进行剪枝；而不是每个模块设置一个prune函数

3. 剪枝工具开发
- shufflenet剪枝: channel shuffle
- FPN剪枝: Add层的处理


4. 实验记录

- 方案一: 直接根据BN层的scale参数的l1-norm大小来剪枝
- 方案二: 根据当前层的输出的l1-norm大小来剪枝
- 