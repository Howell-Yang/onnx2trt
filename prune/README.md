## PRUNE: 自定义剪枝工具

在模型剪枝的工程实践中，我们发现当前的剪枝工具，对很多特殊的结构(depthwise convolution、res block)是不支持的；端上常用的shufflenet等模型，无法使用现有工具进行剪枝；为此，我临时开发了一个简单的剪枝工具，来完成针对shufflenet和PAN结构的channel-wise的剪枝；  

<br></br>

### 1. 问题描述

常见的剪枝工具没有处理特殊结构的能力:
- depthwise convolution: 普通卷积的权重的shape是[Co, Ci, Kw, Kh], 深度分离卷积的权重shape是[Co, 1, Kw, Kh]; 进行channel-wise剪枝时，需要根据前置节点的channel，决定当前节点的channel选择，并且决定了后置节点的输入channel的选择；

- channel shuffle: 通道shuffle后的卷积的输入权重，剪枝通道要与通道shuffle前的通道对应上；需要识别不同卷积层之间的通道对应关系；

- Add结构: 输入的多个卷积层，其剪枝的channel需要保持一致，否则将会出现Add的channel之间的不对应的问题；

- Slice结构: slice前整个feature map的有效channel数是原模型的1/2，但是slice之后的两个feature map中有效channel数不是原来的1/2了；导致模型结构不符合预期；

<br></br>
### 2. 解决方案

放弃的解决方案：
- 使用已有的nni等工具:
   - 基于torch.fx来识别不同模块之间的关联关系；然后开发对应的剪枝功能；
   - torch.fx会转换所有的表达式，粒度很细；学习成本有点高；
- 使用onnx来获取拓扑结构：
   - onnx能够很简单的识别模块间的关联关系
   - onnx对于channel间的转换的识别能力不够(onnx无法单独设置每一层的输入)


最终选择的解决方案：
- 使用torch的hook机制，自行构建mask和剪枝流程:
   - 首先，使用hook获取模块的id，用于后续构架关联关系
   - 其次，识别需要联合剪枝的模块(在这里主要是指Add)
   - 然后，构建剪枝用到的mask
   - 最后，对权重进行修改


特殊要素的处理:
- depthwise convolution: 根据输入来决定输出channel的选择；
- channel shuffle: 通过输出置零来实现需要剪枝的channel之间的传递；
- add: 记录add相关的模块id，计算权重的均值来剪枝
- slice: 针对shufflenet的结构，将channel划分为4组，分别进行channel选择；

<br></br>

### 3. 实际效果

剪枝工具开发
- shufflenet剪枝: channel shuffle
- FPN剪枝: Add层的处理