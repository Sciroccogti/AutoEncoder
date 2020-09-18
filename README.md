# This is the code for competition AI+无线通信 2020 NIAC https://naic.pcl.ac.cn/ with pytorch.

## requirements
- pytorch > 0.4
- numpy
- h5py

<!-- ## result

test score = 0.67 -->

## 原理

编码器将信息比特映射到一个定长嵌入层，再将嵌入层通过信道传给解码器。
这样解码器就只要解码嵌入层，而非人类设计的编码。

> 嵌入层：是使用在模型第一层的一个网络层，其将所有索引标号映射到致密的低维向量中

编码器包含了信道编码和调制。

![](assets/fig2.png)

嵌入层的向量维度大于输入向量，以此来去噪。

在自编码器和自解码器中使用了一维卷积层

### 深度卷积自编码器

第 i 层的第 n 个神经元的输出为 $u^{(i)}[n]=\sigma(\sum_{k=1}^L{w_{k}^{(i)}u^{i-1}[n-k]})$，
其中 $\sigma$ 是激活函数，$w$ 连接 i-1 层的第 k 个神经元 与 i 层的第 n 个神经元，

### 循环卷积层

循环卷积层是为了避免传统卷积层中 padding 带来的误差，
故其每层的头部的 padding 是输入的尾部，每层尾部的 padding 是输入的头部。
网络的输入因此被看作圆，而同一层上的各个神经元的输入会是相同的。

### 训练

自编码器-解码器使用接收方的端到端复原损失来训练。损失函数的梯度会被反向传输回接收方。
尽管使用了噪声层来对噪声建模，但由于噪声是加性且与输入无关，因此梯度不受噪声影响。

在训练时，每个训练样本都以随机比特的形式生成。

## 结构

Type of layer|Kernel size/Annotation|Output size
|---|---|---|
Input|Input layer|K × 1
Conv+Relu|5|K × 256
Conv+Relu|5|K × 128
Conv+Relu|5|K × 64
Conv|3|K × 2
Normalization|Power normalization|K × 2
Noise|Additive noise|K × 2
Conv+Relu|5|K × 256
Conv+Relu|3|K × 128
Conv+Relu|3|K × 128
Conv+Relu|3|K × 128
Conv+Relu|3|K × 64
Conv+Relu|3|K × 64
Conv+Relu|3|K × 64
Conv+Relu|3|K × 32
Conv+Relu|3|K × 32
Conv+Relu|3|K × 32
Conv+Sigmoid|3|K × 1
