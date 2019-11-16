[*<<返回主页*](../index.md)<br><br>
**本文为作者原创，转载请注明出处**<br>
### 深入浅出Attention和Transformer
本文介绍attention机制和基于attention的transformer模型。网上关于这两者的博客很多，但大都照搬论文，千篇一律，不够深入和通俗，本文在参考这些博客和原始论文的基础上，加入自己的理解，深入且通俗的讲解attention和transformer。<br>
#### Attention in RNN
Bengio等人在2014年[Neural machine translation by jointly learning to align and translate](https://arxiv.org/abs/1409.0473)中提出Attention机制。<br><br>
传统的RNN在t时刻的输入是h(t-1)和x(t)，encoder在最后时刻输出一个针对input sequence的特征表示，decoder在生成output sequence时输入该特征表示和上一时刻的解码结果。在长句子翻译任务中，decoder并不关心input sequence的整个句子，而只关心input sequence的某一些局部特征，因此Bengio等人提出Attention机制，学习input sequence和output sequence的对齐(Align)和翻译(Translate)关系，在编码过程中，将原句子编码成特征向量的集合，在解码时，每个时刻会从该特征集合中选择一个子集用于生成解码结果。模型结构如下图所示：<br>
![attention in RNN](../images/NLP/2_attention_transformer/attention_in_rnn.png)<br>
