---
layout: post
title: 读Universal Transformer, 并浅析为什么Transformer在NLP领域兴起
date: 2018-11-01
tags: Paper
---

Universal Transformer [论文地址](https://arxiv.org/abs/1807.03819)

### 引言

Transformer从去年的Attention is all your need提出到最近的BERT的屠榜展示出了惊人的威力，也理所当然的成为了最近NLP领域最引人注目的研究成果，本文将从信息传递的角度谈一谈个人对Transformer的理解。最后讲一下Universal Transformer对Transformer的一些改进。

在NLP领域信息传递方式再此之前经历了线性/非线性空间变换、CNN、RNN、Attention/Dynamic routing等阶段，每个传递方式的提出都志在解决之前方法存在的问题，下面将一一介绍。

### 线性/非线性空间变换

最早人们通过统计机器学习的方法来处理NLP问题很多只是通过统计词的信息来做。比如一个情感分类问题，我们想知道一篇文章的情感取向，只需要统计一下各类情感词出现的个数，然后通过某种加权方法得出最终的情感取向。这种方法非常简单，但是只能处理一些非常简单的问题，而且很多时候效果也不好，后来人们发现很多问题在当前的表示下并不那么容易划分，比如有很多篇文章最终的表示为图左(当然现实中表示维数可能非常大，但是这里为了可视化只能用2维和3维的数据进行表示)，红色为A类，绿色为B类，如果我们要求用一个超平面对其进行划分的话是做不到的（用曲线是能做到的），为了解决这种问题，人们提出了线性/非线性映射将现有表示映射到一个线性可分的空间，本例通过$$k(x,y) = x*y + x^2 + y^2$$将各数据样本映射到图右所示空间，现在就很容易找到一个超平面来划分整个数据。

### CNN

之前的线性空间变换

### Introduction

之前的glove以及word2vec的word embedding在nlp任务中都取得了最好的效果, 现在几乎没有一个NLP的任务中不加word embedding.

我们常用的获取embedding方法都是通过训练language model, 将language model中预测的hidden state做为word的表示, 给定N个tokens的序列$$(t_1, t_2,...,t_n)$$, 前向language model就是通过前k-1个输入序列$$(t_1, t_2, ...,t_k)$$的hidden表示, 预测第k个位置的token, 反向的language model就是给定后面的序列, 预测之前的, 然后将language model的第k个位置的hidden输出做为word embedding. 

之前的做法的缺点是对于每一个单词都有唯一的一个embedding表示, 而对于多义词显然这种做法不符合直觉, 而单词的意思又和上下文相关, ELMo的做法是我们只预训练language model, 而word embedding是通过输入的句子实时输出的, 这样单词的意思就是上下文相关的了, 这样就很大程度上缓解了歧义的发生.

### ELMo: Embeddings from Language Models

ELMo用到上文提到的双向的language model, 给定N个tokens (t1, t2,...,tN),  language model通过给定前面的k-1个位置的token序列计算第k个token的出现的概率:

$$p(t_1, t_2, ..., t_N) = \prod_{k=1}^N p(t_k|t_1, t_2, ..., t_{k-1})$$

后向的计算方法与前向相似:

$$p(t_1, t_2, ..., t_N) = \prod_{k=1}^N p(t_k\vert t_{k+1}, t_{k+2}, ..., t_{N})$$

biLM训练过程中的目标就是最大化:

$$\sum^N_{k=1}(\log p(t_k| t_1, ...,t_{k-1};\Theta _x, \overrightarrow{\Theta}_{LSTM}, \Theta _s) + \log p(t_k\vert t_{k+1}, ...,t_{N}; \Theta _x, \overleftarrow{\Theta}_{LSTM}, \Theta _s))$$

ELMo对于每个token $$t_k$$, 通过一个L层的biLM计算出2L+1个表示:

$$R_k = \{x_k^{LM}, \overrightarrow{h}_{k,j}^{LM}, \overleftarrow{h}_{k, j}^{LM} \vert j=1, ..., L\} = \{h_{k,j}^{LM} \vert j=0,..., L\}$$

其中$$h_{k,0}^{LM}$$ 是对token进行直接编码的结果(这里是字符通过CNN编码), $$h_{k,j}^{LM} = [\overrightarrow{h}_{k,j}^{LM}; \overleftarrow{h}_{k, j}^{LM}]$$ 是每个biLSTM层输出的结果. 在实验中还发现不同层的biLM的输出的token表示对于不同的任务效果不同.

应用中将ELMo中所有层的输出R压缩为单个向量, $$ELMo_k = E(R_k;\Theta _\epsilon)$$, 最简单的压缩方法是取最上层的结果做为token的表示: $$E(R_k) = h_{k,L}^{LM}$$, 更通用的做法是通过一些参数来联合所有层的信息:

$$ELMo_k^{task} = E(R_k;\Theta ^{task}) = \gamma ^{task} \sum _{j=0}^L s_j^{task}h_{k,j}^{LM}$$

其中$$s_j$$是一个softmax出来的结果, $$\gamma$$是一个任务相关的scale参数, 我试了平均每个层的信息和学出来$$s_j$$发现学习出来的效果会好很多. 文中提到$$\gamma$$在不同任务中取不同的值效果会有较大的差异, 需要注意, 在SQuAD中设置为0.01取得的效果要好于设置为1时.

文章中提到的Pre-trained的language model是用了两层的biLM, 对token进行上下文无关的编码是通过CNN对字符进行编码, 然后将三层的输出scale到1024维, 最后对每个token输出3个1024维的向量表示. 这里之所以将3层的输出都作为token的embedding表示是因为实验已经证实不同层的LM输出的信息对于不同的任务作用是不同的, 也就是所不同层的输出捕捉道德token的信息是不相同的.

### 通过AllenNLP使用ELMo

与训练的ELMo已经放出, pytorch用户可以通过AlenNLP使用, 预训的Tensorflow版本也在 [TF版](https://github.com/allenai/bilm-tf) 中放出, 这里介绍一下通过AllenNLP包用ELMo的方法.

通过pip install allennlp 或其他方法安装AllenNLP包之后, 一般我们直接调用allennlp的allennlp.commands.elmo.ElmoEmbedder 中的batch_to_embeddings对一个batch的token序列进行编码, 下面是样例做法, 第一次运行加载模型时运行时间会很长, 需要等待. 这里展示的用法不会对language model的参数进行更新, 如果需要请自己设置, 因为没有给通用接口, 所以需要修改allennlp的源码, 但是因为设置梯度回传之后效率大减, 且所需内存暴增, 我也只测试了一下, 并没有在我自己的模型中使用.

```
from allennlp.commands.elmo import ElmoEmbedder
elmo = ElmoEmbedder(options_file='../data/elmo_options.json', weight_file='../data/elmo_weights.hdf5', cuda_device=0)
context_tokens = [['I', 'love', 'you', '.'], ['Sorry', ',', 'I', 'don', "'t", 'love', 'you', '.']]
elmo_embedding, elmo_mask = elmo.batch_to_embeddings(context_tokens)
print(elmo_embedding)
print(elmo_mask)

1. 导入ElmoEmbedder类
2. 实例化ElmoEmbedder. 3个参数分别为参数配置文件, 预训练的权值文件, 想要用的gpu编号, 这里两个文件我是直接下载好的, 如果指定系统默认自动下载会花费一定的时间, 下载地址
   
    DEFAULT_OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    DEFAULT_WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    
3. 输入是一个list的token序列, 其中外层list的size即内层list的个数就是我们平时说的batch_size, 内层每个list包含一个你想要处理的序列(这里是一句话, 你可以一篇文章或输入任意的序列, 因为这里预训练的模型是在英文wikipidia上训的, 所以输入非英文的序列肯定得到的结果没什么意义).
4. 通过batch_to_embeddings对输入进行计算的到tokens的embedding结果以及我们输入的batch的mask信息(自动求mask)

    Variable containing:
    ( 0  , 0  ,.,.) = 
      0.6923 -0.3261  0.2283  ...   0.1757  0.2660 -0.1013
     -0.7348 -0.0965 -0.1411  ...  -0.3411  0.3681  0.5445
      0.3645 -0.1415 -0.0662  ...   0.1163  0.1783 -0.7290
               ...             ⋱             ...          
      0.0000  0.0000  0.0000  ...   0.0000  0.0000  0.0000
      0.0000  0.0000  0.0000  ...   0.0000  0.0000  0.0000
      0.0000  0.0000  0.0000  ...   0.0000  0.0000  0.0000
      
            ⋮  

    ( 1  , 2  ,.,.) = 
     -0.0830 -1.5891 -0.2576  ...  -1.2944  0.1082  0.6745
     -0.0724 -0.7200  0.1463  ...   0.6919  0.9144 -0.1260
     -2.3460 -1.1714 -0.7065  ...  -1.2885  0.4679  0.3800
               ...             ⋱             ...          
      0.1246 -0.6929  0.6330  ...   0.6294  1.6869 -0.6655
     -0.5757 -1.0845  0.5794  ...   0.0825  0.5020  0.2765
     -1.2392 -0.6155 -0.9032  ...   0.0524 -0.0852  0.0805
    [torch.cuda.FloatTensor of size 2x3x8x1024 (GPU 0)]

    Variable containing:
        1     1     1     1     0     0     0     0
        1     1     1     1     1     1     1     1
    [torch.cuda.LongTensor of size 2x8 (GPU 0)]

输出两个Variable, 第一个是2*3*8*1024的embedding信息, 第二个是mask, 其中2是batch_size, 3是两层biLM的输出加一层CNN对character编码的输出, 8是最长list的长度(对齐), 1024是每层输出的维度; mask的输出2是batch_size, 8实在最长list的长度, 第一个list有4个tokens, 第二个list有8个tokens, 所以对应位置输出1.

```

### 结语

ELMo的效果非常好, 我自己在SQuAD数据集上可以提高3个左右百分点的准确率. 因为是上下文相关的embedding, 所以在一定程度上解决了一词多义的语义问题. 

但是ELMo速度非常慢, 因为对每个token编码都要通过language model计算的出, 不如之前fix的embedding直接拿来用, 效率低到令人发指, 没有充足的计算资源会很难受. 这里一个解决办法是, 我们一般对模型需要多轮的训练, 每次训练都会重新通过language model计算token, 而我们不进行梯度回传更新biLM的参数, 所以我们输入相同的句子(文章或其他序列)输出结果不会改变, 因此我们可以只在第一个epoch中通过biLM计算token的表示, 然后我们保存起来, 下一次用到这个序列时直接加载, 可以节省大量时间, 这方面的分析见下一小节.


### 文章没提的--时间和空间及简单解决方案

为了让大家直观的了解到ELMo到底有多慢, 这里列一下我在SQuAD上的实验(不是特别精确).

数据集中包含越10万篇短文, 每篇约400词, 如果将batch设置为32, 用glove词向量进行编码, 过3个biLSTM, 3个Linear, 3个softmax/logsoftmax(其余dropout, relu这种忽略不计), 在1080Ti(TiTan XP上也差不多)总共需要约15分钟训练完(包括bp)一个epoch. 而如果用ELMo对其进行编码, 仅编码时间就近一个小时, 全部使用的话因为维度非常大, 显存占用极高, 需要使用多张卡, 加上多张卡之间调度和数据传输的花销一个epoch需要2+小时(在4张卡上). 

因为我们需要训练很多歌epoch才能让模型收敛, 而ELMo虽然对同一个单词会编码出不同的结果, 但是上下文相同的时候ELMo编码出的结果是不变的(这里不进行回传更新LM的参数), 为了解决上面的问题, 我们可以将数据集中的所有词的ELMo编码存起来(不同epoch共享同一个编码, 同一个单词编码还是上下文相关的), 这里又有一个新的问题--空间问题, 上文已经提到ELMo每个词编码成3*1024维的向量, 每个用一个单精度float表示, 共需3*1024*4Byte =12KB, 一个单词的编码就需要12KB数据来表示它的语义信息, 1GB内存也就能存个8万多个词的编码, 像上文提到的SQuAD需要约480G的内存来保存所有词的编码信息, 所以这是一个鱼和熊掌的选择.

其实解决方案还是有的, 因为论文中发现不同任务对不同层的LM编码信息的敏感程度不同, 比如SQuAD只对第一和第二层的编码信息敏感, 那我们保存的时候可以只保存ELMo编码的一部分, 在SQuAD中只保存前两层, 存储空间可以降低1/3, 需要320G就可以了,  如果我们事先确定数据集对于所有不同层敏感程度(即上文提到的$$s_j$$), 我们可以直接用系数超参$$s_j$$对3层的输出直接用$$\sum _{j=0}^L s_j^{task}h_{k,j}^{LM}$$压缩成一个1024的向量, 这样只需要160G的存储空间即可满足需求.
