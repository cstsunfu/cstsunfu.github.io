---
layout: post
title: 图神经网络综述（GNN Survey）
date: 2019-04-25
tags: GNN
---

'''
最近了解了一下GNN，写本文概述以加深理解，主要参考一下两篇综述文章：

清华大学孙茂松组的 [Graph Neural Networks: A Review of Methods and Applications](https://arxiv.org/pdf/1812.08434.pdf)

IEEE Fellow, Philip S. Yu的 [A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/pdf/1901.00596v2.pdf)

'''


### 符号定义


| 标记                                                        | 描述                         |
| --------------------------------                            | ------------------           |
| $$\mathbb{R}^m$$                                            | m维欧式空间                  |
| $$ b, \mathbf{b}, \mathbf{B}$$                              | 标量，向量，矩阵             |
| $$ \mathbf{W} ^T$$                                          | 矩阵转置                     |
| $$ \mathcal{N} _v$$                                         | 节点v的邻居个数              |
| $$ e_{vw}$$                                                 | 链接v、w的边                 |
| $$\parallel$$                                               | 拼接                         |
| $$\odot $$                                                  | 矩阵对应元素点乘             |
| $$ g_{\theta}\star h$$                                      | $$ g_{\theta}$$ 和 $$x$$卷积 |
| $$ \sigma$$                                                 | sigmoid                      |
| $$ h_v$$                                                    | 节点v的表示                  |
| $$ \mathbf{H} \in \mathbb{R}^{N\times D} $$                 | 图的所有节点的特征表示       |
| $$ \mathbf{W} $$                                            | 可学习参数矩阵               |
| $$ \mathbf{A} $$                                            | 邻接矩阵                     |
| $$ D \in \mathbb{R}^{N \times N}, D_{ii} = \sum_j A_{ij} $$ | 度矩阵，是一个对角阵         |
| $$ V$$                                                      | 所有的节点                   |
| $$ I_n$$                                                    | n阶单位阵                    |
| $$ \mathcal{F}$$                                            | 傅里叶变换                   |

### GNN( Graph Neural Networks )简介

之前深度学习主要关注例如文字的序列结构、例如图片的平面结构，现在处理这些数据的做法也比较成熟，关注序列任务的NLP领域多用RNN、Transformer、CNN对数据进行Encoder，而关注平面结构的CV领域更多使用CNN及其各种变体对数据进行Encoder。在现实世界中更多的数据表示并不是序列或者平面这种简单的排列，而是表现为更为复杂的图结构，如社交网络、商品-店铺-人之间的关系、分子结构等等

图是由节点及连接节点的边构成的，现在热门的基于深度学习的GNN就是用来处理图类型数据的网络，而该网络的目标就是学习每个节点v的表示$$ h_v \in \mathbb{R}^m $$，而每个节点的表示由该节点的特征、与该节点连接的边的特征、该节点的邻居表示和它邻居节点的特征计算得到：

$$ h_v = f(x_v, x_{co[v]}, h_{ne[v]}, x_{ne[v]})$$

对于关注节点的任务，可以直接拿$$h_v$$的表示去完成特定任务，而对于关注整个图的任务这可以通过将所有的节点的表示做Pooling或其他方法获得一个全局的表示信息然后去做相应的任务。

### GNN分类--按更新方式分类

![GNN类别](/images/posts/gnn_survey/propegation.png)]
如图所示，GNN主要分为图卷积网络(GCN)、基于注意力更新的图网络(GAT)、基于门控的更新的图网络、具有跳边的图网络。G

### 各种GCN

图卷积网络是目前最主要（重要）的图网络，GCN按照更新方式又可以分为基于谱的和基于空间的。

#### 基于谱的GCN

我们常用的GCN模型长这样：


$$ H^{t+1}   = \tilde{A}H^{t}W$$

$$ \tilde{A} = I_n + D^{- \frac{1}{2} }A D^{- \frac{1}{2} }$$

对于这个式子直觉告诉我这和NLP里面的Self-Attention的聚合过程也太像了吧，把$$\tilde{A}$$换成我们Attention时计算出来的权重矩阵，这就是更新过程，而且从直觉上想若两个节点之间存在一种关系（有边）则对应邻接矩阵A上的一个非零值（权重），如果没有关系则对应0，而$$ D^{- \frac{1}{2} }$$在这个式子里面起到的作用就是归一化，单位阵I起到的作用是添加一个自己到自己的边。

这个公式很容易理解，它就是一个图卷积神经网络，$$\tilde{A}$$是一个卷积核，不过看起来和我们见过的卷积操作好像相差比较大，不过这个公式并不是和我们直觉的想法一样得来的，而是经历了一个漫长的过程。

在继续介绍之前先讲一个图信号处理领域的一个结论：

定义归一化的拉普拉斯矩阵 $$ \mathbf{L} = \mathbf{I}_n - D^{- \frac{1}{2} } A D^{- \frac{1}{2} } $$， 拉普拉斯矩阵是实对称矩阵，可以对其进行谱分解得到$$ \mathbf{L} = \mathbf{U} \Lambda \mathbf{U}^T  $$， 其中$$ \mathbf{U}$$是酉矩阵（即矩阵的列向量两两正交，且$$U^{-1} = U^T$$）。 图信号$$h\in \mathbb{R}^N$$是图中第i个节点$$h_i$$的表示向量,信号h的图傅里叶变换定义为$$ \mathcal{F}(h)=U^Th$$,逆傅里叶变换为$$ \mathcal{F}^{-1}(\hat{h})=U\hat{h}, \hat{h}$$ 表示图傅里叶变换对信号h的输出。因为我对这部分数学基础不足，只能根据直觉解释这个过程，在信号处理中，傅里叶变换就是将在时域的信号变换到频域，而所用到的所有的余弦/正弦函数都是正交向量（无穷维向量），每一个正弦/余弦函数就是一个频域的基，所以傅里叶变换终究还是一个变基的过程，就是将信号从一个空间通过改变基的表示映射到另一个空间。上文中提到的$$U$$矩阵作为一个酉矩阵（列向量两两正交）显然满足作为一组基(且是标准正交基)的条件，我们可以通过$$U$$矩阵将h映射到另一个空间，该空间就是图的频域空间，至于为什么选择$$U$$矩阵作为变换矩阵而不是随便选一个酉矩阵作为变换矩阵，显然是因为这个$$U$$矩阵是和图的结构密切相关的（由该图的归一化拉普拉斯矩阵谱分解得到的）。

对于该结论的严格推倒过程可以参考The Emerging Field of Signal Processing on Graphs这篇论文。

下面开始继续介绍GCN，定义的对每个节点v的卷积操作，$$g \star h$$，卷积有一个性质:时域的卷积既是在频域的乘积，所以为了计算方便可以先将信号做傅里叶变换到频域之后再做逆变换即可得到卷积结果，其中*g是与拉普拉斯矩阵的特征值相关的*。

$$ g \star h = \mathcal{F}^{-1}( \mathcal{F}(g \odot \mathcal{F}( h)   )  = \mathbf{U}( \mathbf{U}^T g \odot \mathbf{U}^T h   ) $$

这里如果将$$ \mathbf{U}^T g$$参数化为$$g_{\theta}$$，上式将简化为 $$ g\star h = \mathbf{U}g_{\theta} \mathbf{U}^T h$$，基于谱的卷积都是如此定义，区别在于$$g_{\theta}$$的不同选择。

最简单的方法是直接另$$ g_{\theta}=diag\{ \theta_1, \theta_2, \cdots, \theta_n \}$$，此处 $$ \theta是与拉普拉斯矩阵特征值相关的， 实际为 g_{ \theta }( \Lambda), 为简单省略了 \Lambda$$

这种简单的计算方法每次都要对 $$ \mathbf{L} $$矩阵进行谱分解，参数数量和图的大小一致，且每次都是全局卷积学习参数的复杂度会非常高，不仅如此，每次卷积涉及到的大规模矩阵乘法耗时也是$$ \mathbf{O}(n^2) $$。

**为了解决上面的问题有人提出一种巧妙地方法即下面要讲的*ChebNet*.**

*首先解决参数数量和局部卷积简化参数学习难度的问题。*

我们希望用更少的（与图大小无关）的参数来表示卷积核，另

$$ g_{\theta} = diag( \sum^{K}_{j=0} \alpha _j \lambda_1^j \dots \sum_{j=0}^K \alpha_j \lambda_n^j) = \sum_{j=0}^K \alpha_j \Lambda^j$$

其中$$\Lambda = diag(\lambda_1 \cdots \lambda_n)$$（拉普拉斯矩阵的特征值）

这样由:

$$ \mathbf{L}^2 = \mathbf{U} \Lambda \mathbf{U}^T \mathbf{U} \Lambda \mathbf{U}^T     $$

$$ \mathbf{U} \mathbf{U}^T = I_n  $$

得

$$\begin{aligned}
g_{\theta} \star h &= \mathbf{U} \sum^{K}_{j=0} \alpha_{j} \Lambda_{j} \mathbf{U}^{T} h \\\ 
                &= \sum_{j=0}^{K} \alpha_{j} \mathbf{U} \Lambda^{j} \mathbf{U}^{T}h \\\ 
                &= \sum^{K}_{j=0} \alpha_{j} \mathbf{L}^{j} h 
\end{aligned}$$

现在的图卷积已经不用进行谱分解了，这里只需要计算$$ \mathbf{L}^{j} $$，其中$$ \mathbf{L} $$是由邻接矩阵导出来的（加上自回归的边后，每个位置是否为0是相同的），我们学习图论的时候知道通过计算邻接矩阵的k次方得到的矩阵非0位置表示两个节点存在长度为k的路径，这样的卷积具有很好的空间局部性，我们可以通过控制k来决定每个节点的表示仅由与其距离不超过k的节点及其关系决定，不像之前的图卷积计算当前节点会用到全局所有节点的信息。

*然后解决每次卷积计算复杂度为$$\mathbf{O}(n^2)$$的问题*

这里用到Chebyshev polynomial，该多项式是递归定义的：

$$ T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x)$$

$$T_{0} = 1, T_1 = x$$

为了能够简化卷积的计算复杂度，我们发现卷积核的计算需要计算 $$ \Lambda^j$$, 将卷积核定义为如下可以递归计算的方式可以减少计算量。

$$ g_{\theta}(\Lambda) = \sum_{k=0}^{K-1} \theta_k T_k( \tilde{\Lambda} )$$

其中$$ \tilde{\Lambda} = \frac{2\Lambda}{\lambda_{max}} - I_n$$

然后通过和上面类似的推倒得到递归的卷积：

$$ y = g_{\theta} ( \mathbf{L} )x = \sum_{k=0}^{K-1} \theta_k T_k(\tilde{ \mathbf{L} })x$$

$$ \tilde{ \mathbf{L}  } = \frac{2}{ \lambda _{max}} \mathbf{L} - \mathbf{I}_N$$

现在的卷积操作已经能满足上面提出的所有要求了。

在此基础上的*变种*

很多时候K不需要取的很大就能获得很好的效果，当K=2时，即只取前两项可以得到：

$$ g_{\theta} ( \mathbf{L} )x = \theta_0 x + \theta_1 \tilde{L} x $$

进一步简化，取 $$ \lambda_{max} \approx 2, \theta = \theta_0 = -\theta_1 $$

则：

$$ g_{ \theta \star x} \approx \theta(I_N + D^{- \frac{1}{2} }AD^{- \frac{1}{2} })x$$

这就是本节刚开头提到的形式。

这种形式在堆叠多层时会造成数值上的不稳定及梯度爆炸/消失问题，所以可以用到重归一化技巧：

$$ I_n + D^{- \frac{1}{2} }AD^{- \frac{1}{2} } \rightarrow \tilde{D}^{- \frac{1}{2} }A \tilde{D}^{- \frac{1}{2} } , 其中 \tilde{A} = A + I_N, \tilde{D}_{ii} \sum_{j} \tilde{A}_{ij} $$

#### 非基于谱的方法

基于谱的方法需要学习的参数都是与Laplacian矩阵的特征向量和特征值相关的，即取决于图的结构，这样有什么问题呢，如果你在一个大图或者很多同构的小图上进行训练是可以的，这也有很多应用场景。但是如果我们要处理很多不同结构的图，那就不能用同一套参数来学习，不惧泛化性，比如我们对很多树结构的句子进行分类（比如表示为依存句法树或其他），每个句子的表示可能都不同，那就没办法用上面的方法做。

下面介绍几种不依赖于图谱的方法的GNN。

不基于谱最简单的方法就是直接把每个节点的邻居节点直接求和：

$$ x = h_v + \sum^{ \mathcal{N}_v  }_{i=1}h_i $$

$$ h_v^{'} = \sigma(xW_L^{ \mathcal{N}_v  })  $$

这里根据邻居个数的不同用不同的参数W。

*DCNN*

DCNN的做法也很简单：

$$ H = f(W^c \odot P^* X)$$

其中$$P^* = {P, P^2, \dots,P^K}$$, P是度归一化后的临接矩阵。不要误会，这里的参数和图的结构没有关系，这里的参数W对应于与该节点不同距离的信息。

*GraphSAGE*提出了一个更加通用的框架：

$$ h_{ \mathcal{N}_v }^t = AGGREGATE_t(\{h_u^{t-1}, \forall u \in \mathcal{N}_v \})$$

$$h_v^t = \sigma(W^t \cdot [h_v^{t-1}\|h_{\mathcal{N}_v}^t ])$$

这里的AGGREGATE可以取不同的方法，GraphSAGE论文中提到可以用mean，max-pooling, sum，LSTM。 其实这是一个通用框架，本文提到的很多方法都可以套进来。

*Attention*

图的信息传递当然也可以用attention，通过当前节点的不同邻居与当前节点表示上的关系计算信息流的权重, a为attention参数：

$$ \alpha_{ij} = \frac{exp(LeakyReLu(a^T[Wh_i \| Wh_j]))}{\sum_{k \in \mathcal{N}_i exp(LeakyReLu(a^T[Wh_i \| Wh_k])) }} $$

$$ h_i^{'} = \sigma( \sum_{j \in \mathcal{N}_i} \alpha_{ij}Wh_j)$$

为了获得更充分（不同侧重）的信息，Attention is all you need 提出的multi-head attention，可以表示为下面两种形式

$$ h_i^{'} = \|_{k=1}^K \sigma( \sum_{j \in \mathcal{N}_i } \alpha_{ij}^kW^kh_j )$$

$$ h_i^{'} =  \sigma(\frac{1}{K} \sum_{k=1}^K  \sum_{j \in \mathcal{N}_i } \alpha_{ij}^kW^kh_j )$$

其中$$ \|$$表示拼接


*Skip connection*

相比于传统的网络，GNN的深度一般更浅，原因是GNN的感受野随着深度的增加指数增大在信息传递过程中会引入大量噪声。所以在GNN中也有人尝试加入skip connection来缓解这个问题。下面是一个用Highway的方法的例子：

$$ T(h^t) = \sigma(W^th^t +b^t)$$

$$ h^{t+1} = h^{t+1} \odot T(h^t) + h^t \odot(1 - T(h^t))$$

当前的输出还有一部分来自于上层的输出。

*Gate*

Highway的方法比较简单，还有更复杂的方法来控制信息的传播，比如可以用GRU，LSTM的门控方式来传递图的信息，这个没啥好解释的，看公式应该写的很明白了，这里只介绍一下基于GRU的GGNN：

$$ a_v^t = A_v^T[h_1^{t-1}, \cdots, h_N^{t-1}]^T+ b$$

$$ z_v^t = \sigma(W^z a_v^t + U^z h_v^{t-1})$$

$$ r_v^t = \sigma(W^r a_v^t + U^rh_v^{t-1})$$

$$ \tilde{h_v^t}= tanh(Wa_v^t+ U(r_v^t \odot h_v^{t-1}))$$

$$ h_v^t = (1-z_v^t) \odot h_v^{t-1} + z_v^t \odot \tilde{h_v^t}$$

其中$$ A_v$$是临接矩阵中与v有关的值构成的矩阵。



### GNN 训练方法

原始的GCN方法每个节点的表示依赖于图中所有的其他节点，计算复杂度过大，且由于依赖于拉普拉斯矩阵训练的网络不惧泛化性。

GraphSAGE对于每个节点的计算不涉及整张图，所以效率更加高，不过为了缓解深度加深感受野指数爆炸的现象，GraphSAGE每次信息计算通过采样只使用部分邻居。

FastGCN对GraphSAGE的随机采样加以改进，针对每个节点连接其他节点个数的不同给定不同的重要性。

$$ q(v) \propto \frac{1}{| \mathcal{N}_v| } \sum_{u \in \mathcal{N}_v } \frac{1}{| \mathcal{N}_u | } $$

还有一些限制感受野的其他方法，咱也不懂.


### 通用框架

*Message Passing*

该框架包含3个操作，信息传递操作（M），更新操作（U），读取操作（R）：

$$ m_v^{t+1} = \sum_{w \in \mathcal{N}_v }  M_t(h_v^t, h_w^t, e_{vw})$$

$$ h_v^{t+1} = U_t(h_v^t, m_v^{t+1})$$

$$ \hat{y} = R(\{h_v^T|v \in G\})$$

其中$$ e_{vw}$$表示边的信息。

上文提到的GGNN用这个框架表示为：

$$\begin{aligned}
M_t(h_v^t, h_w^t, e_{vw}) &= A_{e_{vw}}h_w^t \\\
U_t &= GRU(h_v^t, m_v^{t+1}) \\\
R &= \sum_{v \in V} \sigma(i(h_v^T, h_v^0)) \odot(j(j_v^T))
\end{aligned}$$

*NLNN*

何凯明提出的Non-local Neural Networks用来捕捉图像上的non-local信息：

$$ h_i^i = \frac{1}{C(h)} \sum_{ \forall j } f(h_i, h_j) g(h_j) $$

其中$$ f$$用于计算i、j之间的关系，$$ g(h_j)$$ 表示对输入做一个映射， $$ \frac{1}{C(h)} $$表示对结果进行归一化。其中$$ f$$可以有不同的计算方式，比如什么Gaussian，Embedded Gaussian， Dot product, Concatenation等。
感觉这个有点牵强。。

*Graph Networks*

这个就是去年火爆一时的由一群大佬联名的Relational inductive biases, deep learning, and graph networks。

由于这个更复杂更general，先来介绍一下其中的符号表示, 这里V的表示与文章开头定义有所不同：

$$ N_v表示节点总数， N_e表示边的数量$$
图：$$ G = (u, V, E)$$, 其中 $$ V = \{h_i\}_{i=1:N^v}$$表示图中的每个节点的表示，u表示全局信息，$$ E = \{(e_k, r_k, s_k)\}_{k=1:N^e}$$, 这其中，$$ e_k 表示边的信息，r_k表示接收节点，s_k表示发送节点$$

分别针对边、节点、全局节点定义了三种更新函数$$ \phi$$和三种聚合函数 $$ \rho$$:

$$ e_k^{'} = \phi ^e(e_k, h_{rk}, h_{sk}, u)$$

$$ h_i^{'} = \phi ^h(\bar{e}_i^{'}, h_{i}, u)$$

$$ u^{'} = \phi ^u(\bar{e}^{'}, \bar{h}^{'}, u)$$
 
 
$$ \bar{e_i^{'}} = \rho^{e \rightarrow h} (E_i^{'})$$

$$ \bar{e^{'}} = \rho^{e \rightarrow u} (E^{'})$$

$$ \bar{h^{'}} = \rho^{h \rightarrow u} (H^{'})$$

其中$$ E_i^{'} = \{(e_k^{'}, r_k, s_k)\}_{r_k=i, k=1:N^e}$$, $$ H^{'} = \{h_i^{'}\}_{i=1:N^v}$$, $$ E^{'} = \bigcup _i E_i^{'} = \{(e_k^{'}, r_k, s_k)\}_{k=1:N^e}$$

每个操作都很符合直觉，下面给出在该框架下的计算算法：

![Steps of computation in a full GN block](/images/posts/gnn_survey/gn_block.png)]

最终返回全局节点、普通节点、边的表示，在该框架下的图任务可以是基于图的也可以是基于节点的还可以是基于边的。

### ...

以后有时间再补充吧。如果有人看的话，有错误还请指出，共同进步.

^_^
