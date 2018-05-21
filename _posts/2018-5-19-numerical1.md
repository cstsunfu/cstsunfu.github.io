---
layout: post
title: 数值分析-迭代法
date: 2018-05-19
tags: Numerical Analysis
---
### 简单例子

求$$\sqrt a$$的值在计算机中需要通过迭代计算，等价方程$$x^2-a=0$$，转化为方程求根问题。

给定一个初始值$$x_0>0$$，另$$x=x_0+\Delta x$$，$$\Delta x$$是增量，于是

$$(x_0+\Delta x)^2 = a$$, 即 $$x_0^2 + 2x_0\Delta x +(\Delta x)^2 =a$$
于是
$$\begin{align*}
    x &=x_0+\Delta x \approx a \\
    \Longrightarrow \Delta x &\approx \frac{1}{2}\left(\frac{a}{x_0}-x_0\right)\\
    \Longrightarrow x &=x_0+\Delta x \approx\frac{1}{2}\left(\frac{a}{x_0}+x_0\right) =x_1
\end{align*}$$

python:

    while abs(x*x-a) > 0.001:
        x = 1/2*(x+a/x)


### 解线性方程组的迭代法

谱,谱半径定义：$$\lambda$$为A的*特征值*, x为A对应$$\lambda$$的特征向量, A的全体特征值成为A的谱, 记做$$\sigma (A)$$, 记$$\rho (A)=\max_{1\leq i \leq n}\left\lvert\lambda_i\right\rvert$$ 为矩阵A的谱半径.

求解线性方程$$Ax=b$$, 其中$$A=(a_{ij})\in \mathbb{R^{n*n}}$$, 迭代求解步骤:

* 将A分裂为$$A=M-N$$, 其中M为非奇异矩阵, 且$$Mx = d$$容易求解. 于是$$Ax=b \Longrightarrow Mx=Nx +b$$, 即 $$x=M^{-1}Nx+M^{-1}b$$, 即$$x = Bx + f$$
 
* 构造一阶迭代 $$x^{(k+1)} = Bx^{(k)} + f, k = 0,1,...,$$ 其中 $$B = M^{-1}N=I-M^{-1}A, f=M^{-1}b$$, 称B为迭代矩阵

*定理*: 迭代法收敛的充要条件为矩阵B的谱半径$$\rho(B)<1$$, 不证.

*雅克比迭代法*
将A矩阵分解为$$A=D-L-U$$, 其中D为A的主对角线元素组成的矩阵, 其他位置为0, L为A的除对角线元素外的下三角矩阵的相反数组成的矩阵, U为A的除对角线元素外的上三角矩阵的相反数组成的矩阵

有$$A=D-N$$, $$B=I-D^{-1}A=D^{-1}(L+U)=J, f=D^{-1}b$$,  称J为解$$Ax=b$$的雅克比迭代法的迭代矩阵.

由迭代公式可得 $$Dx^{(k+1)} = (L+U)x^(k) + b$$,
或$$a_{ii}x_i^{(k+1)} = -\sum_{j=1}^{i-1}a_{ij}x_j^{(k)} - \sum_{j=i+1}^n a_{ij}x_j^{(k)}+b$$, 其中$$a_{ii}$$为主对角线的元素


还有非常多的矩阵分割方法来进行这个迭代过程

