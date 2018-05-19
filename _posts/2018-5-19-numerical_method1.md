---
layout: post
title: 数值分析(引论)
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


