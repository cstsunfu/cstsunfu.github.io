---
layout: post
title: vim 的一些设置
date: 2017-10-24
tags: vim
---

#### markdown同步渲染，且支持mathjax

安装instant-markdown-d并在 "/usr/lib/node_modules/instant-markdown-d"中修改index.html文件。在<head></head>标签中间插入下面的代码
```bash
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});
</script>
```
再将index.html中相关代码替换为如下代码:
```bash
socket.on('newContent', function(newHTML) {                                  
document.querySelector(".markdown-body").innerHTML = newHTML;              
MathJax.Hub.Queue(["Typeset", MathJax.Hub]);                                                
});
```

$$1^2$$
