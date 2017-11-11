---
layout: post
title: 在fedora中安装windows的一些软件
date: 2017-11-10
tags: fedora
---
> 有几个windows的软件在linux中还是很需要的，下面以QQ为例，简单介绍下安装。由于deepin有一些对wine的优化，所以这里参考网上的方法也移植了deepin的wine

### 需要的软件包

到http://mirrors.ustc.edu.cn/deepin/pool/non-free/d/ 下载以下包
- deepin-wine_*_all.deb
- deepin-wine32_*_i386.deb
- deepin-libwine_*_i386.deb
- deepin-wine32-preloader_*_i386.deb


### 安装配置wine

安装系统的wine
```bash
dnf install wine
```
安装udis86,32位的那一个

将deepin-wine的软件包全部解压到一个文件夹中
将文件夹中的./usr/lib/deepin-wine和./usr/lib/i386-linux-gnu 复制到系统相应目录中(完全对应就行了)


### 安装配置tim

在ustc源下载deepin.com.qq.office_*deepin_i386.deb
将其中的files.7z 解压到$HOME/.deepinwine/Deepin-TIM 文件夹下
复制windows的字体到tim的安装目录

设置启动脚本

```bash
export GTK_IM_MODULE=fcitx
export XMODIFIERS=@im=fcitx
export QT_IM_MODULE=fcitx
LD_LIBRARY_PATH=/usr/lib/i386-linux-gnu/deepin-wine WINEPREFIX=$HOME/.deepinwine/Deepin-TIM WINELOADER=/usr/lib/deepin-wine/wine /usr/lib/deepin-wine/wine "c:\\Program Files\\Tencent\\TIM\\Bin\\TIM.exe"
```

将.deb包中的图标放在.local/share/icons文件夹，.desktop文件放在.local/share/applications文件夹下，且将.desktop文件中的Exec 指向上面的脚本

![](/images/posts/fedora/tim.png)
