---
layout: post
title: Fedora 系统的一些设置
date: 2017-10-23
tags: Fedora   
---
**目的**

## tmux

```bash
#设置前缀为Ctrl + a
set -g prefix C-a
#解除Ctrl+b 与前缀的对应关系
unbind C-b
set-option -g mouse on

#up
bind-key k select-pane -U
#down
bind-key j select-pane -D
#left
bind-key h select-pane -L
#right
bind-key l select-pane -R
#select last window
bind-key C-l select-window -l

#copy-mode 将快捷键设置为vi 模式
setw -g mode-keys vi
# split window
unbind '"'
# vertical split (prefix -)
bind - splitw -v
unbind %
bind | splitw -h # horizontal split (prefix |)



#statusline
set-option -g status off
set-option -g status-interval 0
set-option -g status-justify "centre"
set-option -g status-left-length 60
set-option -g status-right-length 90
set-option -g status-left "#(~/.tmux-powerline/tmux-powerline/powerline.sh left)"
#set-option -g status-right "#(~/.tmux-powerline/tmux-powerline/powerline.sh right)"
set-window-option -g window-status-current-format "#[fg=colour235, bg=colour27]⮀#[fg=colour255, bg=colour27] #I ⮁ #W #[fg=colour27, bg=colour235]⮀"


set -g pane-border-style fg=magenta
set -g pane-active-border-style fg=magenta
set -g pane-active-border-style bg=default
```

以上文件保存为 .tmux.conf
## bashrc

```bash

# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
alias  pdf='evince&'
alias  vim='vimx'
export EDITOR=vim
##export TERM=xterm-256color
#export TERM=xterm+256colors
set completion-ignore-case on
PS1="[\[\e[36;1m\]\u@\[\e[32;1m\]\h\[\e[0m\] \[\e[35;1m\]\W\e[m\]]$"
```


## dircolor
```bash
git clone git://github.com/seebi/dircolors-solarized.git
cp ~/dircolors-solarized/dircolors.256dark ~/.dircolors
```



## git
为避免每次push 到github都要输入账号密码，可以设置明文保存在.git-credentials中(有风险)

只需执行一下代码即可自动保存
```bash
git config --global credential.helper store
```
