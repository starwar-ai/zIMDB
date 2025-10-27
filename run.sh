#!/bin/bash
# zIMDB 项目运行脚本

# 检查是否安装了 uv
if ! command -v uv &> /dev/null; then
    echo "📦 uv 未安装，正在安装..."
    pip install uv
fi

# 同步依赖
echo "📚 安装项目依赖..."
uv sync

# 运行训练
echo "🚀 开始训练..."
uv run train.py
