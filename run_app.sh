#!/bin/bash
# zIMDB 验证页面启动脚本

# 检查是否安装了 uv
if ! command -v uv &> /dev/null; then
    echo "📦 uv 未安装，正在安装..."
    pip install uv
fi

# 检查模型文件是否存在
if [ ! -f "model.pt" ]; then
    echo "❌ 错误: 模型文件不存在 (model.pt)"
    echo "请先运行训练脚本生成模型:"
    echo "  bash run.sh"
    exit 1
fi

# 同步依赖
echo "📚 安装项目依赖..."
uv sync

# 启动 Flask 应用
echo "🚀 启动验证页面..."
echo "访问地址: http://localhost:5000"
echo "按 Ctrl+C 停止服务"
echo ""

uv run python app.py
