# Dockerfile for zIMDB Training on Kubeflow
# 支持多 GPU 分布式训练

# 使用 NVIDIA PyTorch 官方基础镜像，包含 CUDA 和 cuDNN
FROM nvcr.io/nvidia/pytorch:24.01-py3

# 设置工作目录
WORKDIR /workspace/zimdb

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv 包管理器 (更快的 pip 替代品)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# 复制项目文件
COPY pyproject.toml ./
COPY train.py ./
COPY app.py ./
COPY test_optimizations.py ./
COPY templates/ ./templates/
COPY run.sh ./
COPY run_ddp.sh ./
COPY run_app.sh ./
COPY README.md ./
COPY USAGE.md ./

# 安装 Python 依赖
RUN uv pip install --system -e .

# 创建数据缓存目录
RUN mkdir -p /mnt/data/.cache

# 设置默认命令 (在 Kubeflow 中会被覆盖)
CMD ["python", "train.py"]

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1
