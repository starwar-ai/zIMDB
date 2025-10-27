#!/bin/bash

# DDP 训练启动脚本
# 使用 torchrun 启动多 GPU 分布式训练

# 设置 GPU 数量（根据你的系统自动检测或手动指定）
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

# 如果没有检测到 GPU，默认为 1
if [ $NUM_GPUS -eq 0 ]; then
    NUM_GPUS=1
fi

echo "使用 $NUM_GPUS 个 GPU 进行分布式训练"

# 使用 torchrun 启动分布式训练
# --nproc_per_node: 每个节点的进程数（通常等于 GPU 数量）
# --master_port: master 进程的端口（可以自定义）
torchrun --nproc_per_node=$NUM_GPUS \
         --master_port=29500 \
         train.py

# 或者，如果你只想使用特定数量的 GPU，可以手动指定：
# torchrun --nproc_per_node=2 --master_port=29500 train.py

# 单 GPU 模式（不使用 DDP）：
# python train.py
