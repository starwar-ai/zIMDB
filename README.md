# zIMDB - 基于PyTorch的IMDB情感分析项目

## 项目简介

zIMDB是一个使用PyTorch构建的深度学习项目，用于对IMDB电影评论进行情感分析。该项目使用LSTM神经网络模型来预测电影评论的情感倾向（正面或负面）。

## 功能特性

- 🎯 **情感分析**: 使用深度学习模型对电影评论进行情感分类
- 🚀 **GPU加速**: 自动检测并使用CUDA加速训练
- ⚡ **混合精度训练**: 使用AMP加速训练并减少显存占用
- 🔄 **梯度累积**: 通过梯度累积模拟更大的批次大小
- 💾 **梯度检查点**: 节省显存，支持更大模型训练
- 🌐 **分布式训练**: 支持多GPU训练（DistributedDataParallel）
- ☁️ **Kubeflow 支持**: 支持在 Kubeflow 上使用 4 个 T4 GPU 进行训练
- 📊 **数据预处理**: 自动化的文本清理和向量化流程
- 🌐 **Web验证界面**: 提供美观的Web界面进行实时情感分析
- ⚙️ **简洁配置**: 直接在代码中配置超参数

## 技术栈

- **PyTorch**: 深度学习框架
- **Flask**: Web框架用于验证界面
- **LSTM**: 长短期记忆网络用于序列建模
- **Embedding**: 词嵌入层用于文本向量化

## 模型架构

项目使用以下神经网络架构：

```
输入层 (文本) 
    ↓
词嵌入层 (Embedding) - 词汇表大小: 10,000, 嵌入维度: 128
    ↓
LSTM层 - 256个隐藏单元
    ↓
全连接层 - 1个神经元 (二分类)
```

## 安装依赖

本项目使用 `uv` 进行依赖管理。

### 安装 uv

```bash
# 使用 pip 安装
pip install uv

# 或使用 curl 安装（推荐）
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 安装项目依赖

```bash
# 安装项目依赖
uv sync
```

## 使用方法

### 单 GPU 训练

```bash
# 使用 uv 运行训练脚本
uv run train.py
```

### 多 GPU 分布式训练 (DDP)

项目现在支持使用 DistributedDataParallel (DDP) 进行多 GPU 训练，相比 DataParallel 提供更好的性能。

```bash
# 给运行脚本添加执行权限
chmod +x run_ddp.sh

# 运行多 GPU 训练（自动检测所有可用 GPU）
./run_ddp.sh

# 或者手动指定 GPU 数量
torchrun --nproc_per_node=2 train.py

# 或者使用 uv
uv run torchrun --nproc_per_node=2 train.py
```

**DDP 优势**:
- 更快的训练速度（相比 DataParallel）
- 更好的 GPU 利用率
- 支持跨节点训练
- 避免 Python GIL 限制

### Kubeflow 分布式训练

项目支持在 Kubeflow 上运行，可以使用一台节点中的 4 个 T4 GPU 进行训练。

```bash
# 快速部署到 Kubeflow
./deploy-kubeflow.sh your-registry.com

# 或手动部署
# 1. 构建并推送镜像
docker build -t your-registry.com/zimdb-training:latest .
docker push your-registry.com/zimdb-training:latest

# 2. 更新配置文件中的镜像地址
sed -i "s|<YOUR_REGISTRY>|your-registry.com|g" kubeflow-pytorchjob-single-node.yaml

# 3. 部署到 Kubeflow
kubectl apply -f kubeflow-pytorchjob-single-node.yaml

# 4. 查看训练日志
kubectl logs -f -n kubeflow $(kubectl get pods -n kubeflow -l app=zimdb-training -o jsonpath='{.items[0].metadata.name}')
```

**Kubeflow 特性**:
- 🚀 单节点 4 GPU 训练配置
- ☁️ 云原生部署
- 📊 资源管理和调度
- 🔄 自动化工作流
- 📈 可扩展性

详细的 Kubeflow 部署说明请参考 [KUBEFLOW.md](KUBEFLOW.md)

### Web 验证页面

训练完成后，可以启动一个美观的 Web 界面来验证模型：

```bash
# 启动验证页面
./run_app.sh

# 或使用 uv
uv run python app.py
```

然后访问 http://localhost:5000 来使用验证界面。

**验证页面功能**:
- 🎨 现代化的 UI 设计
- 📝 文本输入框输入电影评论
- ⚡ 实时情感分析
- 📊 显示置信度
- 💡 内置示例评论

## 超参数配置

在 `train.py` 文件中可以直接修改以下超参数：

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `VOCAB_SIZE` | 10000 | 词汇表大小 |
| `MAX_LEN` | 250 | 句子最大长度 |
| `EMBEDDING_DIM` | 128 | 词嵌入维度 |
| `HIDDEN_DIM` | 256 | LSTM隐藏层维度 |
| `NUM_EPOCHS` | 3 | 训练轮数 |
| `BATCH_SIZE` | 64 | 批次大小 |
| `LEARNING_RATE` | 0.001 | 学习率 |

### 性能优化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `USE_AMP` | True | 启用混合精度训练（需要GPU） |
| `GRADIENT_ACCUMULATION_STEPS` | 4 | 梯度累积步数，有效批次=64×4=256 |
| `USE_GRADIENT_CHECKPOINTING` | True | 启用梯度检查点节省显存 |

## 代码结构

### 1. 数据加载
- 从 Stanford AI Lab 直接下载 IMDB 数据集
- 自动下载、解压，并缓存到 `/mnt/data/.cache`，避免重复下载

### 2. 数据预处理
- 文本清理：移除HTML标签，保留字母和撇号
- 词汇表构建：统计词频，建立词汇映射
- 序列填充/截断：统一序列长度为250

### 3. 模型定义
- **SentimentLSTM**: LSTM情感分析模型
  - Embedding层：将词汇ID映射到向量
  - LSTM层：捕获长距离依赖关系
  - 全连接层：输出二分类结果

### 4. 训练流程
- 自动检测GPU设备
- 使用Adam优化器
- BCEWithLogitsLoss损失函数
- 训练和验证循环

## GPU支持

项目会自动检测并使用可用的GPU设备：

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

如果有NVIDIA GPU并正确安装了CUDA，训练将自动使用GPU加速。

## 性能优化详解

本项目实现了四种关键的训练优化技术：

### 1. 分布式数据并行 (DDP)

**作用**: 使用多个 GPU 并行训练，每个 GPU 处理不同的数据批次
**优势**:
- 🚀 训练速度随 GPU 数量线性增长
- 💪 避免 Python GIL 限制（使用多进程）
- 🎯 每个 GPU 维护独立的模型副本
- 📊 自动同步梯度和参数

**实现**: 使用 `torch.nn.parallel.DistributedDataParallel` 和 `DistributedSampler`

**使用方法**:
```bash
# 2 GPU 训练
torchrun --nproc_per_node=2 train.py

# 4 GPU 训练
torchrun --nproc_per_node=4 train.py
```

### 2. 混合精度训练 (AMP)

**作用**: 使用16位浮点数（FP16）进行计算，同时保持32位浮点数（FP32）的精度
**优势**:
- ⚡ 训练速度提升 1.5-3倍
- 💾 显存占用减少约 50%
- 📈 不影响模型准确度

**实现**: 使用 `torch.cuda.amp` 的 `autocast` 和 `GradScaler`

### 3. 梯度累积 (Gradient Accumulation)

**作用**: 多个小批次累积梯度后再更新参数，模拟更大的批次大小
**优势**:
- 📊 有效批次大小从 64 提升到 256（64 × 4）
- 🎯 更稳定的梯度，改善收敛性
- 💾 不增加显存占用

**实现**: 每 4 步累积后才调用 `optimizer.step()`

### 4. 梯度检查点 (Gradient Checkpointing)

**作用**: 在前向传播时不保存中间激活值，反向传播时重新计算
**优势**:
- 💾 显著减少显存占用（节省约 30-50%）
- 🚀 支持训练更大的模型或更大的批次
- ⏱️ 轻微增加训练时间（约 10-20%）

**实现**: 使用 `torch.utils.checkpoint.checkpoint` 包装 LSTM 前向传播

### 性能对比

| 配置 | 显存占用 | 训练速度 | 有效批次大小 |
|------|---------|---------|------------|
| 基础配置 (单GPU) | 100% | 1.0x | 64 |
| +AMP | 50% | 1.5-2x | 64 |
| +梯度累积 | 50% | 1.5-2x | 256 |
| +梯度检查点 | 35% | 1.2-1.6x | 256 |
| +DDP (2 GPUs) | 35% | 2.0-2.4x | 512 |
| +DDP (4 GPUs) | 35% | 3.5-4.5x | 1024 |

## 输出信息

训练过程中会显示：
- 使用的设备（CPU或GPU）
- 训练损失和准确率
- 测试损失和准确率
- 每个epoch的训练时间

## 注意事项

1. **内存要求**: 建议至少8GB RAM用于训练
2. **GPU支持**: 如有NVIDIA GPU，建议安装CUDA和cuDNN
3. **首次运行**: 首次运行会下载数据集，需要一些时间
4. **数据集缓存**: 数据集会自动缓存在 `/mnt/data/.cache`
5. **缓存目录**: 如需修改缓存路径，请编辑 `train.py` 中的 `CACHE_DIR` 变量
6. **DDP 训练**:
   - 使用 `torchrun` 启动多 GPU 训练
   - 确保所有 GPU 可见（`nvidia-smi` 检查）
   - 单 GPU 训练时无需使用 `torchrun`，直接运行 `python train.py` 即可

## 许可证

本项目仅供学习和研究使用。

## 贡献

欢迎提交Issue和Pull Request来改进项目。