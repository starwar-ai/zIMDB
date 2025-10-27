# zIMDB - 基于PyTorch的IMDB情感分析项目

## 项目简介

zIMDB是一个使用PyTorch构建的深度学习项目，用于对IMDB电影评论进行情感分析。该项目使用LSTM神经网络模型来预测电影评论的情感倾向（正面或负面）。

## 功能特性

- 🎯 **情感分析**: 使用深度学习模型对电影评论进行情感分类
- 🚀 **GPU加速**: 自动检测并使用CUDA加速训练
- 📊 **数据预处理**: 自动化的文本清理和向量化流程
- 💾 **ModelScope集成**: 使用ModelScope加载数据集
- ⚙️ **简洁配置**: 直接在代码中配置超参数

## 技术栈

- **PyTorch**: 深度学习框架
- **ModelScope**: 数据集加载和管理
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

### 运行训练

```bash
# 使用 uv 运行训练脚本
uv run train.py
```

## 超参数配置

在 `train.py` 文件中可以直接修改以下超参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `VOCAB_SIZE` | 10000 | 词汇表大小 |
| `MAX_LEN` | 250 | 句子最大长度 |
| `EMBEDDING_DIM` | 128 | 词嵌入维度 |
| `HIDDEN_DIM` | 256 | LSTM隐藏层维度 |
| `NUM_EPOCHS` | 3 | 训练轮数 |
| `BATCH_SIZE` | 64 | 批次大小 |
| `LEARNING_RATE` | 0.001 | 学习率 |

## 代码结构

### 1. 数据加载
- 使用ModelScope加载IMDB数据集
- 自动缓存数据集到 `/mnt/data/.cache`，避免重复下载

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

## 许可证

本项目仅供学习和研究使用。

## 贡献

欢迎提交Issue和Pull Request来改进项目。