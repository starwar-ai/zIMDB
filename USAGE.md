# zIMDB 快速使用指南

## 📋 目录

1. [安装依赖](#安装依赖)
2. [训练模型](#训练模型)
3. [使用验证页面](#使用验证页面)
4. [常见问题](#常见问题)

## 安装依赖

### 使用 uv（推荐）

```bash
# 安装 uv
pip install uv

# 同步项目依赖
uv sync
```

### 使用 pip

```bash
pip install torch flask requests
```

## 训练模型

### 单 GPU 训练

```bash
# 方式1: 使用脚本
bash run.sh

# 方式2: 使用 uv
uv run train.py

# 方式3: 直接运行
python train.py
```

### 多 GPU 分布式训练

```bash
# 自动检测所有可用 GPU
./run_ddp.sh

# 或手动指定 GPU 数量
torchrun --nproc_per_node=4 train.py
```

### 训练输出

训练完成后，会生成以下文件：
- `model.pt` - 训练好的模型文件（包含模型权重和词汇表）

## 使用验证页面

### 启动验证页面

```bash
# 方式1: 使用脚本（推荐）
./run_app.sh

# 方式2: 使用 uv
uv run python app.py

# 方式3: 直接运行
python app.py
```

### 访问验证界面

启动后，在浏览器中访问：**http://localhost:5000**

### 使用步骤

1. 在文本框中输入电影评论
2. 点击"分析情感"按钮
3. 查看预测结果和置信度

### 快速测试

页面内置了4个示例评论，点击即可自动填充：

- **正面评价**: "This movie is absolutely fantastic! The acting is superb and the plot is engaging."
- **负面评价**: "What a terrible film. Boring, poorly acted, and completely waste of time."
- **正面评价**: "I loved every minute of this movie! The cinematography is stunning and the story is heartwarming."
- **负面评价**: "Disappointing. The movie had potential but failed to deliver. Very slow and uninteresting."

## 常见问题

### Q1: 训练报错 "Model file not found"

**A**: 需要先训练模型生成 `model.pt` 文件。

```bash
# 先运行训练
bash run.sh

# 然后再启动验证页面
./run_app.sh
```

### Q2: 如何修改验证页面的端口？

**A**: 修改 `app.py` 文件中的端口配置：

```python
app.run(host='0.0.0.0', port=8080, debug=True)  # 改为8080端口
```

### Q3: 如何关闭验证页面？

**A**: 在运行验证页面的终端窗口按 `Ctrl+C`

### Q4: 数据集下载失败怎么办？

**A**: 数据集会自动从 Stanford AI Lab 下载并缓存。如果下载失败，可以：

1. 检查网络连接
2. 手动下载数据集并解压到 `/mnt/data/.cache/aclImdb/`
3. 数据集URL: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

### Q5: 如何在远程服务器上使用验证页面？

**A**: 如果服务器开启了SSH端口转发：

```bash
# 在本地终端运行
ssh -L 5000:localhost:5000 user@remote-server

# 然后在服务器上启动验证页面
./run_app.sh

# 在本地浏览器访问 http://localhost:5000
```

### Q6: 如何增加模型性能？

**A**: 在 `train.py` 中调整超参数：

```python
VOCAB_SIZE = 20000      # 增加词汇表大小
EMBEDDING_DIM = 256     # 增加嵌入维度
HIDDEN_DIM = 512        # 增加LSTM隐藏层
NUM_EPOCHS = 10         # 增加训练轮数
```

## 项目结构

```
zIMDB/
├── train.py              # 训练脚本
├── app.py                # Flask验证应用
├── run.sh                # 单GPU训练脚本
├── run_ddp.sh            # 多GPU训练脚本
├── run_app.sh            # 验证页面启动脚本
├── templates/
│   └── index.html        # 验证页面HTML
├── model.pt              # 训练好的模型（需要先训练）
├── pyproject.toml        # 项目配置
└── README.md             # 项目说明
```

## 下一步

- 📖 阅读 [README.md](README.md) 了解更多项目细节
- 🔧 修改超参数尝试不同的模型配置
- 🎨 自定义验证页面的样式
- 📊 添加更多的性能指标和可视化
