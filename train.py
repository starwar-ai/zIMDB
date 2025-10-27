import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
from collections import Counter
import re
import time
import os
import argparse

# --- 1. DDP 初始化函数 ---
def setup_ddp():
    """初始化分布式训练环境"""
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    args = parser.parse_args()

    # 从环境变量获取分布式训练参数（torchrun 会自动设置这些）
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))

    # 初始化分布式进程组
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        print(f"Initialized DDP: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    else:
        local_rank = 0
        print("Running in single-GPU mode (no DDP)")

    return local_rank, world_size, rank

def cleanup_ddp():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

# --- 2. 设置缓存目录 ---
CACHE_DIR = '/mnt/data/.cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# --- 3. 定义超参数 ---
VOCAB_SIZE = 10000       # 词汇表大小
MAX_LEN = 250            # 句子最大长度
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
OUTPUT_DIM = 1
NUM_EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PAD_TOKEN = 0
UNK_TOKEN = 1

# 性能优化参数
USE_AMP = True                    # 启用混合精度训练
GRADIENT_ACCUMULATION_STEPS = 4   # 梯度累积步数 (有效batch = BATCH_SIZE * 4 = 256)
USE_GRADIENT_CHECKPOINTING = True # 启用梯度检查点节省显存

# 初始化 DDP
local_rank, world_size, rank = setup_ddp()

# 设置设备
device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device} (rank: {rank}/{world_size})")


# --- 3. 加载和处理数据 ---

if rank == 0:
    print("Loading IMDB dataset from ModelScope...")
    print(f"Cache directory: {CACHE_DIR}")

# 使用 ModelScope 加载数据集
# download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS
# 这是关键：如果数据集已存在于本地缓存中（设置为 /mnt/data/.cache），
# 它将直接重用，不会重复下载。
# cache_dir 参数指定数据集的缓存位置
data_dict = MsDataset.load(
    'imdb',
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    cache_dir=CACHE_DIR
)

if rank == 0:
    print("Dataset loaded.")

train_data = data_dict['train']
test_data = data_dict['test']

# 简单的文本清理函数
def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text) # 移除HTML标签
    text = re.sub(r'[^a-zA-Z\']', ' ', text) # 只保留字母和撇号
    text = text.lower()
    return text

# 构建词汇表
def build_vocab(data, vocab_size):
    word_counts = Counter()
    for item in data:
        text = clean_text(item['text'])
        word_counts.update(text.split())
    
    # 移除低频词，只保留前 (vocab_size - 2) 个最常见的词
    # 留出2个位置给 <PAD> 和 <UNK>
    vocab = {word: i+2 for i, (word, _) in enumerate(word_counts.most_common(vocab_size - 2))}
    vocab['<PAD>'] = PAD_TOKEN
    vocab['<UNK>'] = UNK_TOKEN
    return vocab

if rank == 0:
    print("Building vocabulary...")
vocab = build_vocab(train_data, VOCAB_SIZE)
if rank == 0:
    print(f"Vocabulary size: {len(vocab)}")

# 自定义 PyTorch Dataset
class ImdbDataset(Dataset):
    def __init__(self, data, vocab, max_len):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = clean_text(item['text'])
        label = 1 if item['label'] == 'pos' else 0
        
        # 文本转ID
        tokens = [self.vocab.get(word, UNK_TOKEN) for word in text.split()]
        
        # 填充或截断
        if len(tokens) < self.max_len:
            tokens.extend([PAD_TOKEN] * (self.max_len - len(tokens)))
        else:
            tokens = tokens[:self.max_len]
            
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.float)

# 创建 Dataset 和 DataLoader
train_dataset = ImdbDataset(train_data, vocab, MAX_LEN)
test_dataset = ImdbDataset(test_data, vocab, MAX_LEN)

# 使用 DistributedSampler 确保每个进程处理不同的数据
if world_size > 1:
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler)
else:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# --- 4. 定义 PyTorch 模型 (LSTM) ---

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx, use_checkpointing=False):
        super().__init__()

        # 嵌入层
        # padding_idx=pad_idx 告诉嵌入层 PAD_TOKEN 是填充符，训练时应忽略
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # LSTM 层
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=1,
                            bidirectional=False, # 设置为False，使用单向LSTM
                            batch_first=True) # 输入数据格式为 (batch_size, seq_len, features)

        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

        # 梯度检查点标志
        self.use_checkpointing = use_checkpointing

    def _forward_lstm(self, embedded):
        """LSTM前向传播的辅助函数，用于gradient checkpointing"""
        _, (hidden, _) = self.lstm(embedded)
        return hidden

    def forward(self, text):
        # text.shape = [batch_size, seq_len]

        embedded = self.embedding(text)
        # embedded.shape = [batch_size, seq_len, embedding_dim]

        # 使用gradient checkpointing节省显存
        if self.use_checkpointing and self.training:
            # checkpoint会在前向传播时不保存中间激活值，
            # 在反向传播时重新计算，从而节省显存
            hidden = checkpoint(self._forward_lstm, embedded, use_reentrant=False)
        else:
            # 正常前向传播
            _, (hidden, _) = self.lstm(embedded)

        # 我们使用最后一个隐藏状态 (从最后一个时间步)
        # hidden.squeeze(0) 的 shape 是 [batch_size, hidden_dim]
        hidden_last = hidden.squeeze(0)

        output = self.fc(hidden_last)
        # output.shape = [batch_size, output_dim]

        # 移除维度，使其与标签 (batch_size) 匹配
        return output.squeeze(1)


# --- 5. 实例化模型、损失函数和优化器 ---

model = SentimentLSTM(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, PAD_TOKEN,
                      use_checkpointing=USE_GRADIENT_CHECKPOINTING)
model.to(device)

# 使用 DistributedDataParallel 包装模型
if world_size > 1:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    if rank == 0:
        print(f"Model wrapped with DistributedDataParallel")

# 损失函数: BCEWithLogitsLoss
# (它内置了Sigmoid，比单独用Sigmoid+BCELoss更稳定)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 初始化混合精度训练的GradScaler
scaler = GradScaler() if USE_AMP else None

if rank == 0:
    print(f"\n性能优化设置:")
    print(f"  混合精度训练 (AMP): {USE_AMP}")
    print(f"  梯度累积步数: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  有效批次大小: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  梯度检查点: {USE_GRADIENT_CHECKPOINTING}")
    print(f"  分布式训练: {'DDP' if world_size > 1 else 'Single GPU'}")
    if world_size > 1:
        print(f"  全局批次大小: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * world_size}")

# --- 6. 训练和评估函数 ---

def calculate_accuracy(preds, y):
    """计算准确率"""
    # 将模型的原始输出 (logits) 通过 sigmoid 转换为 0-1 之间的概率
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() # 转换为 float
    acc = correct.sum() / len(correct)
    return acc

def train_epoch(model, loader, optimizer, criterion, scaler=None, accumulation_steps=1):
    epoch_loss = 0
    epoch_acc = 0

    model.train() # 设置为训练模式

    optimizer.zero_grad() # 初始梯度清零

    for idx, (text, labels) in enumerate(loader):
        text, labels = text.to(device), labels.to(device)

        # 使用混合精度训练
        if scaler is not None:
            # 使用autocast自动选择合适的精度
            with autocast():
                # 前向传播
                predictions = model(text)
                # 计算损失
                loss = criterion(predictions, labels)
                # 梯度累积：损失需要除以累积步数
                loss = loss / accumulation_steps

            # 计算准确率（在原始精度下）
            with torch.no_grad():
                acc = calculate_accuracy(predictions, labels)

            # 反向传播（使用scaler处理梯度缩放）
            scaler.scale(loss).backward()

            # 只在累积足够步数后更新权重
            if (idx + 1) % accumulation_steps == 0:
                # scaler会自动处理梯度缩放、unscale和梯度裁剪
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # 不使用混合精度的标准训练
            # 前向传播
            predictions = model(text)
            # 计算损失
            loss = criterion(predictions, labels)
            # 梯度累积：损失需要除以累积步数
            loss = loss / accumulation_steps

            # 计算准确率
            acc = calculate_accuracy(predictions, labels)

            # 反向传播
            loss.backward()

            # 只在累积足够步数后更新权重
            if (idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # 累积损失和准确率（注意这里loss已经除以了accumulation_steps）
        epoch_loss += loss.item() * accumulation_steps  # 恢复原始损失用于统计
        epoch_acc += acc.item()

    # 处理最后不完整的批次
    if len(loader) % accumulation_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    # 在分布式训练中，需要聚合所有进程的损失和准确率
    if dist.is_initialized():
        # 将损失和准确率转换为张量
        metrics = torch.tensor([epoch_loss, epoch_acc, len(loader)], device=device)
        # 所有进程的 metrics 求和
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        # 平均
        epoch_loss = metrics[0].item() / metrics[2].item()
        epoch_acc = metrics[1].item() / metrics[2].item()
    else:
        epoch_loss = epoch_loss / len(loader)
        epoch_acc = epoch_acc / len(loader)

    return epoch_loss, epoch_acc

def evaluate_epoch(model, loader, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval() # 设置为评估模式

    with torch.no_grad(): # 评估时不需要计算梯度
        for (text, labels) in loader:
            text, labels = text.to(device), labels.to(device)

            predictions = model(text)
            loss = criterion(predictions, labels)
            acc = calculate_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    # 在分布式训练中，需要聚合所有进程的损失和准确率
    if dist.is_initialized():
        # 将损失和准确率转换为张量
        metrics = torch.tensor([epoch_loss, epoch_acc, len(loader)], device=device)
        # 所有进程的 metrics 求和
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        # 平均
        epoch_loss = metrics[0].item() / metrics[2].item()
        epoch_acc = metrics[1].item() / metrics[2].item()
    else:
        epoch_loss = epoch_loss / len(loader)
        epoch_acc = epoch_acc / len(loader)

    return epoch_loss, epoch_acc


# --- 7. 运行训练循环 ---

if rank == 0:
    print("\nStarting training...")

for epoch in range(NUM_EPOCHS):
    # 设置 sampler 的 epoch（确保每个 epoch 的数据洗牌不同）
    if world_size > 1:
        train_sampler.set_epoch(epoch)

    start_time = time.time()

    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion,
                                       scaler=scaler,
                                       accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    test_loss, test_acc = evaluate_epoch(model, test_loader, criterion)

    end_time = time.time()
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

    if rank == 0:
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins:.0f}m {epoch_secs:.0f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\tTest. Loss: {test_loss:.3f} | Test. Acc: {test_acc*100:.2f}%')

if rank == 0:
    print("Training finished.")

# 清理分布式训练环境
cleanup_ddp()