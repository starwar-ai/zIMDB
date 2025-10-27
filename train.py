import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
from collections import Counter
import re
import time
import os

# --- 1. 设置缓存目录 ---
CACHE_DIR = '/mnt/data/.cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# --- 2. 定义超参数 ---
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

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# --- 3. 加载和处理数据 ---

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

print("Building vocabulary...")
vocab = build_vocab(train_data, VOCAB_SIZE)
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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# --- 4. 定义 PyTorch 模型 (LSTM) ---

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
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
        
    def forward(self, text):
        # text.shape = [batch_size, seq_len]
        
        embedded = self.embedding(text)
        # embedded.shape = [batch_size, seq_len, embedding_dim]
        
        # lstm_out, (hidden, cell) = self.lstm(embedded)
        # 我们只关心最后一个时间步的隐藏状态
        # hidden.shape = [num_layers * num_directions, batch_size, hidden_dim]
        _, (hidden, _) = self.lstm(embedded)
        
        # 我们使用最后一个隐藏状态 (从最后一个时间步)
        # hidden.squeeze(0) 的 shape 是 [batch_size, hidden_dim]
        hidden_last = hidden.squeeze(0)
        
        output = self.fc(hidden_last)
        # output.shape = [batch_size, output_dim]
        
        # 移除维度，使其与标签 (batch_size) 匹配
        return output.squeeze(1)


# --- 5. 实例化模型、损失函数和优化器 ---

model = SentimentLSTM(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, PAD_TOKEN)
model.to(device)

# 损失函数: BCEWithLogitsLoss 
# (它内置了Sigmoid，比单独用Sigmoid+BCELoss更稳定)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 6. 训练和评估函数 ---

def calculate_accuracy(preds, y):
    """计算准确率"""
    # 将模型的原始输出 (logits) 通过 sigmoid 转换为 0-1 之间的概率
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() # 转换为 float
    acc = correct.sum() / len(correct)
    return acc

def train_epoch(model, loader, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train() # 设置为训练模式
    
    for (text, labels) in loader:
        text, labels = text.to(device), labels.to(device)
        
        # 1. 梯度清零
        optimizer.zero_grad()
        
        # 2. 前向传播
        predictions = model(text)
        
        # 3. 计算损失
        loss = criterion(predictions, labels)
        
        # 4. 计算准确率
        acc = calculate_accuracy(predictions, labels)
        
        # 5. 反向传播
        loss.backward()
        
        # 6. 更新权重
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(loader), epoch_acc / len(loader)

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
            
    return epoch_loss / len(loader), epoch_acc / len(loader)


# --- 7. 运行训练循环 ---

print("\nStarting training...")
for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    test_loss, test_acc = evaluate_epoch(model, test_loader, criterion)
    
    end_time = time.time()
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins:.0f}m {epoch_secs:.0f}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\tTest. Loss: {test_loss:.3f} | Test. Acc: {test_acc*100:.2f}%')

print("Training finished.")