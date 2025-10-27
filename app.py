import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
import re
import os

app = Flask(__name__)

# 定义 LSTM 模型类（与训练脚本一致）
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        hidden_last = hidden.squeeze(0)
        output = self.fc(hidden_last)
        return output.squeeze(1)

# 全局变量
model = None
vocab = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 250
PAD_TOKEN = 0
UNK_TOKEN = 1

def clean_text(text):
    """文本清理函数（与训练脚本一致）"""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z\']', ' ', text)
    text = text.lower()
    return text

def text_to_tokens(text, vocab):
    """将文本转换为token列表"""
    cleaned = clean_text(text)
    tokens = [vocab.get(word, UNK_TOKEN) for word in cleaned.split()]
    
    # 填充或截断
    if len(tokens) < MAX_LEN:
        tokens.extend([PAD_TOKEN] * (MAX_LEN - len(tokens)))
    else:
        tokens = tokens[:MAX_LEN]
    
    return torch.tensor([tokens], dtype=torch.long)

def load_model():
    """加载训练好的模型"""
    global model, vocab
    
    model_path = 'model.pt'
    
    if not os.path.exists(model_path):
        return False
    
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    vocab = checkpoint['vocab']
    vocab_size = checkpoint['vocab_size']
    embedding_dim = checkpoint['embedding_dim']
    hidden_dim = checkpoint['hidden_dim']
    output_dim = checkpoint['output_dim']
    pad_token = checkpoint['pad_token']
    
    # 创建模型
    model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, pad_token)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return True

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    if model is None or vocab is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        # 转换为tokens
        tokens = text_to_tokens(text, vocab)
        tokens = tokens.to(device)
        
        # 预测
        with torch.no_grad():
            output = model(tokens)
            probability = torch.sigmoid(output).item()
        
        # 结果
        prediction = 'positive' if probability > 0.5 else 'negative'
        confidence = probability if prediction == 'positive' else 1 - probability
        
        return jsonify({
            'prediction': prediction,
            'confidence': f'{confidence * 100:.2f}%',
            'probability': round(probability, 4)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 加载模型
    if load_model():
        print("Starting Flask app...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Error: Model file not found. Please train the model first.")
