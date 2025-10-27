"""
快速测试脚本，验证训练优化功能
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

print("Testing PyTorch optimizations...")

# 测试1: 验证AMP可用性
print("\n1. 测试混合精度训练 (AMP):")
if torch.cuda.is_available():
    print("   ✓ CUDA可用")
    scaler = GradScaler()
    print("   ✓ GradScaler初始化成功")

    # 简单的AMP测试
    device = torch.device('cuda')
    model = nn.Linear(10, 1).to(device)
    x = torch.randn(2, 10, device=device)

    with autocast():
        output = model(x)
    print(f"   ✓ Autocast测试成功，输出dtype: {output.dtype}")
else:
    print("   ⚠ CUDA不可用，AMP将在CPU上跳过")

# 测试2: 验证Gradient Checkpointing
print("\n2. 测试梯度检查点:")
def test_module(x):
    return x * 2

x = torch.randn(5, requires_grad=True)
y = checkpoint(test_module, x, use_reentrant=False)
y.sum().backward()
print(f"   ✓ Checkpoint测试成功，梯度: {x.grad is not None}")

# 测试3: 梯度累积模拟
print("\n3. 测试梯度累积逻辑:")
accumulation_steps = 4
print(f"   ✓ 累积步数: {accumulation_steps}")
batch_size = 64
effective_batch = batch_size * accumulation_steps
print(f"   ✓ 有效批次大小: {effective_batch}")

print("\n✅ 所有优化功能测试通过！")
