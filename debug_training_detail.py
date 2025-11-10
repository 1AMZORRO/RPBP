#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深入调试训练过程 - 检查梯度、中间输出和数值稳定性
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
from model import RNAProteinBindingModel
from dataset import RNAProteinDataset, collate_fn
from torch.utils.data import DataLoader

print("=" * 80)
print("深入训练调试")
print("=" * 80)

base_dir = os.path.dirname(os.path.abspath(__file__))

# 加载少量数据
print("\n加载小数据集...")
rna_fasta_path = os.path.join(base_dir, 'data/train/rna_sequences.fasta')
protein_emb_path = os.path.join(base_dir, 'data/train/protein_embeddings.pt')

small_rna = []
small_proteins = []
small_labels = []

with open(rna_fasta_path, 'r') as f:
    current_seq = []
    current_prot = None
    current_label = None
    count = 0
    
    for line in f:
        if count >= 32:  # 只用32个样本
            break
        
        line = line.strip()
        if line.startswith('>'):
            if current_prot is not None and current_seq:
                small_rna.append(''.join(current_seq))
                small_proteins.append(current_prot)
                small_labels.append(current_label)
                count += 1
            
            parts = line[1:].split(';')
            first_part = parts[0].strip().split('_')
            current_prot = '_'.join(first_part[1:-1])
            
            for part in parts:
                if 'class:' in part:
                    current_label = int(part.split(':')[1].strip())
            
            current_seq = []
        else:
            current_seq.append(line)

protein_embeddings = torch.load(protein_emb_path)
dataset = RNAProteinDataset(small_rna, small_proteins, small_labels, protein_embeddings)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

print(f"加载了 {len(small_labels)} 个样本")
print(f"正样本: {sum(small_labels)}, 负样本: {len(small_labels) - sum(small_labels)}")

# 创建模型
model = RNAProteinBindingModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\n" + "=" * 80)
print("第一个batch的详细分析")
print("=" * 80)

# 获取第一个batch
batch = next(iter(dataloader))
rna_seqs = batch['rna_sequences']
prot_emb = batch['protein_embeddings']
labels = batch['labels']

print(f"\nBatch信息:")
print(f"  RNA序列数: {len(rna_seqs)}")
print(f"  蛋白质embedding shape: {prot_emb.shape}")
print(f"  标签: {labels.numpy()}")

# 前向传播并检查中间输出
model.eval()
with torch.no_grad():
    # RNA编码
    print("\n1. RNA编码阶段:")
    rna_encoded = model.rna_encoder(rna_seqs)
    print(f"   输出shape: {rna_encoded.shape}")
    print(f"   均值: {rna_encoded.mean().item():.6f}")
    print(f"   标准差: {rna_encoded.std().item():.6f}")
    print(f"   最小值: {rna_encoded.min().item():.6f}")
    print(f"   最大值: {rna_encoded.max().item():.6f}")
    
    # Cross-Attention
    print("\n2. Cross-Attention阶段:")
    attended, attn_weights = model.cross_attention(rna_encoded, prot_emb, return_attention=True)
    print(f"   输出shape: {attended.shape}")
    print(f"   均值: {attended.mean().item():.6f}")
    print(f"   标准差: {attended.std().item():.6f}")
    print(f"   最小值: {attended.min().item():.6f}")
    print(f"   最大值: {attended.max().item():.6f}")
    
    if attn_weights is not None:
        print(f"   注意力权重shape: {attn_weights.shape}")
        print(f"   注意力权重统计:")
        print(f"     均值: {attn_weights.mean().item():.6f}")
        print(f"     标准差: {attn_weights.std().item():.6f}")
        print(f"     最小值: {attn_weights.min().item():.6f}")
        print(f"     最大值: {attn_weights.max().item():.6f}")
        # 检查注意力是否退化为均匀分布
        # 对于单个Key，所有注意力权重应该是1.0
        print(f"     第一个样本的注意力权重: {attn_weights[0, 0, :5, 0].numpy()}")
    
    # Layer Norm
    print("\n3. Layer Normalization阶段:")
    normalized = model.layer_norm(attended)
    print(f"   输出shape: {normalized.shape}")
    print(f"   均值: {normalized.mean().item():.6f}")
    print(f"   标准差: {normalized.std().item():.6f}")
    
    # 池化
    print("\n4. 全局平均池化阶段:")
    pooled = normalized.mean(dim=1)
    print(f"   输出shape: {pooled.shape}")
    print(f"   均值: {pooled.mean().item():.6f}")
    print(f"   标准差: {pooled.std().item():.6f}")
    
    # 分类器
    print("\n5. 分类器阶段:")
    logits, _ = model(rna_seqs, prot_emb)
    print(f"   Logits shape: {logits.shape}")
    print(f"   Logits 值: {logits.squeeze().numpy()}")
    print(f"   Logits 统计:")
    print(f"     均值: {logits.mean().item():.6f}")
    print(f"     标准差: {logits.std().item():.6f}")
    print(f"     最小值: {logits.min().item():.6f}")
    print(f"     最大值: {logits.max().item():.6f}")
    
    # 预测概率
    probs = torch.sigmoid(logits.squeeze())
    print(f"\n   预测概率: {probs.numpy()}")
    print(f"   真实标签: {labels.numpy()}")
    
    # 损失
    loss = criterion(logits.squeeze(), labels)
    print(f"\n   损失值: {loss.item():.6f}")

# 训练模式 - 检查梯度
print("\n" + "=" * 80)
print("梯度检查（训练一步）")
print("=" * 80)

model.train()
optimizer.zero_grad()
logits, _ = model(rna_seqs, prot_emb)
loss = criterion(logits.squeeze(), labels)
loss.backward()

print(f"\n损失值: {loss.item():.6f}")
print(f"\n各层梯度范数:")

layer_gradients = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        layer_gradients[name] = grad_norm
        if grad_norm > 0.001 or 'bias' in name:  # 只显示重要的
            print(f"  {name:50s}: {grad_norm:.6f}")

# 检查是否有梯度消失
print(f"\n梯度消失检查:")
small_grads = [name for name, grad in layer_gradients.items() if grad < 1e-6]
if small_grads:
    print(f"  以下层梯度过小 (<1e-6): {len(small_grads)} 层")
    for name in small_grads[:5]:
        print(f"    {name}")
else:
    print(f"  ✓ 无明显梯度消失")

# 检查是否有梯度爆炸
large_grads = [name for name, grad in layer_gradients.items() if grad > 100]
if large_grads:
    print(f"  以下层梯度过大 (>100): {len(large_grads)} 层")
    for name in large_grads:
        print(f"    {name}: {layer_gradients[name]:.2f}")
else:
    print(f"  ✓ 无梯度爆炸")

# 执行优化步骤并检查参数更新
print("\n" + "=" * 80)
print("参数更新检查")
print("=" * 80)

# 保存更新前的参数
param_before = {name: param.clone() for name, param in model.named_parameters()}

optimizer.step()

# 检查参数变化
print(f"\n参数更新幅度:")
max_update = 0
min_update = float('inf')
total_update = 0
num_params = 0

for name, param in model.named_parameters():
    if name in param_before:
        update = (param - param_before[name]).abs().max().item()
        max_update = max(max_update, update)
        min_update = min(min_update, update)
        total_update += (param - param_before[name]).abs().mean().item()
        num_params += 1

print(f"  最大更新幅度: {max_update:.8f}")
print(f"  最小更新幅度: {min_update:.8f}")
print(f"  平均更新幅度: {total_update / num_params:.8f}")

if max_update < 1e-6:
    print(f"  ❌ 参数几乎没有更新!")
elif max_update > 1:
    print(f"  ⚠ 参数更新过大，可能需要降低学习率")
else:
    print(f"  ✓ 参数更新幅度正常")

# 多步训练测试
print("\n" + "=" * 80)
print("连续训练10步测试")
print("=" * 80)

model = RNAProteinBindingModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 提高学习率

losses = []
accuracies = []

for step in range(10):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        rna_seqs = batch['rna_sequences']
        prot_emb = batch['protein_embeddings']
        labels = batch['labels']
        
        optimizer.zero_grad()
        logits, _ = model(rna_seqs, prot_emb)
        loss = criterion(logits.squeeze(), labels)
        loss.backward()
        
        # 检查梯度
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        
        optimizer.step()
        
        total_loss += loss.item()
        preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(labels)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    losses.append(avg_loss)
    accuracies.append(accuracy)
    
    print(f"  Step {step+1}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}, GradNorm={grad_norm:.4f}")

# 分析训练曲线
print(f"\n训练曲线分析:")
print(f"  初始损失: {losses[0]:.4f}")
print(f"  最终损失: {losses[-1]:.4f}")
print(f"  损失下降: {losses[0] - losses[-1]:.4f} ({(losses[0] - losses[-1])/losses[0]*100:.1f}%)")
print(f"  初始准确率: {accuracies[0]:.4f}")
print(f"  最终准确率: {accuracies[-1]:.4f}")

if losses[-1] < losses[0] * 0.8:
    print(f"  ✓ 损失有下降，模型在学习")
else:
    print(f"  ❌ 损失下降不明显，模型学习困难")

print("\n" + "=" * 80)
print("调试完成")
print("=" * 80)
