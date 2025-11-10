#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小规模训练测试 - 验证完整训练流程
使用少量数据快速验证训练能否正常运行
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
from model import RNAProteinBindingModel
from dataset import RNAProteinDataset, collate_fn

print("=" * 60)
print("小规模训练测试")
print("=" * 60)

# 设置
device = 'cpu'  # 使用CPU测试
num_samples = 10000  # 增大到10000个样本进行测试
num_epochs = 10  # 增加训练轮数到10
batch_size = 32  # 增大批次大小

print(f"\n配置:")
print(f"  设备: {device}")
print(f"  样本数: {num_samples}")
print(f"  批次大小: {batch_size}")
print(f"  训练轮数: {num_epochs}")
print(f"  预计训练时间: 约5-10分钟")

# 步骤1: 读取真实数据（只取前100个样本）
print("\n步骤1: 加载数据")
base_dir = os.path.dirname(os.path.abspath(__file__))

# 读取RNA序列
rna_sequences = []
protein_names = []
with open(os.path.join(base_dir, 'data/train/rna_sequences.fasta'), 'r') as f:
    seq_count = 0
    current_seq = []
    current_prot = None
    
    for line in f:
        if seq_count >= num_samples:
            break
        
        line = line.strip()
        if line.startswith('>'):
            if current_prot is not None:
                rna_sequences.append(''.join(current_seq))
                protein_names.append(current_prot)
                seq_count += 1
            
            # 解析蛋白质名称
            parts = line[1:].split(';')[0].strip().split('_')
            current_prot = '_'.join(parts[1:-1])
            current_seq = []
        else:
            current_seq.append(line)
    
    # 最后一个序列
    if current_prot is not None and seq_count < num_samples:
        rna_sequences.append(''.join(current_seq))
        protein_names.append(current_prot)

# 读取标签
labels = []
with open(os.path.join(base_dir, 'data/train/labels.txt'), 'r') as f:
    for i, line in enumerate(f):
        if i >= num_samples:
            break
        labels.append(int(line.strip()))

print(f"  加载了 {len(rna_sequences)} 个RNA序列")
print(f"  涉及 {len(set(protein_names))} 个不同蛋白质")
print(f"  正样本: {sum(labels)}, 负样本: {len(labels) - sum(labels)}")

# 步骤2: 创建模拟蛋白质embeddings（实际训练时需要用ESM2计算）
print("\n步骤2: 创建模拟蛋白质embeddings")
unique_proteins = list(set(protein_names))
protein_embeddings = {prot: torch.randn(1280) for prot in unique_proteins}
print(f"  创建了 {len(protein_embeddings)} 个蛋白质的embeddings")

# 步骤3: 划分训练集和验证集
print("\n步骤3: 划分数据集")
train_size = int(0.8 * num_samples)
indices = np.arange(num_samples)
np.random.shuffle(indices)

train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_data = (
    [rna_sequences[i] for i in train_indices],
    [protein_names[i] for i in train_indices],
    [labels[i] for i in train_indices]
)
val_data = (
    [rna_sequences[i] for i in val_indices],
    [protein_names[i] for i in val_indices],
    [labels[i] for i in val_indices]
)

print(f"  训练集: {len(train_indices)} 样本")
print(f"  验证集: {len(val_indices)} 样本")

# 步骤4: 创建数据加载器
print("\n步骤4: 创建数据加载器")
train_dataset = RNAProteinDataset(*train_data, protein_embeddings)
val_dataset = RNAProteinDataset(*val_data, protein_embeddings)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

print(f"  训练批次数: {len(train_loader)}")
print(f"  验证批次数: {len(val_loader)}")

# 步骤5: 创建模型
print("\n步骤5: 创建模型")
model = RNAProteinBindingModel(
    rna_vocab_size=5,
    rna_embedding_dim=128,
    protein_embedding_dim=1280,
    num_attention_heads=8,
    attention_hidden_dim=256,
    classifier_hidden_dims=[512, 256, 128]
)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"  模型参数总数: {total_params:,}")

# 步骤6: 设置训练
print("\n步骤6: 设置损失函数和优化器")
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 步骤7: 训练循环
print("\n步骤7: 开始训练")
print("=" * 60)

for epoch in range(num_epochs):
    # 训练
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [训练]"):
        rna_seqs = batch['rna_sequences']
        prot_emb = batch['protein_embeddings'].to(device)
        batch_labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        logits, _ = model(rna_seqs, prot_emb)
        loss = criterion(logits.squeeze(), batch_labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()
        train_correct += (preds == batch_labels).sum().item()
        train_total += len(batch_labels)
    
    # 验证
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [验证]"):
            rna_seqs = batch['rna_sequences']
            prot_emb = batch['protein_embeddings'].to(device)
            batch_labels = batch['labels'].to(device)
            
            logits, _ = model(rna_seqs, prot_emb)
            loss = criterion(logits.squeeze(), batch_labels)
            
            val_loss += loss.item()
            preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()
            val_correct += (preds == batch_labels).sum().item()
            val_total += len(batch_labels)
    
    # 打印结果
    train_acc = train_correct / train_total
    val_acc = val_correct / val_total
    print(f"\nEpoch {epoch+1}:")
    print(f"  训练 - 损失: {train_loss/len(train_loader):.4f}, 准确率: {train_acc:.4f}")
    print(f"  验证 - 损失: {val_loss/len(val_loader):.4f}, 准确率: {val_acc:.4f}")

print("\n" + "=" * 60)
print("训练测试完成！✓")
print("完整训练流程验证成功")
print("=" * 60)
print("\n下一步:")
print("1. 运行 python scripts/precompute_protein_embeddings.py 预计算真实的蛋白质embeddings")
print("2. 运行 python scripts/train.py 进行完整训练")
print("3. 运行 python scripts/predict.py 进行预测和可视化")
