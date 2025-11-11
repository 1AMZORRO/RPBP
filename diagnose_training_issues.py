#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练问题诊断脚本
诊断模型训练时损失不变、准确率0.5的问题
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

print("=" * 80)
print("训练问题诊断工具")
print("=" * 80)

base_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# 1. 检查标签与数据对齐
# ============================================================================
print("\n" + "=" * 80)
print("1. 标签与数据对齐检查")
print("=" * 80)

rna_fasta_path = os.path.join(base_dir, 'data/train/rna_sequences.fasta')
labels_path = os.path.join(base_dir, 'data/train/labels.txt')

# 读取RNA序列和从fasta中提取的标签
rna_count = 0
fasta_labels = []
fasta_proteins = []
with open(rna_fasta_path, 'r') as f:
    for line in f:
        if line.startswith('>'):
            # 格式: >12_AARS_K562_ENCSR825SVO_pos; chr21; class:1
            parts = line.strip().split(';')
            # 提取class标签
            for part in parts:
                if 'class:' in part:
                    label = int(part.split(':')[1].strip())
                    fasta_labels.append(label)
            # 提取蛋白质名称
            first_part = parts[0].strip()[1:].split('_')
            protein_name = '_'.join(first_part[1:-1])
            fasta_proteins.append(protein_name)
            rna_count += 1

# 读取labels.txt
txt_labels = []
with open(labels_path, 'r') as f:
    for line in f:
        txt_labels.append(int(line.strip()))

print(f"RNA序列数量: {rna_count}")
print(f"FASTA中提取的标签数量: {len(fasta_labels)}")
print(f"labels.txt中的标签数量: {len(txt_labels)}")

# 检查对齐
if len(fasta_labels) != len(txt_labels):
    print(f"❌ 错误: 标签数量不一致!")
else:
    mismatches = sum(1 for i in range(len(fasta_labels)) if fasta_labels[i] != txt_labels[i])
    if mismatches > 0:
        print(f"❌ 错误: 有 {mismatches} 个标签不匹配!")
        # 显示前几个不匹配的
        print("前10个不匹配的位置:")
        count = 0
        for i in range(len(fasta_labels)):
            if fasta_labels[i] != txt_labels[i]:
                print(f"  位置 {i}: FASTA={fasta_labels[i]}, TXT={txt_labels[i]}")
                count += 1
                if count >= 10:
                    break
    else:
        print(f"✓ 标签对齐正确")

# ============================================================================
# 2. 样本平衡性检查
# ============================================================================
print("\n" + "=" * 80)
print("2. 样本平衡性检查")
print("=" * 80)

label_counts = Counter(txt_labels)
total = len(txt_labels)
pos_count = label_counts[1]
neg_count = label_counts[0]
pos_ratio = pos_count / total

print(f"总样本数: {total}")
print(f"正样本 (label=1): {pos_count} ({pos_ratio*100:.2f}%)")
print(f"负样本 (label=0): {neg_count} ({(1-pos_ratio)*100:.2f}%)")
print(f"正负比例: 1:{neg_count/pos_count:.2f}")

if abs(pos_ratio - 0.5) > 0.2:
    print(f"⚠ 警告: 样本严重不平衡 (正样本率={pos_ratio*100:.1f}%)")
    print(f"建议: 在BCEWithLogitsLoss中使用pos_weight参数")
    pos_weight = neg_count / pos_count
    print(f"推荐的pos_weight值: {pos_weight:.4f}")
else:
    print(f"✓ 样本相对平衡")

# ============================================================================
# 3. 蛋白质Embedding检查
# ============================================================================
print("\n" + "=" * 80)
print("3. 蛋白质Embedding有效性检查")
print("=" * 80)

protein_emb_path = os.path.join(base_dir, 'data/train/protein_embeddings.pt')
if os.path.exists(protein_emb_path):
    protein_embeddings = torch.load(protein_emb_path)
    print(f"加载的蛋白质数量: {len(protein_embeddings)}")
    
    # 检查维度
    first_protein = list(protein_embeddings.keys())[0]
    first_emb = protein_embeddings[first_protein]
    print(f"Embedding维度: {first_emb.shape}")
    
    if first_emb.shape[0] != 1280:
        print(f"⚠ 警告: Embedding维度不是1280!")
    
    # 检查L2范数和差异性
    print(f"\n随机抽取5个蛋白质的embedding统计:")
    protein_names = list(protein_embeddings.keys())[:5]
    
    all_norms = []
    all_means = []
    all_stds = []
    
    for prot_name in protein_names:
        emb = protein_embeddings[prot_name]
        norm = torch.norm(emb, p=2).item()
        mean = emb.mean().item()
        std = emb.std().item()
        all_norms.append(norm)
        all_means.append(mean)
        all_stds.append(std)
        print(f"  {prot_name[:30]:30s} | L2范数: {norm:8.4f} | 均值: {mean:8.4f} | 标准差: {std:8.4f}")
    
    # 检查是否全零或几乎常数
    if max(all_norms) < 0.01:
        print(f"❌ 错误: 所有embedding几乎为零!")
    elif max(all_stds) < 0.01:
        print(f"❌ 错误: 所有embedding几乎为常数!")
    else:
        print(f"✓ Embedding看起来正常")
    
    # 检查不同蛋白质embedding的差异
    if len(protein_names) >= 2:
        emb1 = protein_embeddings[protein_names[0]]
        emb2 = protein_embeddings[protein_names[1]]
        diff = torch.norm(emb1 - emb2, p=2).item()
        similarity = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        print(f"\n前两个蛋白质embedding的差异:")
        print(f"  L2距离: {diff:.4f}")
        print(f"  余弦相似度: {similarity:.4f}")
        
        if diff < 0.1:
            print(f"❌ 错误: 不同蛋白质的embedding几乎相同!")
        else:
            print(f"✓ 不同蛋白质有明显差异")
else:
    print(f"❌ 错误: 蛋白质embedding文件不存在!")

# ============================================================================
# 4. RNA One-hot编码检查
# ============================================================================
print("\n" + "=" * 80)
print("4. RNA编码检查")
print("=" * 80)

from model import RNAEncoder

rna_encoder = RNAEncoder(vocab_size=5, embedding_dim=128)

# 读取几条RNA序列
test_sequences = []
with open(rna_fasta_path, 'r') as f:
    current_seq = []
    count = 0
    for line in f:
        line = line.strip()
        if line.startswith('>'):
            if current_seq:
                test_sequences.append(''.join(current_seq))
                count += 1
                if count >= 5:
                    break
            current_seq = []
        else:
            current_seq.append(line)

print(f"测试 {len(test_sequences)} 条RNA序列")

for i, seq in enumerate(test_sequences):
    # 统计字符分布
    char_counts = Counter(seq.upper())
    valid_chars = {'A', 'C', 'G', 'T', 'U'}
    invalid_chars = set(char_counts.keys()) - valid_chars - {'N'}
    
    print(f"\n序列 {i+1} (长度={len(seq)}):")
    print(f"  字符分布: {dict(char_counts.most_common(10))}")
    
    if invalid_chars:
        print(f"  ⚠ 警告: 发现非标准字符: {invalid_chars}")
    
    # 检查N的比例
    n_ratio = char_counts.get('N', 0) / len(seq)
    if n_ratio > 0.5:
        print(f"  ⚠ 警告: 序列中N字符占比过高 ({n_ratio*100:.1f}%)")
    
    # 编码并检查
    one_hot = rna_encoder.encode_sequence(seq)
    projected = rna_encoder.projection(one_hot)
    
    print(f"  One-hot编码后: shape={one_hot.shape}")
    print(f"  投影后: shape={projected.shape}, 均值={projected.mean().item():.6f}, 标准差={projected.std().item():.6f}")
    
    if projected.std().item() < 0.01:
        print(f"  ❌ 错误: 投影后的表示几乎为常数!")

# ============================================================================
# 5. 损失函数配置检查
# ============================================================================
print("\n" + "=" * 80)
print("5. 损失函数配置检查")
print("=" * 80)

# 检查train.py中的损失函数使用
train_file = os.path.join(base_dir, 'scripts/train.py')
with open(train_file, 'r') as f:
    train_code = f.read()

if 'BCEWithLogitsLoss' in train_code:
    print("✓ 使用BCEWithLogitsLoss")
    if 'sigmoid' in train_code and 'loss.backward()' in train_code:
        # 检查是否在计算损失前使用了sigmoid
        lines = train_code.split('\n')
        for i, line in enumerate(lines):
            if 'criterion(' in line and 'logits' in line:
                # 检查前几行是否有sigmoid
                context = '\n'.join(lines[max(0, i-3):i+1])
                if 'sigmoid' in context.lower():
                    print("⚠ 警告: 可能在BCEWithLogitsLoss前使用了sigmoid!")
                    break
elif 'BCELoss' in train_code:
    print("使用BCELoss")
    print("⚠ 注意: BCELoss需要先对logits进行sigmoid")

# 模拟检查损失值
print("\n模拟损失值检查:")
criterion = nn.BCEWithLogitsLoss()
# 随机预测的损失应该约为0.693
random_logits = torch.randn(100)
random_labels = torch.randint(0, 2, (100,)).float()
random_loss = criterion(random_logits, random_labels).item()
print(f"随机预测的期望损失: ~0.693")
print(f"实际随机损失: {random_loss:.4f}")
print(f"如果训练损失一直停留在0.69-0.70，说明模型没有学习")

# ============================================================================
# 6. 模型参数更新检查
# ============================================================================
print("\n" + "=" * 80)
print("6. 模型参数和梯度检查")
print("=" * 80)

from model import RNAProteinBindingModel

model = RNAProteinBindingModel()
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"总参数数: {total_params:,}")
print(f"可训练参数数: {trainable_params:,}")

if trainable_params == 0:
    print("❌ 错误: 没有可训练参数!")
else:
    print("✓ 有可训练参数")

# 检查各层的requires_grad
print("\n各层参数状态:")
for name, param in model.named_parameters():
    print(f"  {name:50s} | shape={str(param.shape):20s} | requires_grad={param.requires_grad}")

# ============================================================================
# 7. 小样本过拟合测试
# ============================================================================
print("\n" + "=" * 80)
print("7. 小样本过拟合能力测试")
print("=" * 80)

print("测试模型是否能够过拟合少量样本...")

# 创建小数据集
from dataset import RNAProteinDataset, collate_fn
from torch.utils.data import DataLoader

# 读取少量样本
small_rna = []
small_proteins = []
small_labels = []

with open(rna_fasta_path, 'r') as f:
    current_seq = []
    current_prot = None
    current_label = None
    count = 0
    
    for line in f:
        if count >= 64:  # 只用64个样本
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

if os.path.exists(protein_emb_path):
    protein_embeddings = torch.load(protein_emb_path)
    
    dataset = RNAProteinDataset(small_rna, small_proteins, small_labels, protein_embeddings)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    # 创建新模型并训练
    test_model = RNAProteinBindingModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(test_model.parameters(), lr=0.01)
    
    print(f"使用 {len(small_labels)} 个样本进行过拟合测试...")
    print(f"正样本: {sum(small_labels)}, 负样本: {len(small_labels) - sum(small_labels)}")
    
    initial_loss = None
    for epoch in range(50):
        test_model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            rna_seqs = batch['rna_sequences']
            prot_emb = batch['protein_embeddings']
            labels = batch['labels']
            
            optimizer.zero_grad()
            logits, _ = test_model(rna_seqs, prot_emb)
            loss = criterion(logits.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += len(labels)
        
        avg_loss = epoch_loss / len(dataloader)
        accuracy = correct / total
        
        if initial_loss is None:
            initial_loss = avg_loss
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:2d}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
    
    print(f"\n过拟合测试结果:")
    print(f"  初始损失: {initial_loss:.4f}")
    print(f"  最终损失: {avg_loss:.4f}")
    print(f"  最终准确率: {accuracy:.4f}")
    
    if avg_loss < 0.1 and accuracy > 0.95:
        print(f"✓ 模型能够过拟合小样本，基本功能正常")
    elif avg_loss < initial_loss * 0.5:
        print(f"✓ 损失有明显下降，模型在学习")
    else:
        print(f"❌ 模型无法过拟合小样本，存在严重问题!")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 80)
print("诊断总结")
print("=" * 80)
print("请查看上述各项检查结果，重点关注标记为❌和⚠的问题")
print("=" * 80)
