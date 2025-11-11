#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建测试用的蛋白质embeddings
用于诊断训练问题（非真实ESM2 embeddings）
"""

import os
import torch
from Bio import SeqIO

print("=" * 80)
print("创建测试用蛋白质Embeddings")
print("=" * 80)

base_dir = os.path.dirname(os.path.abspath(__file__))
prot_fasta = os.path.join(base_dir, 'data/raw/prot_seqs.fasta')
output_path = os.path.join(base_dir, 'data/train/protein_embeddings.pt')

# 读取蛋白质序列
protein_seqs = {}
for record in SeqIO.parse(prot_fasta, "fasta"):
    protein_name = record.id
    protein_seq = str(record.seq)
    protein_seqs[protein_name] = protein_seq

print(f"读取了 {len(protein_seqs)} 个蛋白质序列")

# 为每个蛋白质创建不同的embedding
# 使用蛋白质序列的特征来生成有意义的差异
protein_embeddings = {}

for i, (protein_name, protein_seq) in enumerate(protein_seqs.items()):
    # 使用蛋白质索引作为种子，确保每个蛋白质有不同的embedding
    torch.manual_seed(hash(protein_name) % (2**32))
    
    # 生成随机embedding，但基于蛋白质序列的一些特征进行调整
    base_emb = torch.randn(1280) * 0.5  # 降低方差，更接近真实ESM2
    
    # 添加基于序列特征的偏移，使不同蛋白质有明显差异
    seq_len = len(protein_seq)
    seq_feature = torch.tensor([
        seq_len / 1000.0,  # 序列长度归一化
        protein_seq.count('A') / seq_len,  # 氨基酸组成
        protein_seq.count('C') / seq_len,
        protein_seq.count('D') / seq_len,
        protein_seq.count('E') / seq_len,
    ])
    
    # 将序列特征扩展到1280维并加到embedding上
    # 这样不同蛋白质会有不同的embedding
    feature_expansion = torch.zeros(1280)
    for j in range(5):
        feature_expansion[j::5] += seq_feature[j] * 2.0
    
    embedding = base_emb + feature_expansion
    
    # 归一化到合理范围
    embedding = embedding / torch.norm(embedding) * 30.0  # L2范数约为30
    
    protein_embeddings[protein_name] = embedding

# 验证embeddings的差异性
protein_names = list(protein_embeddings.keys())
if len(protein_names) >= 2:
    emb1 = protein_embeddings[protein_names[0]]
    emb2 = protein_embeddings[protein_names[1]]
    diff = torch.norm(emb1 - emb2).item()
    similarity = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    
    print(f"\n验证embedding差异性:")
    print(f"  蛋白质1: {protein_names[0]}")
    print(f"  蛋白质2: {protein_names[1]}")
    print(f"  L2距离: {diff:.4f}")
    print(f"  余弦相似度: {similarity:.4f}")
    print(f"  Embedding范数: {torch.norm(emb1).item():.4f}, {torch.norm(emb2).item():.4f}")

# 保存
torch.save(protein_embeddings, output_path)
print(f"\n✓ 已保存 {len(protein_embeddings)} 个蛋白质的测试embeddings")
print(f"  路径: {output_path}")
print("\n注意: 这些是用于诊断的测试embeddings，不是真实的ESM2 embeddings")
print("=" * 80)
