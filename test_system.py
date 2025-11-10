#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本 - 验证模型组件是否正常工作
"""

import os
import sys
import torch
import numpy as np

print("=" * 60)
print("测试 RNA-蛋白质结合预测系统")
print("=" * 60)

# 测试1: 导入模型模块
print("\n测试1: 导入模型模块")
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
    from model import RNAEncoder, CrossAttentionLayer, RNAProteinBindingModel
    print("✓ 模型模块导入成功")
except Exception as e:
    print(f"✗ 模型模块导入失败: {e}")
    sys.exit(1)

# 测试2: RNA编码器
print("\n测试2: 测试RNA编码器")
try:
    rna_encoder = RNAEncoder(vocab_size=5, embedding_dim=128)
    test_sequences = ["ACGTACGT", "UUUUAAAA", "GCGCGCGC"]
    encoded = rna_encoder(test_sequences)
    print(f"  输入: {len(test_sequences)} 个序列")
    print(f"  输出形状: {encoded.shape}")
    assert encoded.shape == (3, 8, 128), "RNA编码器输出形状不正确"
    print("✓ RNA编码器测试通过")
except Exception as e:
    print(f"✗ RNA编码器测试失败: {e}")
    sys.exit(1)

# 测试3: Cross-Attention层
print("\n测试3: 测试Cross-Attention层")
try:
    attention = CrossAttentionLayer(
        query_dim=128,
        key_value_dim=1280,
        num_heads=8,
        hidden_dim=256
    )
    
    # 模拟输入
    query = torch.randn(2, 101, 128)  # [batch, seq_len, query_dim]
    key_value = torch.randn(2, 1280)  # [batch, protein_dim]
    
    output, attn_weights = attention(query, key_value, return_attention=True)
    print(f"  Query形状: {query.shape}")
    print(f"  Key/Value形状: {key_value.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  注意力权重形状: {attn_weights.shape}")
    
    assert output.shape == (2, 101, 256), "Cross-Attention输出形状不正确"
    assert attn_weights.shape == (2, 8, 101, 1), "注意力权重形状不正确"
    print("✓ Cross-Attention层测试通过")
except Exception as e:
    print(f"✗ Cross-Attention层测试失败: {e}")
    sys.exit(1)

# 测试4: 完整模型
print("\n测试4: 测试完整模型")
try:
    model = RNAProteinBindingModel(
        rna_vocab_size=5,
        rna_embedding_dim=128,
        protein_embedding_dim=1280,
        num_attention_heads=8,
        attention_hidden_dim=256,
        classifier_hidden_dims=[512, 256, 128]
    )
    
    # 模拟输入
    rna_sequences = ["A" * 101, "C" * 101, "G" * 101]
    protein_embeddings = torch.randn(3, 1280)
    
    logits, attention = model(rna_sequences, protein_embeddings, return_attention=True)
    
    print(f"  RNA序列数: {len(rna_sequences)}")
    print(f"  蛋白质embeddings形状: {protein_embeddings.shape}")
    print(f"  输出logits形状: {logits.shape}")
    print(f"  注意力权重形状: {attention.shape}")
    
    assert logits.shape == (3, 1), "模型输出形状不正确"
    print("✓ 完整模型测试通过")
except Exception as e:
    print(f"✗ 完整模型测试失败: {e}")
    sys.exit(1)

# 测试5: 数据集模块
print("\n测试5: 测试数据集模块")
try:
    from dataset import RNAProteinDataset, collate_fn
    from torch.utils.data import DataLoader
    
    # 创建模拟数据
    rna_seqs = ["ACGT" * 25 + "A", "UGCA" * 25 + "U"]
    prot_names = ["protein1", "protein2"]
    labels = [1, 0]
    prot_embeddings = {
        "protein1": torch.randn(1280),
        "protein2": torch.randn(1280)
    }
    
    dataset = RNAProteinDataset(rna_seqs, prot_names, labels, prot_embeddings)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    batch = next(iter(dataloader))
    print(f"  数据集大小: {len(dataset)}")
    print(f"  批次RNA序列数: {len(batch['rna_sequences'])}")
    print(f"  批次蛋白质embeddings形状: {batch['protein_embeddings'].shape}")
    print(f"  批次标签形状: {batch['labels'].shape}")
    
    assert len(batch['rna_sequences']) == 2
    assert batch['protein_embeddings'].shape == (2, 1280)
    assert batch['labels'].shape == (2,)
    print("✓ 数据集模块测试通过")
except Exception as e:
    print(f"✗ 数据集模块测试失败: {e}")
    sys.exit(1)

# 测试6: 前向传播和反向传播
print("\n测试6: 测试训练流程（前向+反向传播）")
try:
    import torch.nn as nn
    import torch.optim as optim
    
    model = RNAProteinBindingModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 创建小批次数据
    rna_seqs = ["ACGT" * 25 + "A"] * 4
    prot_emb = torch.randn(4, 1280)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    
    # 前向传播
    optimizer.zero_grad()
    logits, _ = model(rna_seqs, prot_emb)
    loss = criterion(logits.squeeze(), labels)
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    print(f"  训练损失: {loss.item():.4f}")
    print(f"  参数已更新")
    print("✓ 训练流程测试通过")
except Exception as e:
    print(f"✗ 训练流程测试失败: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("所有测试通过！✓")
print("系统组件工作正常，可以开始训练")
print("=" * 60)
