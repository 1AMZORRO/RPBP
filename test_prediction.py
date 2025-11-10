#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测功能测试 - 验证预测流程是否正常工作
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
from model import RNAProteinBindingModel
from dataset import RNAProteinDataset, collate_fn
from torch.utils.data import DataLoader

print("=" * 60)
print("预测功能测试")
print("=" * 60)

# 设置
device = 'cpu'
num_test_samples = 50  # 使用50个测试样本
print(f"\n配置:")
print(f"  设备: {device}")
print(f"  测试样本数: {num_test_samples}")

# 步骤1: 准备测试数据
print("\n步骤1: 准备测试数据")
base_dir = os.path.dirname(os.path.abspath(__file__))

# 读取RNA序列
rna_sequences = []
protein_names = []
with open(os.path.join(base_dir, 'data/train/rna_sequences.fasta'), 'r') as f:
    seq_count = 0
    current_seq = []
    current_prot = None
    
    for line in f:
        if seq_count >= num_test_samples:
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
    if current_prot is not None and seq_count < num_test_samples:
        rna_sequences.append(''.join(current_seq))
        protein_names.append(current_prot)

# 读取标签
labels = []
with open(os.path.join(base_dir, 'data/train/labels.txt'), 'r') as f:
    for i, line in enumerate(f):
        if i >= num_test_samples:
            break
        labels.append(int(line.strip()))

print(f"  加载了 {len(rna_sequences)} 个测试样本")
print(f"  正样本: {sum(labels)}, 负样本: {len(labels) - sum(labels)}")

# 步骤2: 创建模拟蛋白质embeddings
print("\n步骤2: 创建模拟蛋白质embeddings")
unique_proteins = list(set(protein_names))
protein_embeddings = {prot: torch.randn(1280) for prot in unique_proteins}
print(f"  创建了 {len(protein_embeddings)} 个蛋白质的embeddings")

# 步骤3: 创建并训练一个简单模型（用于测试）
print("\n步骤3: 创建并快速训练一个测试模型")
model = RNAProteinBindingModel(
    rna_vocab_size=5,
    rna_embedding_dim=128,
    protein_embedding_dim=1280,
    num_attention_heads=8,
    attention_hidden_dim=256,
    classifier_hidden_dims=[512, 256, 128]
)
model = model.to(device)

# 快速训练几个epoch（仅用于测试）
dataset = RNAProteinDataset(rna_sequences, protein_names, labels, protein_embeddings)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("  快速训练模型（5个epoch）...")
for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in dataloader:
        rna_seqs = batch['rna_sequences']
        prot_emb = batch['protein_embeddings'].to(device)
        batch_labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        logits, _ = model(rna_seqs, prot_emb)
        loss = criterion(logits.squeeze(), batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"    Epoch {epoch+1}/5, 损失: {avg_loss:.4f}")

print("  ✓ 模型训练完成")

# 步骤4: 保存模型
print("\n步骤4: 保存测试模型")
os.makedirs('models/test_checkpoints', exist_ok=True)
test_model_path = 'models/test_checkpoints/test_model.pth'

torch.save({
    'epoch': 5,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_loss': 0.5,
    'val_metrics': {
        'accuracy': 0.8,
        'precision': 0.75,
        'recall': 0.82,
        'f1': 0.78,
        'auc': 0.85
    },
    'config': {
        'model': {
            'rna_vocab_size': 5,
            'rna_embedding_dim': 128,
            'protein_embedding_dim': 1280,
            'attention': {
                'num_heads': 8,
                'hidden_dim': 256,
                'dropout': 0.1
            },
            'classifier': {
                'hidden_dims': [512, 256, 128],
                'dropout': 0.3
            }
        }
    }
}, test_model_path)
print(f"  ✓ 模型已保存到: {test_model_path}")

# 步骤5: 加载模型并进行预测
print("\n步骤5: 加载模型并进行预测")
checkpoint = torch.load(test_model_path, map_location=device)
model_config = checkpoint['config']

pred_model = RNAProteinBindingModel(
    rna_vocab_size=model_config['model']['rna_vocab_size'],
    rna_embedding_dim=model_config['model']['rna_embedding_dim'],
    protein_embedding_dim=model_config['model']['protein_embedding_dim'],
    num_attention_heads=model_config['model']['attention']['num_heads'],
    attention_hidden_dim=model_config['model']['attention']['hidden_dim'],
    classifier_hidden_dims=model_config['model']['classifier']['hidden_dims']
)
pred_model.load_state_dict(checkpoint['model_state_dict'])
pred_model = pred_model.to(device)
pred_model.eval()

print("  ✓ 模型加载成功")
print(f"  模型来自epoch: {checkpoint['epoch']}")
print(f"  验证损失: {checkpoint['val_loss']:.4f}")

# 步骤6: 进行预测
print("\n步骤6: 进行预测")
test_loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

all_probs = []
all_labels = []
all_attentions = []

with torch.no_grad():
    for batch in test_loader:
        rna_seqs = batch['rna_sequences']
        prot_emb = batch['protein_embeddings'].to(device)
        batch_labels = batch['labels'].to(device)
        
        logits, attention = pred_model(rna_seqs, prot_emb, return_attention=True)
        probs = torch.sigmoid(logits.squeeze())
        
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())
        all_attentions.extend(attention.cpu().numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

print(f"  ✓ 预测完成")
print(f"  预测样本数: {len(all_probs)}")

# 步骤7: 计算评估指标
print("\n步骤7: 计算评估指标")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

preds = (all_probs > 0.5).astype(int)

metrics = {
    'accuracy': accuracy_score(all_labels, preds),
    'precision': precision_score(all_labels, preds, zero_division=0),
    'recall': recall_score(all_labels, preds, zero_division=0),
    'f1': f1_score(all_labels, preds, zero_division=0),
    'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0
}

print(f"  Accuracy: {metrics['accuracy']:.4f}")
print(f"  Precision: {metrics['precision']:.4f}")
print(f"  Recall: {metrics['recall']:.4f}")
print(f"  F1-score: {metrics['f1']:.4f}")
print(f"  AUC-ROC: {metrics['auc']:.4f}")

# 步骤8: 保存预测结果
print("\n步骤8: 保存预测结果")
os.makedirs('output/test', exist_ok=True)
pred_file = 'output/test/test_predictions.txt'

with open(pred_file, 'w') as f:
    f.write("样本ID\t真实标签\t预测概率\t预测标签\n")
    for i, (prob, label) in enumerate(zip(all_probs, all_labels)):
        pred_label = 1 if prob > 0.5 else 0
        f.write(f"{i}\t{int(label)}\t{prob:.6f}\t{pred_label}\n")

print(f"  ✓ 预测结果已保存到: {pred_file}")

# 步骤9: 生成注意力热力图
print("\n步骤9: 生成可视化（注意力热力图）")
sample_idx = 0
attention_weights = all_attentions[sample_idx]  # [num_heads, seq_len, 1]
rna_seq = rna_sequences[sample_idx]

# 平均所有注意力头
avg_attention = attention_weights.mean(axis=0).squeeze()  # [seq_len]

fig, ax = plt.subplots(figsize=(15, 3))
im = ax.imshow(avg_attention.reshape(1, -1), cmap='YlOrRd', aspect='auto')

# 设置刻度
seq_len = len(rna_seq)
tick_spacing = max(1, seq_len // 20)
xticks = list(range(0, seq_len, tick_spacing))
ax.set_xticks(xticks)
ax.set_xticklabels([f"{i}\n{rna_seq[i]}" for i in xticks])
ax.set_yticks([0])
ax.set_yticklabels(['蛋白质'])

ax.set_xlabel('RNA序列位置')
ax.set_title(f'样本 {sample_idx} 的注意力权重热力图')
plt.colorbar(im, ax=ax, label='注意力权重')

attention_file = 'output/test/test_attention_sample_0.png'
plt.tight_layout()
plt.savefig(attention_file, dpi=150, bbox_inches='tight')
plt.close()

print(f"  ✓ 注意力热力图已保存到: {attention_file}")

# 步骤10: 生成结合位点分析图
print("\n步骤10: 生成可视化（结合位点分析）")
top_k = 10
top_indices = np.argsort(avg_attention)[-top_k:][::-1]
top_scores = avg_attention[top_indices]
top_nucleotides = [rna_seq[i] for i in top_indices]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 子图1: 沿序列的注意力权重
ax1.plot(avg_attention, linewidth=2)
ax1.scatter(top_indices, top_scores, color='red', s=100, zorder=5, label=f'Top-{top_k} 结合位点')
ax1.set_xlabel('RNA序列位置')
ax1.set_ylabel('注意力权重')
ax1.set_title(f'样本 {sample_idx} 的RNA序列注意力权重分布')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2: Top-K结合位点
colors = plt.cm.Reds(np.linspace(0.4, 0.9, top_k))
bars = ax2.barh(range(top_k), top_scores, color=colors)
ax2.set_yticks(range(top_k))
ax2.set_yticklabels([f"位置 {idx} ({nt})" for idx, nt in zip(top_indices, top_nucleotides)])
ax2.set_xlabel('注意力权重')
ax2.set_title(f'Top-{top_k} 最可能的结合位点')
ax2.invert_yaxis()

for i, (bar, score) in enumerate(zip(bars, top_scores)):
    ax2.text(score, i, f' {score:.4f}', va='center')

binding_file = 'output/test/test_binding_sites_sample_0.png'
plt.tight_layout()
plt.savefig(binding_file, dpi=150, bbox_inches='tight')
plt.close()

print(f"  ✓ 结合位点分析图已保存到: {binding_file}")

# 总结
print("\n" + "=" * 60)
print("预测功能测试完成！✓")
print("=" * 60)
print("\n生成的文件:")
print(f"  - {test_model_path} (测试模型)")
print(f"  - {pred_file} (预测结果)")
print(f"  - {attention_file} (注意力热力图)")
print(f"  - {binding_file} (结合位点分析图)")
print("\n预测功能验证成功，所有组件工作正常！")
print("=" * 60)
