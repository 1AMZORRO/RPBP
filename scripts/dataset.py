#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集和数据加载器
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split


class RNAProteinDataset(Dataset):
    """RNA-蛋白质结合数据集"""
    
    def __init__(
        self,
        rna_sequences: List[str],
        protein_names: List[str],
        labels: List[int],
        protein_embeddings: Dict[str, torch.Tensor]
    ):
        """
        Args:
            rna_sequences: RNA序列列表
            protein_names: 对应的蛋白质名称列表
            labels: 标签列表（0或1）
            protein_embeddings: 蛋白质embeddings字典
        """
        self.rna_sequences = rna_sequences
        self.protein_names = protein_names
        self.labels = labels
        self.protein_embeddings = protein_embeddings
        
        assert len(rna_sequences) == len(protein_names) == len(labels), \
            "RNA序列、蛋白质名称和标签的数量必须一致"
    
    def __len__(self):
        return len(self.rna_sequences)
    
    def __getitem__(self, idx):
        rna_seq = self.rna_sequences[idx]
        protein_name = self.protein_names[idx]
        label = self.labels[idx]
        
        # 获取蛋白质embedding
        protein_emb = self.protein_embeddings[protein_name]
        
        return {
            'rna_sequence': rna_seq,
            'protein_embedding': protein_emb,
            'protein_name': protein_name,
            'label': torch.tensor(label, dtype=torch.float32)
        }


def collate_fn(batch):
    """自定义批次整理函数"""
    rna_sequences = [item['rna_sequence'] for item in batch]
    protein_embeddings = torch.stack([item['protein_embedding'] for item in batch])
    protein_names = [item['protein_name'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    return {
        'rna_sequences': rna_sequences,
        'protein_embeddings': protein_embeddings,
        'protein_names': protein_names,
        'labels': labels
    }


def load_data(
    rna_fasta: str,
    labels_file: str,
    protein_embeddings: Dict[str, torch.Tensor]
) -> Tuple[List[str], List[str], List[int]]:
    """
    加载RNA序列、蛋白质名称和标签
    
    Args:
        rna_fasta: RNA序列文件
        labels_file: 标签文件
        protein_embeddings: 蛋白质embeddings字典
    
    Returns:
        (RNA序列列表, 蛋白质名称列表, 标签列表)
    """
    # 读取RNA序列和蛋白质名称
    rna_sequences = []
    protein_names = []
    
    with open(rna_fasta, 'r') as f:
        current_seq = []
        current_protein = None
        
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # 保存前一个序列
                if current_protein is not None:
                    rna_sequences.append(''.join(current_seq))
                    protein_names.append(current_protein)
                
                # 解析标题获取蛋白质名称
                # 格式: >12_AARS_K562_ENCSR825SVO_pos; chr21; class:1
                parts = line[1:].split(';')[0].strip().split('_')
                # 跳过第一个数字ID和最后的pos/neg
                protein_name = '_'.join(parts[1:-1])
                current_protein = protein_name
                current_seq = []
            else:
                current_seq.append(line)
        
        # 保存最后一个序列
        if current_protein is not None:
            rna_sequences.append(''.join(current_seq))
            protein_names.append(current_protein)
    
    # 读取标签
    labels = []
    with open(labels_file, 'r') as f:
        for line in f:
            labels.append(int(line.strip()))
    
    # 验证数据一致性
    assert len(rna_sequences) == len(protein_names) == len(labels), \
        f"数据数量不一致: RNA={len(rna_sequences)}, 蛋白质={len(protein_names)}, 标签={len(labels)}"
    
    # 验证所有蛋白质都有embedding
    unique_proteins = set(protein_names)
    missing_proteins = unique_proteins - set(protein_embeddings.keys())
    if missing_proteins:
        print(f"警告: 以下蛋白质缺少embedding: {missing_proteins}")
    
    return rna_sequences, protein_names, labels


def create_data_splits(
    rna_sequences: List[str],
    protein_names: List[str],
    labels: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[Tuple, Tuple, Tuple]:
    """
    划分训练集、验证集和测试集
    
    Args:
        rna_sequences: RNA序列列表
        protein_names: 蛋白质名称列表
        labels: 标签列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子
    
    Returns:
        ((train_rna, train_prot, train_labels),
         (val_rna, val_prot, val_labels),
         (test_rna, test_prot, test_labels))
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "比例之和必须等于1"
    
    # 首先分出训练集
    indices = np.arange(len(rna_sequences))
    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=random_seed,
        stratify=labels
    )
    
    # 从剩余数据中分出验证集和测试集
    val_size = val_ratio / (val_ratio + test_ratio)
    temp_labels = [labels[i] for i in temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_size,
        random_state=random_seed,
        stratify=temp_labels
    )
    
    # 提取数据
    def extract_data(indices):
        return (
            [rna_sequences[i] for i in indices],
            [protein_names[i] for i in indices],
            [labels[i] for i in indices]
        )
    
    train_data = extract_data(train_indices)
    val_data = extract_data(val_indices)
    test_data = extract_data(test_indices)
    
    print(f"数据集划分:")
    print(f"  训练集: {len(train_indices)} 样本 ({len(train_indices)/len(rna_sequences)*100:.1f}%)")
    print(f"  验证集: {len(val_indices)} 样本 ({len(val_indices)/len(rna_sequences)*100:.1f}%)")
    print(f"  测试集: {len(test_indices)} 样本 ({len(test_indices)/len(rna_sequences)*100:.1f}%)")
    
    return train_data, val_data, test_data


def save_test_data(
    test_rna: List[str],
    test_proteins: List[str],
    test_labels: List[int],
    test_dir: str
):
    """
    保存测试集数据
    
    Args:
        test_rna: 测试集RNA序列
        test_proteins: 测试集蛋白质名称
        test_labels: 测试集标签
        test_dir: 输出目录
    """
    # 保存RNA序列
    rna_file = os.path.join(test_dir, 'rna_sequences.fasta')
    with open(rna_file, 'w') as f:
        for i, (seq, prot, label) in enumerate(zip(test_rna, test_proteins, test_labels)):
            # 重建标题
            header = f">{i}_{prot}_test; class:{label}"
            f.write(f"{header}\n{seq}\n")
    
    # 保存标签
    labels_file = os.path.join(test_dir, 'labels.txt')
    with open(labels_file, 'w') as f:
        for label in test_labels:
            f.write(f"{label}\n")
    
    # 保存蛋白质序列（从原始文件复制）
    print(f"测试集数据已保存到: {test_dir}")
    print(f"  RNA序列: {rna_file}")
    print(f"  标签: {labels_file}")
