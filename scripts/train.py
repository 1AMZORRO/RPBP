#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练脚本
"""

import os
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from model import RNAProteinBindingModel
from dataset import (
    RNAProteinDataset, 
    collate_fn, 
    load_data, 
    create_data_splits,
    save_test_data
)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    max_grad_norm: float = 1.0
) -> Tuple[float, float]:
    """
    训练一个epoch
    
    Args:
        max_grad_norm: 梯度裁剪的最大范数
    
    Returns:
        (平均损失, 准确率)
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="训练中"):
        rna_sequences = batch['rna_sequences']
        protein_embeddings = batch['protein_embeddings'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        logits, _ = model(rna_sequences, protein_embeddings)
        logits = logits.squeeze()
        
        # 计算损失
        loss = criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        # 记录
        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float, dict]:
    """
    验证模型
    
    Returns:
        (平均损失, 准确率, 指标字典)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="验证中"):
            rna_sequences = batch['rna_sequences']
            protein_embeddings = batch['protein_embeddings'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            logits, _ = model(rna_sequences, protein_embeddings)
            logits = logits.squeeze()
            
            # 计算损失
            loss = criterion(logits, labels)
            
            # 记录
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    # 计算指标
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0
    }
    
    return avg_loss, metrics['accuracy'], metrics


def plot_training_history(history: dict, output_path: str):
    """绘制训练历史曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 损失曲线
    axes[0, 0].plot(history['train_loss'], label='训练损失')
    axes[0, 0].plot(history['val_loss'], label='验证损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('损失曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 准确率曲线
    axes[0, 1].plot(history['train_acc'], label='训练准确率')
    axes[0, 1].plot(history['val_acc'], label='验证准确率')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('准确率曲线')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1分数曲线
    axes[1, 0].plot(history['val_f1'], label='验证F1分数')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1分数曲线')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # AUC曲线
    axes[1, 1].plot(history['val_auc'], label='验证AUC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].set_title('AUC曲线')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"训练曲线已保存到: {output_path}")
    plt.close()


def main():
    """主函数"""
    # 加载配置
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config', 'config.yaml')
    config = load_config(config_path)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu'
    
    # 可视化设备信息
    print("\n" + "=" * 60)
    print("设备信息")
    print("=" * 60)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"使用设备: {device.upper()}")
    if device == 'cuda':
        print("✓ 正在使用GPU进行训练")
    else:
        print("⚠ 正在使用CPU进行训练（建议使用GPU以加快速度）")
    print("=" * 60)
    
    print("=" * 60)
    print("RNA-蛋白质结合预测模型训练")
    print("=" * 60)
    
    # 步骤1: 加载蛋白质embeddings
    print("\n步骤1: 加载蛋白质embeddings")
    protein_emb_path = os.path.join(base_dir, config['data']['protein_embeddings'])
    protein_embeddings = torch.load(protein_emb_path)
    print(f"加载了 {len(protein_embeddings)} 个蛋白质的embeddings")
    
    # 步骤2: 加载数据
    print("\n步骤2: 加载RNA序列和标签")
    rna_fasta = os.path.join(base_dir, config['data']['rna_fasta'])
    labels_file = os.path.join(base_dir, config['data']['labels'])
    rna_sequences, protein_names, labels = load_data(rna_fasta, labels_file, protein_embeddings)
    print(f"加载了 {len(rna_sequences)} 个RNA序列")
    
    # 步骤3: 划分数据集
    print("\n步骤3: 划分训练集、验证集和测试集")
    train_data, val_data, test_data = create_data_splits(
        rna_sequences,
        protein_names,
        labels,
        train_ratio=config['split']['train'],
        val_ratio=config['split']['val'],
        test_ratio=config['split']['test'],
        random_seed=config['split']['random_seed']
    )
    
    # 保存测试集
    test_dir = os.path.join(base_dir, 'data', 'test')
    save_test_data(test_data[0], test_data[1], test_data[2], test_dir)
    
    # 步骤4: 创建数据集和数据加载器
    print("\n步骤4: 创建数据加载器")
    train_dataset = RNAProteinDataset(*train_data, protein_embeddings)
    val_dataset = RNAProteinDataset(*val_data, protein_embeddings)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # 步骤5: 创建模型
    print("\n步骤5: 创建模型")
    model = RNAProteinBindingModel(
        rna_vocab_size=config['model']['rna_vocab_size'],
        rna_embedding_dim=config['model']['rna_embedding_dim'],
        protein_embedding_dim=config['model']['protein_embedding_dim'],
        num_attention_heads=config['model']['attention']['num_heads'],
        attention_hidden_dim=config['model']['attention']['hidden_dim'],
        attention_dropout=config['model']['attention']['dropout'],
        classifier_hidden_dims=config['model']['classifier']['hidden_dims'],
        classifier_dropout=config['model']['classifier']['dropout']
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    
    # 步骤6: 设置训练
    print("\n步骤6: 设置损失函数和优化器")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['training']['lr_scheduler']['factor'],
        patience=config['training']['lr_scheduler']['patience'],
        min_lr=config['training']['lr_scheduler']['min_lr']
    )
    
    # 步骤7: 训练循环
    print("\n步骤7: 开始训练")
    print("=" * 60)
    
    # 显示设备使用情况
    if device == 'cuda':
        print(f"模型已加载到GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print("=" * 60)
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_auc': []
    }
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # 显示GPU内存使用（如果使用GPU）
        if device == 'cuda':
            print(f"GPU内存使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB / "
                  f"{torch.cuda.memory_reserved(0) / 1024**2:.2f} MB (已分配/已保留)")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc, val_metrics = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        
        # 打印结果
        print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")
        print(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            # 保存模型
            model_dir = os.path.join(base_dir, config['output']['model_dir'])
            os.makedirs(model_dir, exist_ok=True)
            best_model_path = os.path.join(base_dir, config['output']['best_model'])
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': config
            }, best_model_path)
            
            print(f"✓ 保存最佳模型 (验证损失: {val_loss:.4f})")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f"\n早停: 验证损失在 {config['training']['early_stopping_patience']} 个epoch内未改善")
            break
    
    # 步骤8: 保存训练历史
    print("\n步骤8: 保存训练历史和绘制曲线")
    history_path = os.path.join(base_dir, config['output']['training_history'])
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"训练历史已保存到: {history_path}")
    
    # 绘制训练曲线
    plots_dir = os.path.join(base_dir, config['output']['plots_dir'])
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, 'training_history.png')
    plot_training_history(history, plot_path)
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳模型在第 {best_epoch} 个epoch")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存在: {best_model_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
