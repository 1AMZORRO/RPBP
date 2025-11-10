#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型预测脚本
"""

import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from model import RNAProteinBindingModel
from dataset import RNAProteinDataset, collate_fn, load_data


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_path: str, device: str) -> Tuple[RNAProteinBindingModel, dict]:
    """
    加载训练好的模型
    
    Returns:
        (模型, 配置)
    """
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"模型加载成功 (来自epoch {checkpoint['epoch']})")
    print(f"验证集最佳损失: {checkpoint['val_loss']:.4f}")
    
    return model, config


def predict(
    model: RNAProteinBindingModel,
    dataloader: DataLoader,
    device: str,
    return_attention: bool = False
) -> Tuple[np.ndarray, np.ndarray, List, List]:
    """
    使用模型进行预测
    
    Returns:
        (预测概率, 真实标签, RNA序列, 注意力权重列表)
    """
    model.eval()
    all_probs = []
    all_labels = []
    all_rna_seqs = []
    all_attentions = []
    
    with torch.no_grad():
        for batch in dataloader:
            rna_sequences = batch['rna_sequences']
            protein_embeddings = batch['protein_embeddings'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            logits, attention_weights = model(
                rna_sequences,
                protein_embeddings,
                return_attention=return_attention
            )
            
            probs = torch.sigmoid(logits.squeeze())
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_rna_seqs.extend(rna_sequences)
            
            if return_attention and attention_weights is not None:
                all_attentions.extend(attention_weights.cpu().numpy())
    
    return np.array(all_probs), np.array(all_labels), all_rna_seqs, all_attentions


def calculate_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """计算评估指标"""
    preds = (probs > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
        'auc': roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0
    }
    
    return metrics


def save_predictions(probs: np.ndarray, labels: np.ndarray, output_file: str):
    """保存预测结果"""
    with open(output_file, 'w') as f:
        f.write("样本ID\t真实标签\t预测概率\t预测标签\n")
        for i, (prob, label) in enumerate(zip(probs, labels)):
            pred = 1 if prob > 0.5 else 0
            f.write(f"{i}\t{int(label)}\t{prob:.6f}\t{pred}\n")
    
    print(f"预测结果已保存到: {output_file}")


def save_metrics(metrics: Dict[str, float], output_file: str):
    """保存评估指标"""
    with open(output_file, 'w') as f:
        f.write("评估指标\n")
        f.write("=" * 40 + "\n")
        f.write(f"Accuracy (准确率): {metrics['accuracy']:.4f}\n")
        f.write(f"Precision (精确率): {metrics['precision']:.4f}\n")
        f.write(f"Recall (召回率): {metrics['recall']:.4f}\n")
        f.write(f"F1-score (F1分数): {metrics['f1']:.4f}\n")
        f.write(f"AUC-ROC (ROC曲线下面积): {metrics['auc']:.4f}\n")
    
    print(f"评估指标已保存到: {output_file}")
    print("\n评估指标:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-score: {metrics['f1']:.4f}")
    print(f"  AUC-ROC: {metrics['auc']:.4f}")


def visualize_attention(
    attention_weights: np.ndarray,
    rna_sequence: str,
    sample_id: int,
    output_dir: str
):
    """
    可视化注意力热力图
    
    Args:
        attention_weights: 注意力权重 [num_heads, seq_len, 1]
        rna_sequence: RNA序列
        sample_id: 样本ID
        output_dir: 输出目录
    """
    # 平均所有注意力头
    avg_attention = attention_weights.mean(axis=0).squeeze()  # [seq_len]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 3))
    
    # 绘制热力图
    im = ax.imshow(avg_attention.reshape(1, -1), cmap='YlOrRd', aspect='auto')
    
    # 设置刻度
    seq_len = len(rna_sequence)
    tick_spacing = max(1, seq_len // 20)  # 最多显示20个刻度
    xticks = list(range(0, seq_len, tick_spacing))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{i}\n{rna_sequence[i]}" for i in xticks])
    ax.set_yticks([0])
    ax.set_yticklabels(['蛋白质'])
    
    ax.set_xlabel('RNA序列位置')
    ax.set_title(f'样本 {sample_id} 的注意力权重热力图')
    
    # 添加颜色条
    plt.colorbar(im, ax=ax, label='注意力权重')
    
    # 保存
    output_file = os.path.join(output_dir, f'attention_sample_{sample_id}.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"注意力热力图已保存: {output_file}")


def analyze_binding_sites(
    attention_weights: np.ndarray,
    rna_sequence: str,
    sample_id: int,
    output_dir: str,
    top_k: int = 10
):
    """
    分析并可视化结合位点
    
    Args:
        attention_weights: 注意力权重 [num_heads, seq_len, 1]
        rna_sequence: RNA序列
        sample_id: 样本ID
        output_dir: 输出目录
        top_k: 显示前K个结合位点
    """
    # 平均所有注意力头
    avg_attention = attention_weights.mean(axis=0).squeeze()  # [seq_len]
    
    # 找到Top-K位点
    top_indices = np.argsort(avg_attention)[-top_k:][::-1]
    top_scores = avg_attention[top_indices]
    top_nucleotides = [rna_sequence[i] for i in top_indices]
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 子图1: 沿序列的注意力权重
    ax1.plot(avg_attention, linewidth=2)
    ax1.scatter(top_indices, top_scores, color='red', s=100, zorder=5, label=f'Top-{top_k} 结合位点')
    ax1.set_xlabel('RNA序列位置')
    ax1.set_ylabel('注意力权重')
    ax1.set_title(f'样本 {sample_id} 的RNA序列注意力权重分布')
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
    
    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars, top_scores)):
        ax2.text(score, i, f' {score:.4f}', va='center')
    
    plt.tight_layout()
    
    # 保存
    output_file = os.path.join(output_dir, f'binding_sites_sample_{sample_id}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"结合位点分析图已保存: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RNA-蛋白质结合预测')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--rna-fasta', type=str, required=True, help='RNA序列文件')
    parser.add_argument('--protein-fasta', type=str, help='蛋白质序列文件（可选，使用训练时的蛋白质embeddings）')
    parser.add_argument('--labels', type=str, required=True, help='标签文件')
    parser.add_argument('--output', type=str, default='predictions.txt', help='预测结果输出文件')
    parser.add_argument('--visualize', action='store_true', help='是否生成可视化')
    parser.add_argument('--num-visualize', type=int, default=5, help='可视化样本数量')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
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
        print("✓ 正在使用GPU进行预测")
    else:
        print("⚠ 正在使用CPU进行预测")
    print("=" * 60)
    
    print("=" * 60)
    print("RNA-蛋白质结合预测")
    print("=" * 60)
    
    # 步骤1: 加载模型
    print("\n步骤1: 加载模型")
    model, model_config = load_model(args.model, device)
    
    # 步骤2: 加载蛋白质embeddings
    print("\n步骤2: 加载蛋白质embeddings")
    protein_emb_path = os.path.join(base_dir, config['data']['protein_embeddings'])
    protein_embeddings = torch.load(protein_emb_path, map_location=device)
    print(f"加载了 {len(protein_embeddings)} 个蛋白质的embeddings")
    
    # 步骤3: 加载测试数据
    print("\n步骤3: 加载测试数据")
    rna_sequences, protein_names, labels = load_data(
        args.rna_fasta,
        args.labels,
        protein_embeddings
    )
    print(f"加载了 {len(rna_sequences)} 个测试样本")
    
    # 步骤4: 创建数据加载器
    print("\n步骤4: 创建数据加载器")
    test_dataset = RNAProteinDataset(rna_sequences, protein_names, labels, protein_embeddings)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['prediction']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # 步骤5: 进行预测
    print("\n步骤5: 进行预测")
    
    # 显示GPU内存使用（如果使用GPU）
    if device == 'cuda':
        print(f"模型已加载到GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    probs, true_labels, rna_seqs, attentions = predict(
        model,
        test_loader,
        device,
        return_attention=args.visualize
    )
    
    # 步骤6: 计算指标
    print("\n步骤6: 计算评估指标")
    metrics = calculate_metrics(probs, true_labels)
    
    # 步骤7: 保存结果
    print("\n步骤7: 保存结果")
    save_predictions(probs, true_labels, args.output)
    
    metrics_file = args.output.replace('.txt', '_metrics.txt')
    if metrics_file == args.output:
        metrics_file = 'output/metrics.txt'
    save_metrics(metrics, metrics_file)
    
    # 步骤8: 可视化（如果需要）
    if args.visualize and len(attentions) > 0:
        print("\n步骤8: 生成可视化")
        output_dir = os.path.join(base_dir, config['output']['plots_dir'])
        os.makedirs(output_dir, exist_ok=True)
        
        num_samples = min(args.num_visualize, len(rna_seqs))
        print(f"为前 {num_samples} 个样本生成可视化...")
        
        for i in range(num_samples):
            print(f"\n处理样本 {i}...")
            
            # 注意力热力图
            visualize_attention(
                attentions[i],
                rna_seqs[i],
                i,
                output_dir
            )
            
            # 结合位点分析
            analyze_binding_sites(
                attentions[i],
                rna_seqs[i],
                i,
                output_dir,
                top_k=config['prediction']['top_k_binding_sites']
            )
    
    print("\n" + "=" * 60)
    print("预测完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
