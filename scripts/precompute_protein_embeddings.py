#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蛋白质Embedding预计算脚本
使用ESM2模型为所有蛋白质序列生成embedding，并保存以供训练时使用
"""

import os
import torch
import esm
import yaml
from typing import Dict, List, Tuple
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def read_protein_sequences(fasta_file: str) -> Dict[str, str]:
    """
    读取蛋白质序列
    
    Args:
        fasta_file: FASTA文件路径
    
    Returns:
        字典 {蛋白质名称: 序列}
    """
    proteins = {}
    current_name = None
    current_seq = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # 保存前一个序列
                if current_name is not None:
                    proteins[current_name] = ''.join(current_seq)
                
                # 开始新序列
                current_name = line[1:]  # 去掉'>'
                current_seq = []
            else:
                current_seq.append(line)
        
        # 保存最后一个序列
        if current_name is not None:
            proteins[current_name] = ''.join(current_seq)
    
    return proteins


def compute_protein_embeddings(
    proteins: Dict[str, str],
    model_name: str,
    batch_size: int = 4,
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """
    使用ESM2模型计算蛋白质embeddings
    
    Args:
        proteins: 蛋白质序列字典
        model_name: ESM2模型名称
        batch_size: 批次大小
        device: 计算设备
    
    Returns:
        字典 {蛋白质名称: embedding tensor}
    """
    print(f"加载ESM2模型: {model_name}")
    
    # 加载ESM2模型
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()
    
    print(f"模型加载完成，开始计算embeddings...")
    
    embeddings = {}
    protein_list = list(proteins.items())
    
    # 分批处理
    with torch.no_grad():
        for i in tqdm(range(0, len(protein_list), batch_size), desc="计算蛋白质embeddings"):
            batch = protein_list[i:i + batch_size]
            
            # 准备批次数据
            # ESM期望的格式: [(name, sequence), ...]
            batch_data = [(name, seq) for name, seq in batch]
            
            # 转换为模型输入
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)
            
            # 前向传播
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            
            # 提取每个蛋白质的embedding
            # 使用最后一层的平均池化作为蛋白质表示
            representations = results["representations"][33]
            
            for idx, (name, seq) in enumerate(batch):
                # 去除特殊token（首尾的BOS和EOS），只保留序列部分
                seq_len = len(seq)
                seq_embedding = representations[idx, 1:seq_len+1, :]
                
                # 使用平均池化得到固定长度的蛋白质表示
                protein_embedding = seq_embedding.mean(dim=0)  # [embedding_dim]
                
                embeddings[name] = protein_embedding.cpu()
    
    print(f"完成! 共计算了 {len(embeddings)} 个蛋白质的embeddings")
    
    return embeddings


def save_embeddings(embeddings: Dict[str, torch.Tensor], output_file: str):
    """
    保存embeddings到文件
    
    Args:
        embeddings: 蛋白质embeddings字典
        output_file: 输出文件路径
    """
    print(f"保存embeddings到: {output_file}")
    torch.save(embeddings, output_file)
    print(f"保存完成!")
    
    # 打印一些统计信息
    first_name = list(embeddings.keys())[0]
    embedding_dim = embeddings[first_name].shape[0]
    print(f"\nEmbedding维度: {embedding_dim}")
    print(f"蛋白质数量: {len(embeddings)}")


def main():
    """主函数"""
    # 加载配置
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config', 'config.yaml')
    config = load_config(config_path)
    
    # 设置路径
    protein_fasta = os.path.join(base_dir, config['data']['protein_fasta'])
    output_file = os.path.join(base_dir, config['data']['protein_embeddings'])
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu'
    print(f"使用设备: {device}")
    
    print("=" * 60)
    print("蛋白质Embedding预计算")
    print("=" * 60)
    
    # 读取蛋白质序列
    print("\n步骤1: 读取蛋白质序列")
    proteins = read_protein_sequences(protein_fasta)
    print(f"读取到 {len(proteins)} 个蛋白质序列")
    
    # 计算embeddings
    print("\n步骤2: 计算蛋白质embeddings")
    embeddings = compute_protein_embeddings(
        proteins,
        model_name=config['esm']['model_name'],
        batch_size=config['esm']['batch_size'],
        device=device
    )
    
    # 保存embeddings
    print("\n步骤3: 保存embeddings")
    save_embeddings(embeddings, output_file)
    
    print("\n" + "=" * 60)
    print("蛋白质Embedding预计算完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
