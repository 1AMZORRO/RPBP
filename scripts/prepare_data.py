#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据准备脚本
功能：
1. 从part1-8.fasta提取RNA序列和标签
2. 按照prot_seqs.fasta中的蛋白质顺序对RNA序列和标签进行排序
3. 生成data/train/rna_sequences.fasta和data/train/labels.txt
"""

import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple


def read_protein_order(prot_file: str) -> List[str]:
    """
    读取蛋白质序列文件，获取蛋白质名称的顺序
    
    Args:
        prot_file: 蛋白质序列文件路径
    
    Returns:
        蛋白质名称列表（按文件顺序）
    """
    protein_names = []
    with open(prot_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                protein_name = line.strip()[1:]  # 去掉'>'
                protein_names.append(protein_name)
    
    print(f"读取到 {len(protein_names)} 个蛋白质序列")
    return protein_names


def parse_rna_header(header: str) -> Tuple[str, int]:
    """
    解析RNA序列标题，提取蛋白质名称和标签
    
    Args:
        header: RNA序列标题行（去掉'>'）
    
    Returns:
        (蛋白质名称, 标签)
    """
    # 标题格式: >12_AARS_K562_ENCSR825SVO_pos; chr21; class:1
    parts = header.split(';')
    
    # 从第一部分提取蛋白质名称
    first_part = parts[0].strip()
    # 格式: 12_AARS_K562_ENCSR825SVO_pos
    # 需要提取: AARS_K562_ENCSR825SVO
    tokens = first_part.split('_')
    # 跳过第一个数字ID，取中间部分直到最后的pos/neg
    protein_name = '_'.join(tokens[1:-1])
    
    # 从最后一部分提取标签
    class_part = parts[-1].strip()
    # 格式: class:1 或 class:0
    label = int(class_part.split(':')[1])
    
    return protein_name, label


def read_rna_sequences(part_files: List[str]) -> Dict[str, List[Tuple[str, str, int]]]:
    """
    读取所有part文件中的RNA序列
    
    Args:
        part_files: part文件路径列表
    
    Returns:
        字典 {蛋白质名称: [(标题, 序列, 标签), ...]}
    """
    rna_data = defaultdict(list)
    total_sequences = 0
    
    for part_file in part_files:
        print(f"正在处理: {part_file}")
        with open(part_file, 'r') as f:
            header = None
            sequence_lines = []
            
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # 处理前一个序列
                    if header is not None:
                        sequence = ''.join(sequence_lines)
                        protein_name, label = parse_rna_header(header)
                        rna_data[protein_name].append((header, sequence, label))
                        total_sequences += 1
                    
                    # 开始新序列
                    header = line[1:]  # 去掉'>'
                    sequence_lines = []
                else:
                    sequence_lines.append(line)
            
            # 处理最后一个序列
            if header is not None:
                sequence = ''.join(sequence_lines)
                protein_name, label = parse_rna_header(header)
                rna_data[protein_name].append((header, sequence, label))
                total_sequences += 1
    
    print(f"总共读取 {total_sequences} 个RNA序列")
    print(f"涉及 {len(rna_data)} 个不同的蛋白质")
    
    return rna_data


def write_sorted_data(rna_data: Dict[str, List[Tuple[str, str, int]]], 
                     protein_order: List[str],
                     output_fasta: str,
                     output_labels: str):
    """
    按照蛋白质顺序写入排序后的RNA序列和标签
    
    Args:
        rna_data: RNA数据字典
        protein_order: 蛋白质名称顺序列表
        output_fasta: 输出的RNA序列文件
        output_labels: 输出的标签文件
    """
    total_written = 0
    
    with open(output_fasta, 'w') as f_seq, open(output_labels, 'w') as f_label:
        for protein_name in protein_order:
            if protein_name in rna_data:
                sequences = rna_data[protein_name]
                print(f"写入蛋白质 {protein_name} 的 {len(sequences)} 个RNA序列")
                
                for header, sequence, label in sequences:
                    # 写入序列
                    f_seq.write(f">{header}\n{sequence}\n")
                    # 写入标签
                    f_label.write(f"{label}\n")
                    total_written += 1
            else:
                print(f"警告: 蛋白质 {protein_name} 在RNA数据中未找到")
    
    print(f"\n总共写入 {total_written} 个序列和标签")


def main():
    """主函数"""
    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    train_dir = os.path.join(base_dir, 'data', 'train')
    
    # 蛋白质序列文件
    prot_file = os.path.join(raw_dir, 'prot_seqs.fasta')
    
    # part文件列表
    part_files = [os.path.join(raw_dir, f'part{i}.fasta') for i in range(1, 9)]
    
    # 输出文件
    output_fasta = os.path.join(train_dir, 'rna_sequences.fasta')
    output_labels = os.path.join(train_dir, 'labels.txt')
    
    print("=" * 60)
    print("开始数据准备流程")
    print("=" * 60)
    
    # 步骤1: 读取蛋白质顺序
    print("\n步骤1: 读取蛋白质序列顺序")
    protein_order = read_protein_order(prot_file)
    
    # 步骤2: 读取所有RNA序列
    print("\n步骤2: 读取RNA序列数据")
    rna_data = read_rna_sequences(part_files)
    
    # 步骤3: 按蛋白质顺序写入数据
    print("\n步骤3: 按蛋白质顺序写入排序后的数据")
    write_sorted_data(rna_data, protein_order, output_fasta, output_labels)
    
    print("\n" + "=" * 60)
    print("数据准备完成!")
    print(f"RNA序列文件: {output_fasta}")
    print(f"标签文件: {output_labels}")
    print("=" * 60)


if __name__ == '__main__':
    main()
