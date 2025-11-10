#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU/CPU设备检测和可视化脚本
"""

import torch
import sys

def print_separator(char="=", length=60):
    """打印分隔线"""
    print(char * length)

def visualize_device_info():
    """可视化设备信息"""
    print_separator()
    print("设备信息检测")
    print_separator()
    
    # PyTorch版本
    print(f"\n📦 PyTorch版本: {torch.__version__}")
    
    # CUDA信息
    print(f"\n🔍 CUDA检测:")
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA是否可用: {'✓ 是' if cuda_available else '✗ 否'}")
    
    if cuda_available:
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"  cuDNN启用: {torch.backends.cudnn.enabled}")
        
        # GPU信息
        print(f"\n🎮 GPU信息:")
        gpu_count = torch.cuda.device_count()
        print(f"  GPU数量: {gpu_count}")
        
        for i in range(gpu_count):
            print(f"\n  GPU {i}:")
            props = torch.cuda.get_device_properties(i)
            print(f"    名称: {torch.cuda.get_device_name(i)}")
            print(f"    计算能力: {props.major}.{props.minor}")
            print(f"    总内存: {props.total_memory / 1024**3:.2f} GB")
            print(f"    多处理器数量: {props.multi_processor_count}")
            
            # 当前内存使用
            if i == torch.cuda.current_device():
                allocated = torch.cuda.memory_allocated(i) / 1024**2
                reserved = torch.cuda.memory_reserved(i) / 1024**2
                print(f"    当前内存使用:")
                print(f"      已分配: {allocated:.2f} MB")
                print(f"      已保留: {reserved:.2f} MB")
        
        # 推荐配置
        print(f"\n💡 推荐设置:")
        print(f"  当前设备: GPU {torch.cuda.current_device()}")
        print(f"  建议批次大小: 根据GPU内存调整")
        
        # 测试GPU计算
        print(f"\n🧪 GPU计算测试:")
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print(f"  矩阵乘法测试: ✓ 成功")
            print(f"  计算设备: {z.device}")
        except Exception as e:
            print(f"  GPU计算测试失败: {e}")
        
    else:
        print(f"\n⚠️  警告: 未检测到GPU")
        print(f"  将使用CPU进行计算")
        print(f"  建议:")
        print(f"    - 如果您有GPU，请安装GPU版本的PyTorch")
        print(f"    - 安装命令: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
        # CPU信息
        print(f"\n💻 CPU信息:")
        print(f"  可用核心数: {torch.get_num_threads()}")
    
    # 最终建议
    print_separator()
    print("训练建议")
    print_separator()
    
    if cuda_available:
        print("✓ 检测到GPU，可以开始GPU训练")
        print("  运行命令:")
        print("    python scripts/precompute_protein_embeddings.py")
        print("    python scripts/train.py")
    else:
        print("⚠ 未检测到GPU，将使用CPU训练")
        print("  CPU训练速度较慢，建议：")
        print("    1. 使用GPU服务器或云服务")
        print("    2. 减小批次大小和数据量进行测试")
    
    print_separator()

if __name__ == '__main__':
    visualize_device_info()
