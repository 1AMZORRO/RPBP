#!/bin/bash
# -*- coding: utf-8 -*-
# RNA-蛋白质结合预测系统 - 完整运行示例

echo "============================================================"
echo "RNA-蛋白质结合预测系统 - 完整运行流程"
echo "============================================================"

# 步骤0: 检查GPU
echo ""
echo "步骤0: 检查GPU支持"
echo "------------------------------------------------------------"
python check_gpu.py
echo ""
read -p "按Enter继续..."

# 步骤1: 数据准备
echo ""
echo "步骤1: 数据准备"
echo "------------------------------------------------------------"
echo "从原始FASTA文件提取并排序RNA序列和标签..."
python scripts/prepare_data.py
echo ""
read -p "数据准备完成。按Enter继续..."

# 步骤2: 预计算蛋白质embeddings
echo ""
echo "步骤2: 预计算蛋白质embeddings"
echo "------------------------------------------------------------"
echo "使用ESM2模型计算168个蛋白质的embeddings..."
echo "注意: 此步骤需要较长时间（GPU: 5-15分钟，CPU: 30-60分钟）"
python scripts/precompute_protein_embeddings.py
echo ""
read -p "蛋白质embeddings计算完成。按Enter继续..."

# 步骤3: 训练模型
echo ""
echo "步骤3: 训练模型"
echo "------------------------------------------------------------"
echo "开始训练RNA-蛋白质结合预测模型..."
echo "注意: 完整训练需要较长时间（GPU: 2-4小时，CPU: 不建议）"
python scripts/train.py
echo ""
read -p "模型训练完成。按Enter继续..."

# 步骤4: 预测和可视化
echo ""
echo "步骤4: 预测和可视化"
echo "------------------------------------------------------------"
echo "使用训练好的模型进行预测..."
python scripts/predict.py \
    --config config/config.yaml \
    --model models/checkpoints/best_model.pth \
    --rna-fasta data/test/rna_sequences.fasta \
    --labels data/test/labels.txt \
    --output predictions.txt \
    --visualize \
    --num-visualize 5

echo ""
echo "============================================================"
echo "完整流程执行完毕！"
echo "============================================================"
echo ""
echo "生成的文件："
echo "  - models/checkpoints/best_model.pth (最佳模型)"
echo "  - models/checkpoints/training_history.json (训练历史)"
echo "  - output/training_history.png (训练曲线图)"
echo "  - predictions.txt (预测结果)"
echo "  - predictions_metrics.txt (评估指标)"
echo "  - output/attention_sample_*.png (注意力热力图)"
echo "  - output/binding_sites_sample_*.png (结合位点分析图)"
echo ""
