# 快速入门指南

## 1. 环境准备

### 检查GPU
```bash
python check_gpu.py
```

### 安装依赖
```bash
# 如果有GPU（推荐）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 如果只有CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install -r requirements.txt
```

## 2. 数据准备

```bash
python scripts/prepare_data.py
```

输出：
- `data/train/rna_sequences.fasta` - 1,245,616个RNA序列
- `data/train/labels.txt` - 对应标签

## 3. 预计算蛋白质Embeddings

```bash
python scripts/precompute_protein_embeddings.py
```

⏱️ 预计时间：
- GPU: 5-15分钟
- CPU: 30-60分钟（不推荐）

输出：
- `data/train/protein_embeddings.pt`

## 4. 训练模型

```bash
python scripts/train.py
```

⏱️ 预计时间：
- GPU: 2-4小时
- CPU: 数十小时（不推荐）

训练过程中会显示：
```
============================================================
设备信息
============================================================
使用设备: CUDA
✓ 正在使用GPU进行训练
GPU名称: NVIDIA GeForce RTX 3090
GPU内存: 24.00 GB
============================================================

Epoch 1/50
GPU内存使用: 2345.67 MB / 3456.78 MB (已分配/已保留)
训练中: 100%|████████████| 27363/27363 [05:23<00:00, 84.65it/s]
验证中: 100%|████████████| 5865/5865 [00:45<00:00, 129.89it/s]

训练 - 损失: 0.3456, 准确率: 0.8523
验证 - 损失: 0.3821, 准确率: 0.8312, F1: 0.8256, AUC: 0.9012
✓ 保存最佳模型 (验证损失: 0.3821)
```

输出文件：
- `models/checkpoints/best_model.pth` - 最佳模型
- `models/checkpoints/training_history.json` - 训练历史
- `output/training_history.png` - 训练曲线图
- `data/test/rna_sequences.fasta` - 测试集RNA序列
- `data/test/labels.txt` - 测试集标签

## 5. 预测

```bash
python scripts/predict.py \
    --config config/config.yaml \
    --model models/checkpoints/best_model.pth \
    --rna-fasta data/test/rna_sequences.fasta \
    --labels data/test/labels.txt \
    --output predictions.txt \
    --visualize \
    --num-visualize 5
```

预测过程中会显示：
```
============================================================
设备信息
============================================================
使用设备: CUDA
✓ 正在使用GPU进行预测
============================================================

步骤5: 进行预测
模型已加载到GPU: NVIDIA GeForce RTX 3090
GPU内存使用: 1234.56 MB
```

输出文件：
- `predictions.txt` - 预测结果（包含每个样本的预测概率）
- `predictions_metrics.txt` - 评估指标
- `output/attention_sample_*.png` - 注意力热力图（5个样本）
- `output/binding_sites_sample_*.png` - 结合位点分析图（5个样本）

## 6. 查看结果

### 评估指标
```bash
cat predictions_metrics.txt
```

示例输出：
```
评估指标
========================================
Accuracy (准确率): 0.8523
Precision (精确率): 0.8612
Recall (召回率): 0.8435
F1-score (F1分数): 0.8523
AUC-ROC (ROC曲线下面积): 0.9123
```

### 预测结果
```bash
head predictions.txt
```

示例输出：
```
样本ID  真实标签  预测概率  预测标签
0       1        0.892341  1
1       0        0.123456  0
2       1        0.956789  1
...
```

### 可视化结果

查看生成的图片：
- `output/training_history.png` - 查看训练过程
- `output/attention_sample_0.png` - 查看注意力权重分布
- `output/binding_sites_sample_0.png` - 查看关键结合位点

## 7. 一键运行（可选）

如果想自动运行完整流程：

```bash
./run_full_pipeline.sh
```

## 常见问题

### Q: GPU内存不足怎么办？
A: 在`config/config.yaml`中减小`batch_size`：
```yaml
training:
  batch_size: 16  # 从32减小到16
```

### Q: 训练速度太慢？
A: 确保使用GPU：
```bash
python check_gpu.py  # 检查GPU是否可用
```

### Q: 如何使用自己的数据？
A: 
1. 准备FASTA格式的RNA序列文件
2. 准备对应的标签文件（每行一个标签：0或1）
3. 准备蛋白质序列FASTA文件
4. 修改`config/config.yaml`中的路径
5. 运行训练脚本

### Q: 如何调整模型参数？
A: 编辑`config/config.yaml`文件，可调整：
- 学习率 (`learning_rate`)
- 批次大小 (`batch_size`)
- 训练轮数 (`num_epochs`)
- 注意力头数 (`attention.num_heads`)
- 隐藏层维度 (`attention.hidden_dim`)

## 技术支持

如有问题，请查看：
- `README.md` - 完整文档
- `test_system.py` - 系统组件测试
- `test_training.py` - 小规模训练测试
