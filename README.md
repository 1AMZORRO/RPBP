# RNA-蛋白质结合位点预测 (RPBP)

基于深度学习的RNA-蛋白质结合位点预测项目，参考iDeepG的工作，使用Cross-Attention机制预测RNA序列与蛋白质的结合概率。

## 项目简介

本项目实现了一个完整的RNA-蛋白质结合预测流程，包括：
- 数据预处理和排序
- 使用ESM2模型提取蛋白质特征
- 使用One-hot编码提取RNA特征
- Cross-Attention机制学习RNA-蛋白质交互
- 模型训练、验证和测试
- 预测和可视化

## 目录结构

```
RPBP/
├── config/
│   └── config.yaml           # 配置文件
├── data/
│   ├── raw/                  # 原始数据
│   │   ├── part1.fasta       # RNA序列数据（共8个文件）
│   │   ├── ...
│   │   ├── part8.fasta
│   │   └── prot_seqs.fasta   # 168个蛋白质序列
│   ├── train/                # 训练数据
│   │   ├── rna_sequences.fasta
│   │   ├── labels.txt
│   │   └── protein_embeddings.pt
│   └── test/                 # 测试数据
│       ├── rna_sequences.fasta
│       └── labels.txt
├── models/
│   └── checkpoints/          # 模型检查点
│       ├── best_model.pth
│       └── training_history.json
├── output/                   # 输出文件
│   ├── training_history.png
│   ├── attention_sample_*.png
│   └── binding_sites_sample_*.png
├── scripts/
│   ├── prepare_data.py       # 数据预处理
│   ├── precompute_protein_embeddings.py  # 蛋白质embedding预计算
│   ├── model.py              # 模型定义
│   ├── dataset.py            # 数据集定义
│   ├── train.py              # 训练脚本
│   └── predict.py            # 预测脚本
├── requirements.txt          # 依赖包
└── README.md                 # 本文件
```

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- PyTorch >= 2.0.0
- fair-esm >= 2.0.0 (ESM2蛋白质语言模型)
- BioPython >= 1.81
- scikit-learn >= 1.3.0
- matplotlib, seaborn (可视化)

## 使用流程

### 1. 数据准备

从原始FASTA文件提取RNA序列和标签，并按蛋白质种类排序：

```bash
python scripts/prepare_data.py
```

输出：
- `data/train/rna_sequences.fasta` - 1,245,616个RNA序列
- `data/train/labels.txt` - 对应的标签

### 2. 预计算蛋白质Embeddings

使用ESM2模型为168个蛋白质序列生成embeddings：

```bash
python scripts/precompute_protein_embeddings.py
```

输出：
- `data/train/protein_embeddings.pt` - 蛋白质embeddings字典

**注意**: 此步骤需要GPU，可能需要10-30分钟。

### 3. 训练模型

训练RNA-蛋白质结合预测模型：

```bash
python scripts/train.py
```

训练过程会：
- 自动划分训练集(70%)、验证集(15%)、测试集(15%)
- 使用Cross-Attention机制学习RNA-蛋白质交互
- 保存最佳模型到`models/checkpoints/best_model.pth`
- 生成训练历史到`models/checkpoints/training_history.json`
- 绘制训练曲线到`output/training_history.png`
- 保存测试集到`data/test/`

训练参数可在`config/config.yaml`中调整。

### 4. 预测

使用训练好的模型进行预测：

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

参数说明：
- `--config`: 配置文件路径
- `--model`: 训练好的模型路径
- `--rna-fasta`: 待预测的RNA序列文件
- `--labels`: 真实标签文件（用于评估）
- `--output`: 预测结果输出文件
- `--visualize`: 是否生成可视化
- `--num-visualize`: 可视化样本数量

输出文件：
- `predictions.txt` - 每个样本的预测概率
- `predictions_metrics.txt` - 评估指标（Accuracy, Precision, Recall, F1, AUC）
- `output/attention_sample_*.png` - 注意力热力图
- `output/binding_sites_sample_*.png` - 结合位点分析图

## 模型架构

### RNA编码器
- One-hot编码（A, C, G, T/U, N）
- 线性投影到128维embedding空间

### 蛋白质编码器
- 使用ESM2 (esm2_t33_650M_UR50D) 预训练模型
- 提取1280维蛋白质表示

### Cross-Attention层
- RNA序列作为Query
- 蛋白质表示作为Key/Value
- 8个注意力头
- 256维隐藏层

### 分类器
- 多层感知机：512 → 256 → 128 → 1
- ReLU激活 + Dropout + BatchNorm
- 二分类输出（结合/不结合）

## 配置说明

主要配置项在`config/config.yaml`：

```yaml
# 数据集划分
split:
  train: 0.7
  val: 0.15
  test: 0.15

# 训练参数
training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.001
  early_stopping_patience: 10

# 模型参数
model:
  attention:
    num_heads: 8
    hidden_dim: 256
```

## 可视化输出

### 1. 训练曲线
显示训练和验证的损失、准确率、F1分数、AUC变化

### 2. 注意力热力图
- X轴：RNA序列位置
- Y轴：蛋白质
- 颜色深度：注意力权重大小

### 3. 结合位点分析图
- 显示RNA序列每个位置的注意力权重
- 标注Top-K最可能的结合位点
- 包括具体核苷酸信息

## 数据格式

### RNA序列文件格式 (FASTA)
```
>12_AARS_K562_ENCSR825SVO_pos; chr21; class:1
CGCCGGGACCGGGGTCCGGTGCGGAGTGCCCTTCGTCCTGGGAAACGGGGCGCGGCCGGA...
```

### 标签文件格式
```
1
0
1
...
```

### 蛋白质序列文件格式 (FASTA)
```
>AARS_K562_ENCSR825SVO
MDSTLTASEIRQRFIDFFKRNEHTYVHSSATIPLDDPTLLFANAGMNQFKPIFLNTIDPS...
```

## 性能指标

模型在测试集上评估以下指标：
- **Accuracy**: 分类准确率
- **Precision**: 精确率
- **Recall**: 召回率
- **F1-score**: F1分数
- **AUC-ROC**: ROC曲线下面积

## 注意事项

1. **显存要求**: ESM2模型需要至少8GB GPU显存，建议使用16GB以上
2. **训练时间**: 完整训练可能需要数小时，取决于硬件配置
3. **数据文件**: 大型数据文件(fasta, embeddings)已在.gitignore中排除
4. **随机种子**: 使用固定随机种子(42)确保结果可复现

## 参考

本项目参考了以下工作：
- iDeepG: https://github.com/userscy/iDeepG
- ESM2: https://github.com/facebookresearch/esm

## 许可证

MIT License
