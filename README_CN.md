# IMDB 情感分类与联邦学习

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![联邦学习](https://img.shields.io/badge/%E8%81%94%E9%82%A6%E5%AD%A6%E4%B9%A0-FedAvg-green)
![自然语言处理](https://img.shields.io/badge/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86-Sentiment%20Analysis-orange)

一个完整的 NLP 项目，使用 **集中式训练** 和 **联邦学习** 两种方法实现 IMDB 电影评论情感分类。本项目展示了如何在保护数据隐私的前提下，通过联邦学习在分散的数据上进行机器学习模型训练。

**English Documentation**: [README.md](README.md)

## 📋 目录

- [项目概述](#项目概述)
- [项目特点](#项目特点)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [运行方法](#运行方法)
- [实验结果](#实验结果)
- [可视化](#可视化)
- [配置说明](#配置说明)
- [技术细节](#技术细节)
- [未来改进](#未来改进)

## 🎯 项目概述

本项目实现了一个端到端的 IMDB 电影评论情感分类系统：

1. **集中式训练**: 传统机器学习方法，将所有数据集中在一个地方训练
2. **联邦学习 (FedAvg)**: 隐私保护方法，模型在分布式客户端上训练，不共享原始数据

### 核心技术

- **PyTorch**: 深度学习框架
- **LSTM**: 双向 LSTM 用于序列建模
- **FedAvg**: 联邦平均算法
- **HuggingFace Datasets**: IMDB 数据集加载
- **NLTK**: 文本预处理

## ✨ 项目特点

### NLP 流水线
- 文本预处理（小写化、标点符号去除、停用词去除）
- 分词和词汇表构建
- 序列填充和编码
- 自定义 PyTorch Dataset 和 DataLoader

### 模型
- **基线模型**: Embedding + 线性分类器
- **LSTM 分类器**: 带 dropout 和全连接层的双向 LSTM

### 联邦学习
- 跨客户端的非独立同分布 (Non-IID) 数据分布（狄利克雷分配）
- FedAvg 算法实现
- 可配置的客户端数量和本地轮次
- 通信轮次跟踪

### 评估
- 综合指标（准确率、精确率、召回率、F1分数）
- 混淆矩阵可视化
- 训练曲线比较

## 📁 项目结构

```
IMDB/
├── README.md
├── README_CN.md
├── .gitignore
├── configs/
│   └── config.yaml               # 配置文件
├── src/
│   ├── models/                   # 神经网络模型架构
│   │   ├── __init__.py
│   │   └── sentiment_model.py   # LSTM 和基线模型
│   ├── data/                     # 数据加载与预处理
│   │   ├── __init__.py
│   │   ├── loader.py             # 数据集加载与划分
│   │   └── preprocess.py         # 文本预处理与词汇表
│   ├── training/                 # 训练脚本
│   │   ├── __init__.py
│   │   ├── centralized.py        # 集中式训练
│   │   └── federated.py          # 联邦学习训练
│   ├── federated/                # 联邦学习组件
│   │   ├── __init__.py
│   │   ├── server.py             # 联邦服务器与聚合
│   │   └── client.py             # 联邦客户端逻辑
│   ├── evaluation/               # 评估与可视化
│   │   ├── __init__.py
│   │   └── evaluate.py           # 指标计算与绘图
│   └── utils/                    # 工具函数
│       ├── __init__.py
│       └── utils.py              # 配置、指标、IO 辅助函数
├── docs/
│   └── images/                   # 文档图片
│       ├── centralized_confusion_matrix.png
│       ├── federated_confusion_matrix.png
│       ├── federated_training_curves.png
│       └── model_comparison.png
├── outputs/                      # 运行产物（Git 忽略）
│   ├── models/                   # 保存的模型权重
│   ├── logs/                     # 训练日志和指标
│   └── plots/                    # 生成的可视化图表
├── data/                         # 下载的数据集缓存（Git 忽略）
├── requirements.txt
└── environment.yml
```

**GitHub 同步建议:**
- 只提交源码、配置和文档
- 不要提交 `outputs/`、`data/` 或模型权重等生成文件
- 所有运行产物通过 `.gitignore` 自动忽略

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Conda（推荐）或 pip
- CUDA GPU（可选，用于加速训练）

### 方式 A: Conda 环境（推荐）

```bash
# 从文件创建环境
conda env create -f environment.yml

# 激活环境
conda activate fl_imdb
```

### 方式 B: 手动安装

```bash
# 创建新的 conda 环境
conda create -n fl_imdb python=3.10
conda activate fl_imdb

# 安装依赖
pip install -r requirements.txt
```

### 下载 NLTK 数据

首次运行项目时，NLTK 会自动下载所需资源：
- 停用词
- Punkt 分词器

## 📖 运行方法

### 1. 集中式训练

在完整的集中式数据集上训练模型：

```bash
python src/training/centralized.py
```

**功能:**
- 自动下载 IMDB 数据集
- 划分为训练/验证/测试集
- 训练 LSTM 情感分类模型
- 将最佳模型保存到 `outputs/models/centralized.pt`
- 将指标记录到 `outputs/logs/centralized_metrics.json`

### 2. 联邦学习训练

使用联邦学习在 5 个客户端上训练模型：

```bash
python src/training/federated.py
```

**功能:**
- 自动下载 IMDB 数据集
- 将数据以非独立同分布方式分配给 5 个客户端
- 运行 FedAvg 算法进行 10 轮通信
- 每轮每个客户端本地训练 2 个 epoch
- 将最佳全局模型保存到 `outputs/models/federated.pt`
- 将指标记录到 `outputs/logs/federated_metrics.json`

### 3. 评估

评估两个模型并生成对比可视化：

```bash
python src/evaluation/evaluate.py
```

**生成内容:**
- 两个模型的混淆矩阵
- 训练曲线
- 模型对比柱状图
- 所有图表保存到 `outputs/plots/`

## 📊 实验结果

### 结果预览

#### 混淆矩阵

| 集中式 | 联邦学习 |
|---|---|
| ![集中式混淆矩阵](docs/images/centralized_confusion_matrix.png) | ![联邦学习混淆矩阵](docs/images/federated_confusion_matrix.png) |

#### 训练曲线与对比

![联邦学习训练曲线](docs/images/federated_training_curves.png)

![模型对比](docs/images/model_comparison.png)

### 预期性能

| 模型 | 准确率 | 精确率 | 召回率 | F1 分数 |
|-------|----------|-----------|--------|----------|
| 集中式 | ~85-88% | ~85-88% | ~85-88% | ~85-88% |
| 联邦学习 | ~82-86% | ~82-86% | ~82-86% | ~82-86% |

*注意: 实际结果可能因随机种子和数据划分而有所不同。*

### 联邦学习过程

联邦学习过程包含以下步骤：

1. **初始化**: 服务器向所有客户端广播全局模型
2. **本地训练**: 每个客户端在其本地数据上训练
3. **权重聚合**: 服务器收集并使用 FedAvg 平均客户端权重
4. **重复**: 重复步骤 1-3 进行多轮通信

```
第 1 轮:  客户端 0 [准确率: 0.72] → 客户端 1 [准确率: 0.68] → ... → 聚合
第 2 轮:  客户端 0 [准确率: 0.76] → 客户端 1 [准确率: 0.74] → ... → 聚合
...
第 10 轮: 客户端 0 [准确率: 0.85] → 客户端 1 [准确率: 0.82] → ... → 聚合
```

## 📈 可视化

评估脚本生成以下图表：

### 1. 混淆矩阵
显示真实标签和预测标签的分布情况。

### 2. 训练曲线
- **集中式**: 基于 epoch 的损失、准确率和 F1 曲线
- **联邦学习**: 基于通信轮次的客户端指标进展

### 3. 模型对比
柱状图比较集中式和联邦学习模型的准确率、精确率、召回率和 F1 分数。

## 🔧 配置说明

编辑 `configs/config.yaml` 进行自定义：

```yaml
# 数据配置
data:
  max_vocab_size: 20000    # 词汇表大小
  max_seq_length: 256      # 最大序列长度
  batch_size: 32           # 批次大小

# 模型配置
model:
  embedding_dim: 128       # 词嵌入维度
  hidden_dim: 256          # LSTM 隐藏层维度
  num_layers: 2            # LSTM 层数
  dropout: 0.5             # Dropout 率

# 集中式训练配置
centralized:
  learning_rate: 0.001
  epochs: 10
  early_stopping_patience: 3

# 联邦学习配置
federated:
  num_clients: 5           # 客户端数量
  local_epochs: 2          # 本地训练轮次
  global_rounds: 10        # 通信轮次
  alpha: 0.5               # 狄利克雷分布参数
  learning_rate: 0.001
```

## 🔬 技术细节

### FedAvg 算法

联邦平均 (FedAvg) 算法：

```
1. 服务器初始化全局模型 w₀
2. 对于每轮 t = 1, 2, ...:
   a. 服务器向所有客户端广播 wₜ₋₁
   b. 对于每个客户端 k（并行）:
      - 接收全局权重
      - 本地训练 E 个 epoch
      - 将更新后的权重 wₖᵗ 发送给服务器
   c. 服务器聚合: wₜ = Σₖ (nₖ/n) × wₖᵗ
```

### 非独立同分布数据

我们使用狄利克雷分布（alpha 参数）模拟真实的非独立同分布数据：

- 低 alpha (0.1): 高度非独立同分布，每个客户端的类别分布倾斜
- 高 alpha (1.0+): 更接近独立同分布，分布较为均衡

### 模型架构

**LSTM 分类器:**
- 嵌入层 (vocab_size × embedding_dim)
- 双向 LSTM（2 层，hidden_dim=256）
- 带 dropout 的全连接层
- 二分类输出（正面/负面情感）

## 🔮 未来改进

1. **差分隐私**: 在梯度中添加噪声以增强隐私保护
2. **安全聚合**: 使用加密技术实现安全的权重聚合
3. **模型压缩**: 实现量化和剪枝以提高联邦学习效率
4. **额外模型**: 添加 Transformer 模型（BERT、DistilBERT）
5. **通信效率**: 实现梯度压缩技术
6. **动态客户端选择**: 每轮只选择可用/相关的客户端

## 📝 许可证

本项目是开源的，采用 MIT 许可证。

## 👨‍💻 作者

作为展示以下技术的作品集项目：
- 深度学习（NLP）
- 联邦学习
- PyTorch
- 机器学习中的数据隐私

---

如有问题或反馈，请在 GitHub 上提出 issue。
