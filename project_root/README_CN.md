# IMDB 情感分类与联邦学习

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![联邦学习](https://img.shields.io/badge/%E8%81%94%E9%82%A6%E5%AD%A6%E4%B9%A0-FedAvg-green)
![自然语言处理](https://img.shields.io/badge/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86-Sentiment%20Analysis-orange)

一个完整的 NLP 项目，使用 **集中式训练** 和 **联邦学习** 两种方法实现 IMDB 电影评论情感分类。本项目展示了如何在保护数据隐私的前提下，通过联邦学习在分散的数据上进行机器学习模型训练。

## 📋 目录

- [项目概述](#项目概述)
- [项目特点](#项目特点)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [运行方法](#运行方法)
- [实验结果](#实验结果)
- [可视化](#可视化)
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
├── project_root/
│   ├── configs/
│   │   └── config.yaml               # 配置文件
│   ├── src/
│   │   ├── __init__.py
│   │   ├── data_loader.py            # 数据加载与划分
│   │   ├── preprocess.py             # 文本预处理与词汇表
│   │   ├── model.py                  # 模型定义
│   │   ├── train_centralized.py      # 集中式训练
│   │   ├── train_federated.py        # 联邦训练
│   │   ├── client.py                 # 联邦客户端逻辑
│   │   ├── server.py                 # 联邦服务器与聚合
│   │   ├── evaluate.py               # 评估与可视化
│   │   └── utils.py                  # 工具函数
│   ├── outputs/                      # 运行产物（Git 忽略）
│   │   ├── models/
│   │   ├── logs/
│   │   └── plots/
│   ├── data/                         # 数据缓存（Git 忽略）
│   ├── requirements.txt
│   ├── environment.yml
│   ├── .gitignore                    # 项目级忽略规则
│   ├── README.md
│   └── README_CN.md
├── outputs/                          # 仓库根目录历史产物（已忽略）
└── .gitignore                        # 仓库级忽略规则（用于 GH 同步）
```

GitHub 同步建议：
- 只提交源码、配置和文档。
- 不要提交 `outputs/`、`data/`、模型权重等生成文件。
- 统一在 `project_root` 目录执行脚本，避免产物分散到多个路径。

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Conda（推荐）或 pip
- CUDA 加速 GPU（可选，加快训练速度）

### 方法一：Conda 环境（推荐）

```bash
# 从配置文件创建环境
conda env create -f environment.yml

# 激活环境
conda activate fl_imdb
```

### 方法二：手动安装

```bash
# 创建新的 conda 环境
conda create -n fl_imdb python=3.10
conda activate fl_imdb

# 安装依赖
pip install -r requirements.txt
```

### 下载 NLTK 数据

首次运行项目时，NLTK 会自动下载所需资源：
- 停用词 (Stopwords)
- Punkt 分词器

## 📖 运行方法

### 1. 集中式训练

在完整的集中式数据集上训练模型：

```bash
cd project_root
python src/train_centralized.py
```

**执行内容：**
- 自动下载 IMDB 数据集
- 划分为训练/验证/测试集
- 训练 LSTM 模型进行情感分类
- 保存最佳模型到 `outputs/models/centralized.pt`
- 记录指标到 `outputs/logs/centralized_metrics.json`

### 2. 联邦学习训练

在 5 个客户端上使用联邦学习训练模型：

```bash
cd project_root
python src/train_federated.py
```

**执行内容：**
- 自动下载 IMDB 数据集
- 将数据分配给 5 个客户端，使用非 IID 分布
- 运行 10 轮 FedAvg 通信
- 每轮每个客户端本地训练 2 个 epoch
- 保存最佳全局模型到 `outputs/models/federated.pt`
- 记录指标到 `outputs/logs/federated_metrics.json`

### 3. 评估

评估两个模型并生成对比可视化：

```bash
cd project_root
python src/evaluate.py
```

**生成内容：**
- 两个模型的混淆矩阵
- 训练曲线
- 模型对比柱状图

## 📊 实验结果

### 预期性能

| 模型 | 准确率 | 精确率 | 召回率 | F1分数 |
|------|--------|--------|--------|--------|
| 集中式 | ~85-88% | ~85-88% | ~85-88% | ~85-88% |
| 联邦学习 | ~82-86% | ~82-86% | ~82-86% | ~82-86% |

*注：实际结果可能因随机种子和数据划分而有所不同*

### 联邦学习流程

联邦学习过程遵循以下步骤：

1. **初始化**：服务器向所有客户端广播全局模型
2. **本地训练**：每个客户端在本地数据上训练
3. **权重聚合**：服务器使用 FedAvg 收集并平均客户端权重
4. **重复**：多轮通信

```
第 1 轮:  客户端 0 [准确率: 0.72] → 客户端 1 [准确率: 0.68] → ... → 聚合
第 2 轮:  客户端 0 [准确率: 0.76] → 客户端 1 [准确率: 0.74] → ... → 聚合
...
第 10 轮: 客户端 0 [准确率: 0.85] → 客户端 1 [准确率: 0.82] → ... → 聚合
```

## 📈 可视化

评估脚本生成以下图表：

### 1. 混淆矩阵
显示预测在真实标签和预测标签上的分布。

### 2. 训练曲线
- **集中式**: 基于 Epoch 的损失、准确率和 F1 曲线
- **联邦学习**: 基于轮次的客户端指标进展

### 3. 模型对比
柱状图比较集中式和联邦学习模型的准确率、精确率、召回率和 F1 分数。

## 🔧 配置

编辑 `configs/config.yaml` 来自定义设置：

```yaml
# 数据配置
data:
  max_vocab_size: 20000    # 词汇表大小
  max_seq_length: 256       # 最大序列长度
  batch_size: 32            # 批次大小

# 模型配置
model:
  embedding_dim: 128         # 词嵌入维度
  hidden_dim: 256            # LSTM 隐藏层维度
  num_layers: 2              # LSTM 层数
  dropout: 0.5               # Dropout 比率

# 联邦学习配置
federated:
  num_clients: 5             # 客户端数量
  local_epochs: 2            # 本地训练轮次
  global_rounds: 10          # 通信轮次
  alpha: 0.5                 # 狄利克雷分布参数
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
      - 发送更新后的权重 wₖᵗ 到服务器
   c. 服务器聚合: wₜ = Σₖ (nₖ/n) × wₖᵗ
```

### 非独立同分布 (Non-IID) 数据分布

我们使用狄利克雷分布（alpha 参数）来模拟真实的非 IID 数据：

- 低 alpha (0.1): 高度非 IID，每个客户端的类别分布倾斜
- 高 alpha (1.0+): 更接近 IID，分布更均衡

## 🔮 未来改进

1. **差分隐私**: 为梯度添加噪声以增强隐私保护
2. **安全聚合**: 使用密码学技术进行安全的权重聚合
3. **模型压缩**: 实现量化和剪枝以提高 FL 效率
4. **更多模型**: 添加基于 Transformer 的模型（BERT、DistilBERT）
5. **通信效率**: 实现梯度压缩技术
6. **动态客户端选择**: 每轮只选择可用的/相关的客户端

## 📝 许可证

本项目是开源的，采用 MIT 许可证。

## 👨‍💻 作者

作为作品集项目展示：
- 深度学习（自然语言处理）
- 联邦学习
- PyTorch
- 机器学习中的数据隐私

---

如有问题或建议，请在 GitHub 上提交 issue。
