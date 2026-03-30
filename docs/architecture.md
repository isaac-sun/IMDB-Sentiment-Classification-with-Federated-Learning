# Architecture Documentation

## System Overview

This project implements a complete federated learning system for sentiment classification on IMDB movie reviews. The system supports both centralized and federated training paradigms.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        IMDB FL System                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────┐        ┌──────────────┐                      │
│  │  Data Layer   │───────▶│ Model Layer  │                      │
│  └───────────────┘        └──────────────┘                      │
│         │                         │                              │
│         │                         │                              │
│         ▼                         ▼                              │
│  ┌───────────────────────────────────────┐                      │
│  │       Training Layer                  │                      │
│  │  ┌──────────────┐  ┌───────────────┐ │                      │
│  │  │ Centralized  │  │  Federated    │ │                      │
│  │  └──────────────┘  └───────────────┘ │                      │
│  └───────────────────────────────────────┘                      │
│                         │                                        │
│                         ▼                                        │
│                 ┌──────────────┐                                │
│                 │ Evaluation   │                                │
│                 └──────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

### 1. Data Module (`src/data/`)

**Purpose**: Handle data loading, preprocessing, and distribution

**Components:**
- `loader.py`: Download and split IMDB dataset
  - Downloads from HuggingFace
  - Splits into train/val/test
  - Creates client datasets with non-IID distribution
  
- `preprocess.py`: Text preprocessing pipeline
  - Tokenization (NLTK)
  - Stopword removal
  - Vocabulary building
  - Sequence encoding and padding

### 2. Models Module (`src/models/`)

**Purpose**: Neural network architectures

**Components:**
- `sentiment_model.py`:
  - `BaselineModel`: Simple embedding + linear classifier
  - `LSTMClassifier`: Bidirectional LSTM with FC layers
  - Factory function for model creation

**LSTM Architecture:**
```
Input Text
    │
    ▼
Embedding Layer (vocab_size × embedding_dim)
    │
    ▼
Bidirectional LSTM (2 layers, hidden_dim=256)
    │
    ▼
Concatenate Forward/Backward Hidden States
    │
    ▼
Fully Connected (hidden_dim)
    │
    ▼
ReLU + Dropout
    │
    ▼
Fully Connected (1)
    │
    ▼
Output Logit (BCE Loss)
```

### 3. Training Module (`src/training/`)

**Purpose**: Training scripts for both paradigms

**Components:**
- `centralized.py`: Standard centralized training
  - Single model trained on all data
  - Early stopping based on validation F1
  - Saves best model checkpoint
  
- `federated.py`: Federated learning orchestration
  - Creates clients and server
  - Runs communication rounds
  - Coordinates FedAvg algorithm

### 4. Federated Module (`src/federated/`)

**Purpose**: Federated learning components

**Components:**
- `server.py`: FL server
  - Maintains global model
  - Broadcasts weights to clients
  - Aggregates client updates
  
- `client.py`: FL client
  - Receives global weights
  - Trains on local data
  - Returns updated weights

**FedAvg Algorithm Flow:**
```
Server                           Clients
  │                               │
  │──── Broadcast w_global ─────▶│
  │                               │
  │                          Local Training
  │                          (E epochs)
  │                               │
  │◀───── Send w_local ──────────│
  │                               │
Aggregate                         │
w_global = Σ(n_k/n) × w_k         │
  │                               │
```

### 5. Evaluation Module (`src/evaluation/`)

**Purpose**: Model evaluation and visualization

**Components:**
- `evaluate.py`:
  - Loads trained models
  - Calculates metrics (accuracy, precision, recall, F1)
  - Generates confusion matrices
  - Creates training curves
  - Produces comparison charts

### 6. Utils Module (`src/utils/`)

**Purpose**: Shared utilities

**Components:**
- `utils.py`:
  - Configuration loading (YAML)
  - Random seed setting
  - Model save/load
  - Metrics calculation
  - Output directory management
  - Average meter for tracking

## Data Flow

### Centralized Training Flow

```
IMDB Dataset
    │
    ▼
Download & Split
    │
    ├──▶ Train Set (22,500)
    ├──▶ Val Set (2,500)
    └──▶ Test Set (25,000)
    │
    ▼
Preprocessing
(tokenize, encode, pad)
    │
    ▼
DataLoader (batch_size=32)
    │
    ▼
LSTM Model
    │
    ▼
Training Loop
(10 epochs, early stopping)
    │
    ▼
Best Model Saved
    │
    ▼
Evaluation on Test Set
```

### Federated Learning Flow

```
IMDB Dataset
    │
    ▼
Download & Split
    │
    ├──▶ Train Set (25,000)
    └──▶ Test Set (25,000)
    │
    ▼
Non-IID Distribution
(Dirichlet α=0.5)
    │
    ├──▶ Client 0 (skewed toward negative)
    ├──▶ Client 1 (skewed toward positive)
    ├──▶ Client 2 (balanced)
    ├──▶ Client 3 (skewed toward negative)
    └──▶ Client 4 (balanced)
    │
    ▼
For each round (10 rounds):
    │
    ├─ Server broadcasts global model
    │
    ├─ Each client:
    │  ├─ Receives weights
    │  ├─ Trains locally (2 epochs)
    │  └─ Returns updated weights
    │
    └─ Server aggregates with FedAvg
    │
    ▼
Best Global Model Saved
    │
    ▼
Evaluation on Test Set
```

## Configuration System

The system uses a YAML configuration file (`configs/config.yaml`) to control all hyperparameters:

```yaml
seed: 42
output_dir: outputs

data:
  max_vocab_size: 20000
  max_seq_length: 256
  batch_size: 32
  val_split: 0.1

model:
  embedding_dim: 128
  hidden_dim: 256
  num_layers: 2
  dropout: 0.5
  bidirectional: true

centralized:
  learning_rate: 0.001
  weight_decay: 1.0e-5
  epochs: 10
  early_stopping_patience: 3

federated:
  num_clients: 5
  local_epochs: 2
  global_rounds: 10
  learning_rate: 0.001
  weight_decay: 1.0e-5
  alpha: 0.5

evaluation:
  batch_size: 64
```

## Privacy Considerations

### Federated Learning Privacy Benefits

1. **Data Locality**: Raw data never leaves client devices
2. **Aggregated Updates**: Only model weights are shared
3. **Differential Privacy Ready**: Architecture supports adding noise to gradients

### Non-IID Data Simulation

The Dirichlet distribution (controlled by alpha parameter) creates realistic heterogeneous data:
- Simulates different user preferences
- Each client has unique class distribution
- Tests model robustness to data heterogeneity

## Performance Characteristics

### Centralized Training
- **Speed**: Fast (single model, all data)
- **Accuracy**: Highest (~87%)
- **Privacy**: None (all data centralized)

### Federated Learning
- **Speed**: Slower (communication overhead)
- **Accuracy**: Slightly lower (~84%)
- **Privacy**: High (data stays local)
- **Scalability**: Better (distributable)

## Extension Points

The architecture is designed for easy extension:

1. **New Models**: Add to `src/models/sentiment_model.py`
2. **New Preprocessing**: Extend `src/data/preprocess.py`
3. **New FL Algorithms**: Implement in `src/federated/`
4. **New Metrics**: Add to `src/utils/utils.py`
5. **Differential Privacy**: Add noise in `src/federated/client.py`

## Dependencies

**Core:**
- PyTorch 2.0+
- transformers (HuggingFace)
- datasets (HuggingFace)

**NLP:**
- nltk (tokenization, stopwords)

**Data Science:**
- numpy
- scikit-learn (metrics, splits)

**Visualization:**
- matplotlib
- seaborn

**Configuration:**
- PyYAML

## Testing Strategy

While formal tests are not included, the system uses:

1. **Validation Sets**: Monitor overfitting
2. **Multiple Metrics**: Accuracy, Precision, Recall, F1
3. **Visualization**: Confusion matrices catch prediction issues
4. **Comparison**: Centralized vs Federated validates FL implementation

## Future Architecture Enhancements

1. **Differential Privacy Module**: Add noise calibration
2. **Secure Aggregation**: Implement cryptographic protocols
3. **Client Sampling**: Dynamic client selection each round
4. **Asynchronous FL**: Support asynchronous updates
5. **Model Compression**: Quantization for efficient communication
6. **Multi-Task Learning**: Support multiple NLP tasks
