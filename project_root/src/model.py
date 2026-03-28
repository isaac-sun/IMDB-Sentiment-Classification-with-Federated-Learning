"""
Model Module for IMDB Sentiment Classification

This module implements neural network models for sentiment classification:
1. BaselineModel: Simple embedding + linear classifier
2. LSTMClassifier: LSTM-based classifier for better sequence understanding
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineModel(nn.Module):
    """
    Baseline model: Embedding layer followed by a linear classifier.
    
    This is a simple bag-of-words style model that averages word embeddings
    and applies a linear transformation for classification.
    
    Architecture:
        - Embedding Layer
        - Mean Pooling
        - Linear Classifier
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0.5):
        """
        Initialize baseline model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden layer
            dropout: Dropout rate
        """
        super(BaselineModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, lengths=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
            lengths: Actual sequence lengths (optional)
        
        Returns:
            Output logits of shape (batch_size,)
        """
        # x: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # Mean pooling (ignoring padding)
        if lengths is not None:
            # Create mask
            mask = (x != 0).unsqueeze(-1).float()  # (batch_size, seq_length, 1)
            embedded = embedded * mask
            pooled = embedded.sum(dim=1) / lengths.unsqueeze(-1).float()
        else:
            pooled = embedded.mean(dim=1)
        
        # Fully connected layers
        x = self.dropout(pooled)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x).squeeze(-1)
        
        return logits


class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for sentiment analysis.
    
    Uses bidirectional LSTM to capture sequential patterns in text,
    followed by fully connected layers for classification.
    
    Architecture:
        - Embedding Layer
        - Bidirectional LSTM
        - Fully Connected Layers
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, 
                 dropout=0.5, bidirectional=True):
        """
        Initialize LSTM classifier.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate input dimension for fully connected layer
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, lengths=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
            lengths: Actual sequence lengths (optional)
        
        Returns:
            Output logits of shape (batch_size,)
        """
        # x: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # Pack sequence if lengths provided (for efficiency)
        if lengths is not None and lengths.max().item() < x.size(1):
            # Pack the embedded sequences
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, 
                lengths.cpu(), 
                batch_first=True, 
                enforce_sorted=False
            )
            lstm_out, (hidden, cell) = self.lstm(packed)
            # Unpack
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, 
                batch_first=True,
                total_length=x.size(1)
            )
        else:
            lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state from both directions
        # hidden: (num_layers * num_directions, batch_size, hidden_dim)
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            # Take the last layer's hidden states
            forward_hidden = hidden[-2, :, :]  # Forward last layer
            backward_hidden = hidden[-1, :, :]  # Backward last layer
            hidden_concat = torch.cat((forward_hidden, backward_hidden), dim=1)
        else:
            hidden_concat = hidden[-1, :, :]
        
        # Fully connected layers
        x = F.relu(self.fc1(hidden_concat))
        x = self.dropout1(x)
        logits = self.fc2(x).squeeze(-1)
        
        return logits
    
    def get_embedding(self, x):
        """
        Get word embeddings for input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
        
        Returns:
            Embeddings of shape (batch_size, seq_length, embedding_dim)
        """
        return self.embedding(x)


def get_model(model_type, config, vocab_size):
    """
    Factory function to get a model by name.
    
    Args:
        model_type: Type of model ('baseline' or 'lstm')
        config: Configuration dictionary
        vocab_size: Size of vocabulary
    
    Returns:
        model: Initialized model
    """
    model_config = config['model']
    
    if model_type == 'baseline':
        model = BaselineModel(
            vocab_size=vocab_size,
            embedding_dim=model_config['embedding_dim'],
            hidden_dim=model_config['hidden_dim'],
            dropout=model_config['dropout']
        )
    elif model_type == 'lstm':
        model = LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=model_config['embedding_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout'],
            bidirectional=model_config['bidirectional']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    vocab_size = 10000
    batch_size = 16
    seq_length = 100
    
    # Create dummy input
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    lengths = torch.randint(5, seq_length, (batch_size,))
    
    print("Testing Baseline Model:")
    baseline = BaselineModel(vocab_size, 128, 256)
    output = baseline(x, lengths)
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {count_parameters(baseline):,}")
    
    print("\nTesting LSTM Model:")
    lstm = LSTMClassifier(vocab_size, 128, 256, 2, bidirectional=True)
    output = lstm(x, lengths)
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {count_parameters(lstm):,}")
