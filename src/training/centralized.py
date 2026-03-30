"""
Centralized Training Script for IMDB Sentiment Classification

This script trains an LSTM model on the full IMDB training dataset
in a centralized manner for comparison with federated learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import os
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import download_imdb_dataset, split_dataset, TextPreprocessor, VocabularyBuilder, download_nltk_resources
from src.models import LSTMClassifier
from src.utils import (
    set_seed, load_config, save_model, save_metrics, 
    create_output_dirs, calculate_metrics, print_metrics,
    AverageMeter, get_timestamp
)


class IMDBDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for IMDB reviews.
    
    Handles tokenization, encoding, and padding of text data.
    """
    
    def __init__(self, texts, labels, vocab, max_seq_length, preprocessor):
        """
        Initialize dataset.
        
        Args:
            texts: List of text strings
            labels: List of labels (0 or 1)
            vocab: Vocabulary dictionary
            max_seq_length: Maximum sequence length
            preprocessor: TextPreprocessor instance
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.preprocessor = preprocessor
        
        # Pre-tokenize all texts
        print("Tokenizing texts...")
        self.encoded_texts = []
        self.lengths = []
        
        for text in tqdm(texts, desc="Encoding"):
            tokens = self.preprocessor.preprocess(text)
            indices = vocab.encode(tokens)
            
            # Truncate or pad
            if len(indices) > max_seq_length:
                indices = indices[:max_seq_length]
                length = max_seq_length
            else:
                length = len(indices)
                indices = indices + [vocab.vocab['<pad>']] * (max_seq_length - len(indices))
            
            self.encoded_texts.append(indices)
            self.lengths.append(length)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.encoded_texts[idx], dtype=torch.long),
            torch.tensor(self.lengths[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float)
        )


def collate_batch(batch):
    """
    Custom collate function for DataLoader.
    
    Args:
        batch: List of (text, length, label) tuples
    
    Returns:
        Batched tensors
    """
    texts, lengths, labels = zip(*batch)
    texts = torch.stack(texts)
    lengths = torch.stack(lengths)
    labels = torch.stack(labels)
    return texts, lengths, labels


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training DataLoader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
    
    Returns:
        dict: Training metrics
    """
    model.train()
    
    loss_meter = AverageMeter()
    all_preds = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc="Training")
    for texts, lengths, labels in pbar:
        texts = texts.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        # Record metrics
        loss_meter.update(loss.item(), texts.size(0))
        all_preds.extend(outputs.detach().cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss_meter.avg})
    
    # Calculate metrics
    metrics = calculate_metrics(np.array(all_preds), np.array(all_targets))
    metrics['loss'] = loss_meter.avg
    
    return metrics


def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: PyTorch model
        dataloader: Validation DataLoader
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        dict: Validation metrics
    """
    model.eval()
    
    loss_meter = AverageMeter()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for texts, lengths, labels in tqdm(dataloader, desc="Validation"):
            texts = texts.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)
            
            loss_meter.update(loss.item(), texts.size(0))
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    metrics = calculate_metrics(np.array(all_preds), np.array(all_targets))
    metrics['loss'] = loss_meter.avg
    
    return metrics


def main():
    """Main training function."""
    # Configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.yaml')
    config = load_config(config_path)
    
    # Set random seed
    set_seed(config['seed'])
    
    # Create output directories
    output_dir, plots_dir, logs_dir = create_output_dirs(config)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Download and prepare data
    print("\n" + "=" * 60)
    print("LOADING AND PREPARING DATA")
    print("=" * 60)
    
    train_dataset, test_dataset = download_imdb_dataset()
    train_dataset, val_dataset = split_dataset(train_dataset, val_split=config['data']['val_split'])
    
    # Extract texts and labels
    train_texts = [example['text'] for example in train_dataset]
    train_labels = [example['label'] for example in train_dataset]
    
    val_texts = [example['text'] for example in val_dataset]
    val_labels = [example['label'] for example in val_dataset]
    
    test_texts = [example['text'] for example in test_dataset]
    test_labels = [example['label'] for example in test_dataset]
    
    print(f"\nData splits:")
    print(f"  Training: {len(train_texts)} samples")
    print(f"  Validation: {len(val_texts)} samples")
    print(f"  Test: {len(test_texts)} samples")
    
    # Preprocessing
    print("\n" + "=" * 60)
    print("PREPROCESSING DATA")
    print("=" * 60)
    
    preprocessor = TextPreprocessor(use_stopwords=True)
    
    # Build vocabulary
    vocab_builder = VocabularyBuilder(max_vocab_size=config['data']['max_vocab_size'])
    vocab_builder.build_vocab(train_texts)
    vocab = vocab_builder
    
    # Create datasets
    train_ds = IMDBDataset(
        train_texts, train_labels, vocab, 
        config['data']['max_seq_length'], preprocessor
    )
    val_ds = IMDBDataset(
        val_texts, val_labels, vocab, 
        config['data']['max_seq_length'], preprocessor
    )
    test_ds = IMDBDataset(
        test_texts, test_labels, vocab, 
        config['data']['max_seq_length'], preprocessor
    )
    
    # Create dataloaders
    trainloader = DataLoader(
        train_ds, batch_size=config['data']['batch_size'],
        shuffle=True, collate_fn=collate_batch
    )
    valloader = DataLoader(
        val_ds, batch_size=config['evaluation']['batch_size'],
        shuffle=False, collate_fn=collate_batch
    )
    testloader = DataLoader(
        test_ds, batch_size=config['evaluation']['batch_size'],
        shuffle=False, collate_fn=collate_batch
    )
    
    # Initialize model
    print("\n" + "=" * 60)
    print("INITIALIZING MODEL")
    print("=" * 60)
    
    model = LSTMClassifier(
        vocab_size=len(vocab.vocab),
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        bidirectional=config['model']['bidirectional']
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    centralized_lr = float(config['centralized']['learning_rate'])
    centralized_weight_decay = float(config['centralized']['weight_decay'])
    optimizer = optim.Adam(
        model.parameters(),
        lr=centralized_lr,
        weight_decay=centralized_weight_decay
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("STARTING CENTRALIZED TRAINING")
    print("=" * 60)
    
    best_val_f1 = 0.0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    for epoch in range(config['centralized']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['centralized']['epochs']}")
        print("-" * 40)
        
        # Train
        train_metrics = train_epoch(model, trainloader, optimizer, criterion, device)
        print(f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f} | "
              f"F1: {train_metrics['f1']:.4f}")
        
        # Validate
        val_metrics = validate(model, valloader, criterion, device)
        print(f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"F1: {val_metrics['f1']:.4f}")
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            model_path = os.path.join(output_dir, 'centralized.pt')
            save_model(model, model_path)
            print(f"Best model saved! F1: {best_val_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['centralized']['early_stopping_patience']:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Load best model and evaluate on test set
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)
    
    model.load_state_dict(torch.load(os.path.join(output_dir, 'centralized.pt')))
    test_metrics = validate(model, testloader, criterion, device)
    print_metrics(test_metrics, prefix="Test ")
    
    # Save final metrics
    final_metrics = {
        'test_metrics': test_metrics,
        'best_val_f1': best_val_f1,
        'history': history,
        'config': config
    }
    save_metrics(final_metrics, os.path.join(logs_dir, 'centralized_metrics.json'))
    
    print("\n" + "=" * 60)
    print("CENTRALIZED TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {os.path.join(output_dir, 'centralized.pt')}")
    print(f"Metrics saved to: {os.path.join(logs_dir, 'centralized_metrics.json')}")
    
    return history, test_metrics


if __name__ == "__main__":
    history, test_metrics = main()
