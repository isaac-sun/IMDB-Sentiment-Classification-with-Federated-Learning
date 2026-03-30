"""
Federated Learning Training Script for IMDB Sentiment Classification

This script implements the FedAvg algorithm for training an LSTM model
on IMDB reviews in a federated manner across multiple clients.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import copy
import json
import os
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import download_imdb_dataset, split_dataset, create_client_datasets, TextPreprocessor, VocabularyBuilder, download_nltk_resources
from src.models import LSTMClassifier
from src.federated import FederatedServer, fedavg_aggregate, FederatedClient
from src.training.centralized import IMDBDataset, collate_batch
from src.utils import (
    set_seed, load_config, save_model, save_metrics,
    create_output_dirs, calculate_metrics, print_metrics,
    AverageMeter, get_timestamp
)


def create_clients(model, client_texts, client_labels, vocab, max_seq_length,
                   batch_size, config, device):
    """
    Create federated learning clients.
    
    Args:
        model: PyTorch model template
        client_texts: List of text lists for each client
        client_labels: List of label lists for each client
        vocab: Vocabulary builder
        max_seq_length: Maximum sequence length
        batch_size: Batch size
        config: Configuration dictionary
        device: Device
    
    Returns:
        list: List of FederatedClient instances
    """
    num_clients = config['federated']['num_clients']
    preprocessor = TextPreprocessor(use_stopwords=True)
    
    clients = []
    for client_id in range(num_clients):
        # Create dataset
        dataset = IMDBDataset(
            client_texts[client_id],
            client_labels[client_id],
            vocab,
            max_seq_length,
            preprocessor
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_batch
        )
        
        # Create client with its own model copy
        client_model = copy.deepcopy(model)
        client = FederatedClient(
            client_id=client_id,
            model=client_model,
            trainloader=dataloader,
            config=config,
            device=device
        )
        
        clients.append(client)
        print(f"  Created client {client_id} with {len(dataset)} samples")
    
    return clients


def federated_training_round(server, clients):
    """
    Execute one round of federated training.
    
    Steps:
    1. Broadcast global model to clients
    2. Each client trains locally
    3. Collect updated weights from clients
    4. Aggregate weights using FedAvg
    5. Update global model
    
    Args:
        server: FederatedServer instance
        clients: List of FederatedClient instances
    
    Returns:
        dict: Aggregated metrics from this round
    """
    # Broadcast global model to all clients
    server.broadcast_to_clients(clients)
    
    # Each client trains locally
    client_metrics = []
    for client in clients:
        metrics = client.train_local()
        client_metrics.append(metrics)
    
    # Collect weights from clients
    client_weights, client_sizes = server.collect_from_clients(clients)
    
    # Aggregate weights using FedAvg
    aggregated_weights = fedavg_aggregate(client_weights, client_sizes)
    
    # Update global model
    server.set_global_model(aggregated_weights)
    
    # Calculate average metrics
    avg_metrics = {
        'avg_loss': np.mean([m['loss'] for m in client_metrics]),
        'avg_acc': np.mean([m['accuracy'] for m in client_metrics]),
        'avg_f1': np.mean([m['f1'] for m in client_metrics]),
        'client_metrics': client_metrics
    }
    
    return avg_metrics


def evaluate_global_model(model, testloader, criterion, device):
    """
    Evaluate the global model on test data.
    
    Args:
        model: PyTorch model
        testloader: Test DataLoader
        criterion: Loss function
        device: Device
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    loss_total = 0
    num_batches = 0
    
    with torch.no_grad():
        for texts, lengths, labels in testloader:
            texts = texts.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)
            
            loss_total += loss.item()
            num_batches += 1
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    metrics = calculate_metrics(np.array(all_preds), np.array(all_targets))
    metrics['loss'] = loss_total / num_batches
    
    return metrics


def main():
    """Main federated learning training function."""
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
    print("LOADING AND PREPARING DATA FOR FEDERATED LEARNING")
    print("=" * 60)
    
    train_dataset, test_dataset = download_imdb_dataset()
    train_dataset, val_dataset = split_dataset(train_dataset, val_split=config['data']['val_split'])
    
    # Extract texts and labels
    train_texts = [example['text'] for example in train_dataset]
    train_labels = [example['label'] for example in train_dataset]
    
    test_texts = [example['text'] for example in test_dataset]
    test_labels = [example['label'] for example in test_dataset]
    
    print(f"\nTotal training samples: {len(train_texts)}")
    
    # Create non-IID client data distributions
    client_texts, client_labels, client_sizes = create_client_datasets(
        train_dataset,
        num_clients=config['federated']['num_clients'],
        alpha=config['federated']['alpha'],
        seed=config['seed']
    )
    
    # Preprocessing
    print("\n" + "=" * 60)
    print("PREPROCESSING DATA")
    print("=" * 60)
    
    preprocessor = TextPreprocessor(use_stopwords=True)
    
    # Build vocabulary (using all training data)
    vocab_builder = VocabularyBuilder(max_vocab_size=config['data']['max_vocab_size'])
    vocab_builder.build_vocab(train_texts)
    vocab = vocab_builder
    
    # Create test dataset
    test_ds = IMDBDataset(
        test_texts, test_labels, vocab,
        config['data']['max_seq_length'], preprocessor
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
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create server
    server = FederatedServer(model, config, device)
    
    # Create clients
    print("\n" + "=" * 60)
    print("CREATING FEDERATED CLIENTS")
    print("=" * 60)
    
    clients = create_clients(
        model=model,
        client_texts=client_texts,
        client_labels=client_labels,
        vocab=vocab,
        max_seq_length=config['data']['max_seq_length'],
        batch_size=config['data']['batch_size'],
        config=config,
        device=device
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("STARTING FEDERATED TRAINING")
    print("=" * 60)
    
    criterion = nn.BCEWithLogitsLoss()
    
    history = {
        'rounds': [],
        'avg_client_loss': [],
        'avg_client_acc': [],
        'avg_client_f1': [],
        'test_loss': [],
        'test_acc': [],
        'test_f1': []
    }
    
    best_test_f1 = 0.0
    
    for round_num in range(1, config['federated']['global_rounds'] + 1):
        print(f"\n{'='*60}")
        print(f"FEDERATED ROUND {round_num}/{config['federated']['global_rounds']}")
        print(f"{'='*60}")
        
        # Execute federated training round
        round_metrics = federated_training_round(server, clients)
        
        # Log round metrics
        server.log_round(round_num, round_metrics)
        
        # Evaluate global model on test set
        test_metrics = evaluate_global_model(
            server.global_model, testloader, criterion, device
        )
        
        print(f"\n  Global Model Test Metrics:")
        print(f"    Test Loss: {test_metrics['loss']:.4f}")
        print(f"    Test Acc:  {test_metrics['accuracy']:.4f}")
        print(f"    Test F1:   {test_metrics['f1']:.4f}")
        
        # Record history
        history['rounds'].append(round_num)
        history['avg_client_loss'].append(round_metrics['avg_loss'])
        history['avg_client_acc'].append(round_metrics['avg_acc'])
        history['avg_client_f1'].append(round_metrics['avg_f1'])
        history['test_loss'].append(test_metrics['loss'])
        history['test_acc'].append(test_metrics['accuracy'])
        history['test_f1'].append(test_metrics['f1'])
        
        # Save best model
        if test_metrics['f1'] > best_test_f1:
            best_test_f1 = test_metrics['f1']
            model_path = os.path.join(output_dir, 'federated.pt')
            save_model(server.global_model, model_path)
            print(f"  Best model saved! F1: {best_test_f1:.4f}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    # Load best model
    server.global_model.load_state_dict(
        torch.load(os.path.join(output_dir, 'federated.pt'))
    )
    
    final_metrics = evaluate_global_model(
        server.global_model, testloader, criterion, device
    )
    print_metrics(final_metrics, prefix="Final Test ")
    
    # Save final metrics
    final_results = {
        'test_metrics': final_metrics,
        'best_test_f1': best_test_f1,
        'history': history,
        'config': config,
        'client_sizes': client_sizes
    }
    save_metrics(final_results, os.path.join(logs_dir, 'federated_metrics.json'))
    
    print("\n" + "=" * 60)
    print("FEDERATED TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {os.path.join(output_dir, 'federated.pt')}")
    print(f"Metrics saved to: {os.path.join(logs_dir, 'federated_metrics.json')}")
    
    return history, final_metrics


if __name__ == "__main__":
    history, final_metrics = main()
