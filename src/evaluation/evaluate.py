"""
Evaluation Module for IMDB Sentiment Classification

This module provides evaluation functionality:
- Metrics calculation (accuracy, precision, recall, F1)
- Confusion matrix generation
- Visualization (plots, curves)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
import sys

# Add project root to path so the package can be imported when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models import LSTMClassifier
from src.utils import load_config, calculate_metrics, print_metrics
from src.data import (
    TextPreprocessor, VocabularyBuilder, download_nltk_resources,
    download_imdb_dataset, IMDBDataset, collate_batch,
)


def find_artifact_path(filename, search_dirs):
    """Return the first existing artifact path from candidate directories."""
    for directory in search_dirs:
        candidate = os.path.join(directory, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def load_json_safe(json_path, label):
    """Load JSON safely and return None when content is missing or malformed."""
    if not json_path or not os.path.exists(json_path):
        return None

    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Failed to parse {label} at {json_path}: {e}")
        return None


def load_trained_model(model_path, config, vocab_size, device):
    """
    Load a trained model from file.
    
    Args:
        model_path: Path to model file
        config: Configuration dictionary
        vocab_size: Size of vocabulary
        device: Device to load model on
    
    Returns:
        model: Loaded PyTorch model
    """
    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        bidirectional=config['model']['bidirectional']
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate a model and return detailed metrics.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        criterion: Loss function
        device: Device
    
    Returns:
        dict: Evaluation metrics and predictions
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    loss_total = 0
    num_batches = 0
    
    with torch.no_grad():
        for texts, lengths, labels in dataloader:
            texts = texts.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)
            
            loss_total += loss.item()
            num_batches += 1
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            
            all_preds.extend(outputs.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    metrics = calculate_metrics(all_probs, all_targets)
    metrics['loss'] = loss_total / num_batches
    
    return {
        'metrics': metrics,
        'predictions': all_preds,
        'probabilities': all_probs,
        'targets': all_targets
    }


def plot_confusion_matrix(results, save_path):
    """
    Plot and save confusion matrix.
    
    Args:
        results: Evaluation results dictionary
        save_path: Path to save plot
    """
    y_true = results['targets']
    y_pred = (results['probabilities'] >= 0.5).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {save_path}")


def plot_training_curves(centralized_history, federated_history, save_path):
    """
    Plot training curves comparing centralized and federated learning.
    
    Args:
        centralized_history: Training history from centralized training
        federated_history: Training history from federated training
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Centralized training
    if centralized_history:
        epochs = range(1, len(centralized_history['train_loss']) + 1)
        
        # Loss curve
        axes[0].plot(epochs, centralized_history['train_loss'], 'b-', label='Train')
        axes[0].plot(epochs, centralized_history['val_loss'], 'r-', label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Centralized - Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy curve
        axes[1].plot(epochs, centralized_history['train_acc'], 'b-', label='Train')
        axes[1].plot(epochs, centralized_history['val_acc'], 'r-', label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Centralized - Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        # F1 curve
        axes[2].plot(epochs, centralized_history['train_f1'], 'b-', label='Train')
        axes[2].plot(epochs, centralized_history['val_f1'], 'r-', label='Val')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1 Score')
        axes[2].set_title('Centralized - F1 Score')
        axes[2].legend()
        axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Centralized training curves saved to: {save_path}")


def plot_federated_curves(federated_history, save_path):
    """
    Plot federated learning training curves.
    
    Args:
        federated_history: Training history from federated training
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    rounds = federated_history['rounds']
    
    # Client average loss
    axes[0].plot(rounds, federated_history['avg_client_loss'], 'b-', marker='o')
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Federated - Avg Client Loss')
    axes[0].grid(True)
    
    # Client average accuracy
    axes[1].plot(rounds, federated_history['avg_client_acc'], 'g-', marker='o')
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Federated - Avg Client Accuracy')
    axes[1].grid(True)
    
    # Client average F1
    axes[2].plot(rounds, federated_history['avg_client_f1'], 'r-', marker='o')
    axes[2].set_xlabel('Round')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('Federated - Avg Client F1')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Federated training curves saved to: {save_path}")


def plot_model_comparison(centralized_metrics, federated_metrics, save_path):
    """
    Plot bar chart comparing centralized and federated model performance.
    
    Args:
        centralized_metrics: Metrics from centralized model
        federated_metrics: Metrics from federated model
        save_path: Path to save plot
    """
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    
    centralized_values = [
        centralized_metrics['accuracy'],
        centralized_metrics['precision'],
        centralized_metrics['recall'],
        centralized_metrics['f1']
    ]
    
    federated_values = [
        federated_metrics['accuracy'],
        federated_metrics['precision'],
        federated_metrics['recall'],
        federated_metrics['f1']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, centralized_values, width, label='Centralized', color='steelblue')
    bars2 = ax.bar(x + width/2, federated_values, width, label='Federated', color='darkorange')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Centralized vs Federated Model Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim((0, 1.1))
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison saved to: {save_path}")


def main():
    """Main evaluation function."""
    # Resolve project root so the script can be run from any working directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(project_root, 'configs', 'config.yaml')
    config = load_config(config_path)

    # Create output directories
    output_dir = os.path.join(project_root, 'outputs')
    plots_dir = os.path.join(output_dir, 'plots')
    model_search_dirs = [os.path.join(project_root, 'outputs', 'models')]
    log_search_dirs = [os.path.join(project_root, 'outputs', 'logs')]
    
    os.makedirs(plots_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Download dataset
    print("\nLoading dataset...")
    train_dataset, test_dataset = download_imdb_dataset()
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    train_texts = [example['text'] for example in train_dataset]
    vocab_builder = VocabularyBuilder(max_vocab_size=config['data']['max_vocab_size'])
    vocab_builder.build_vocab(train_texts)
    vocab = vocab_builder
    
    # Prepare test data
    test_texts = [example['text'] for example in test_dataset]
    test_labels = [example['label'] for example in test_dataset]
    
    preprocessor = TextPreprocessor(use_stopwords=True)
    test_ds = IMDBDataset(
        test_texts, test_labels, vocab,
        config['data']['max_seq_length'], preprocessor
    )
    testloader = DataLoader(
        test_ds, batch_size=config['evaluation']['batch_size'],
        shuffle=False, collate_fn=collate_batch
    )
    
    criterion = nn.BCEWithLogitsLoss()
    
    centralized_metrics = None
    federated_metrics = None
    centralized_history = None
    federated_history = None
    
    # Load centralized model
    centralized_path = find_artifact_path('centralized.pt', model_search_dirs)
    if centralized_path and os.path.exists(centralized_path):
        print("\n" + "=" * 60)
        print("EVALUATING CENTRALIZED MODEL")
        print("=" * 60)
        print(f"Using centralized model: {centralized_path}")
        
        model = load_trained_model(centralized_path, config, len(vocab.vocab), device)
        results = evaluate_model(model, testloader, criterion, device)
        centralized_metrics = results['metrics']
        print_metrics(centralized_metrics, prefix="Centralized Test ")
        
        # Confusion matrix
        plot_confusion_matrix(results, os.path.join(plots_dir, 'centralized_confusion_matrix.png'))
        
        # Load centralized history
        centralized_metrics_path = find_artifact_path('centralized_metrics.json', log_search_dirs)
        centralized_log = load_json_safe(centralized_metrics_path, 'centralized metrics')
        if centralized_log and 'history' in centralized_log:
            centralized_history = centralized_log['history']
    else:
        print("Warning: No centralized model found in expected output directories.")
    
    # Load federated model
    federated_path = find_artifact_path('federated.pt', model_search_dirs)
    if federated_path and os.path.exists(federated_path):
        print("\n" + "=" * 60)
        print("EVALUATING FEDERATED MODEL")
        print("=" * 60)
        print(f"Using federated model: {federated_path}")
        
        model = load_trained_model(federated_path, config, len(vocab.vocab), device)
        results = evaluate_model(model, testloader, criterion, device)
        federated_metrics = results['metrics']
        print_metrics(federated_metrics, prefix="Federated Test ")
        
        # Confusion matrix
        plot_confusion_matrix(results, os.path.join(plots_dir, 'federated_confusion_matrix.png'))
        
        # Load federated history
        federated_metrics_path = find_artifact_path('federated_metrics.json', log_search_dirs)
        federated_log = load_json_safe(federated_metrics_path, 'federated metrics')
        if federated_log and 'history' in federated_log:
            federated_history = federated_log['history']
    else:
        print("Warning: No federated model found in expected output directories.")
    
    # Generate comparison plots
    if centralized_history:
        plot_training_curves(
            centralized_history,
            federated_history,
            os.path.join(plots_dir, 'centralized_training_curves.png')
        )
    
    if federated_history:
        plot_federated_curves(
            federated_history,
            os.path.join(plots_dir, 'federated_training_curves.png')
        )
    
    if centralized_metrics and federated_metrics:
        plot_model_comparison(
            centralized_metrics,
            federated_metrics,
            os.path.join(plots_dir, 'model_comparison.png')
        )
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
