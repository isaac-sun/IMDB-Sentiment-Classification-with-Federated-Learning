"""
Utility Module for IMDB Federated Learning Project

This module provides utility functions for:
- Setting random seeds
- Loading configuration
- Saving/loading models
- Metrics calculation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import torch
import yaml
import json
from datetime import datetime


def _to_serializable(obj):
    """Recursively convert numpy/torch objects to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def set_seed(seed=42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def load_config(config_path='configs/config.yaml'):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_model(model, path):
    """
    Save model state dict to file.
    
    Args:
        model: PyTorch model
        path: Save path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to: {path}")


def load_model(model, path, device='cpu'):
    """
    Load model state dict from file.
    
    Args:
        model: PyTorch model
        path: Load path
        device: Device to load model on
    
    Returns:
        model: Model with loaded weights
    """
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def save_metrics(metrics, path):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        path: Save path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serializable_metrics = _to_serializable(metrics)
    with open(path, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    print(f"Metrics saved to: {path}")


def get_timestamp():
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_output_dirs(config):
    """
    Create output directories.
    
    Args:
        config: Configuration dictionary
    """
    output_dir = config.get('output', {}).get('models_dir', 'outputs/models')
    plots_dir = config.get('output', {}).get('plots_dir', 'outputs/plots')
    logs_dir = config.get('output', {}).get('logs_dir', 'outputs/logs')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    return output_dir, plots_dir, logs_dir


def calculate_metrics(preds, targets, threshold=0.5):
    """
    Calculate classification metrics.
    
    Args:
        preds: Predicted probabilities or logits
        targets: True labels
        threshold: Classification threshold
    
    Returns:
        dict: Dictionary of metrics
    """
    # Convert to binary predictions
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Handle logits (apply sigmoid if needed)
    if preds.max() > 1 or preds.min() < 0:
        preds = 1 / (1 + np.exp(-preds))
    
    pred_labels = (preds >= threshold).astype(int)
    
    # Calculate metrics
    tp = np.sum((pred_labels == 1) & (targets == 1))
    tn = np.sum((pred_labels == 0) & (targets == 0))
    fp = np.sum((pred_labels == 1) & (targets == 0))
    fn = np.sum((pred_labels == 0) & (targets == 1))
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    
    # Precision
    precision = tp / (tp + fp + 1e-10)
    
    # Recall
    recall = tp / (tp + fn + 1e-10)
    
    # F1 Score
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def print_metrics(metrics, prefix=""):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix string for printing
    """
    print(f"\n{prefix}Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        """
        Update statistics.
        
        Args:
            val: Current value
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    # Test utilities
    print("Testing configuration loading...")
    config = load_config('configs/config.yaml')
    print(f"Config loaded: {config['data']['name']}")
    
    print("\nTesting seed setting...")
    set_seed(42)
    
    print("\nTesting metrics calculation...")
    preds = np.array([0.9, 0.8, 0.3, 0.2])
    targets = np.array([1, 1, 0, 0])
    metrics = calculate_metrics(preds, targets)
    print_metrics(metrics)
