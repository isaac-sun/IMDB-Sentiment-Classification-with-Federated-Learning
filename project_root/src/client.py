"""
Federated Learning Client Module

This module implements the client-side logic for federated learning.
Each client:
1. Receives global model weights from the server
2. Trains locally on its private data
3. Returns updated weights to the server
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

from utils import AverageMeter, calculate_metrics


class FederatedClient:
    """
    Federated Learning Client.
    
    Represents a single client in the federated learning system.
    Each client holds its own local dataset and trains on it.
    """
    
    def __init__(self, client_id, model, trainloader, config, device):
        """
        Initialize federated client.
        
        Args:
            client_id: Unique identifier for this client
            model: PyTorch model (will receive global weights)
            trainloader: DataLoader for local training data
            config: Configuration dictionary
            device: Device to train on
        """
        self.client_id = client_id
        self.model = model
        self.trainloader = trainloader
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        federated_lr = float(config['federated']['learning_rate'])
        federated_weight_decay = float(config['federated']['weight_decay'])
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=federated_lr,
            weight_decay=federated_weight_decay
        )
        
        # Local training parameters
        self.local_epochs = config['federated']['local_epochs']
        
    def set_model_weights(self, global_weights):
        """
        Set the model weights from the global model.
        
        Args:
            global_weights: State dict from global model
        """
        self.model.load_state_dict(global_weights)
    
    def get_model_weights(self):
        """
        Get the current model weights.
        
        Returns:
            State dict of current model
        """
        return self.model.state_dict()
    
    def train_local(self):
        """
        Train the model locally on this client's data.
        
        Returns:
            dict: Training metrics for this round
        """
        # Set model to training mode
        self.model.train()
        
        # Training statistics
        loss_meter = AverageMeter()
        all_preds = []
        all_targets = []
        
        # Train for local epochs
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for texts, lengths, labels in self.trainloader:
                texts = texts.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(texts, lengths)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Record metrics
                loss_meter.update(loss.item(), texts.size(0))
                all_preds.extend(outputs.detach().cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches
            print(f"    Client {self.client_id} - Epoch {epoch + 1}/{self.local_epochs} - Loss: {avg_epoch_loss:.4f}")
        
        # Calculate final metrics
        metrics = calculate_metrics(np.array(all_preds), np.array(all_targets))
        metrics['loss'] = loss_meter.avg
        metrics['num_samples'] = len(self.trainloader.dataset)
        
        return metrics
    
    def evaluate_local(self):
        """
        Evaluate the model on this client's local data.
        
        Returns:
            dict: Evaluation metrics
        """
        self.model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for texts, lengths, labels in self.trainloader:
                texts = texts.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(texts, lengths)
                
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        metrics = calculate_metrics(np.array(all_preds), np.array(all_targets))
        
        return metrics


def create_client(model, client_texts, client_labels, vocab, max_seq_length, 
                  batch_size, config, device, client_id):
    """
    Factory function to create a federated client.
    
    Args:
        model: PyTorch model template
        client_texts: List of texts for this client
        client_labels: List of labels for this client
        vocab: Vocabulary builder
        max_seq_length: Maximum sequence length
        batch_size: Batch size for DataLoader
        config: Configuration dictionary
        device: Device to train on
        client_id: Client ID
    
    Returns:
        FederatedClient: Initialized client
    """
    # Create dataset
    from train_centralized import IMDBDataset, collate_batch
    from preprocess import TextPreprocessor
    
    preprocessor = TextPreprocessor(use_stopwords=True)
    
    dataset = IMDBDataset(
        client_texts, client_labels, vocab,
        max_seq_length, preprocessor
    )
    
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, collate_fn=collate_batch
    )
    
    # Create client
    client = FederatedClient(
        client_id=client_id,
        model=model,
        trainloader=dataloader,
        config=config,
        device=device
    )
    
    return client


if __name__ == "__main__":
    # Test client functionality
    print("Federated Client module loaded successfully")
    print("This module provides the FederatedClient class for local training in FL")
