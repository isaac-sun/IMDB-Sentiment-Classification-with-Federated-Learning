"""
Federated Learning Server Module

This module implements the server-side logic for federated learning.
The server:
1. Maintains the global model
2. Distributes weights to clients
3. Aggregates client updates using FedAvg
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import copy
from tqdm import tqdm


class FederatedServer:
    """
    Federated Learning Server.
    
    Coordinates the federated learning process:
    1. Maintains global model
    2. Broadcasts model to clients
    3. Aggregates client updates using FedAvg
    """
    
    def __init__(self, model, config, device):
        """
        Initialize federated server.
        
        Args:
            model: PyTorch model (global model)
            config: Configuration dictionary
            device: Device to run model on
        """
        self.global_model = model
        self.config = config
        self.device = device
        self.current_round = 0
        
        # Move model to device
        self.global_model.to(device)
        
        # Initialize metrics history
        self.history = {
            'rounds': [],
            'avg_client_loss': [],
            'avg_client_acc': [],
            'avg_client_f1': []
        }
    
    def get_global_model(self):
        """
        Get the current global model weights.
        
        Returns:
            State dict of global model
        """
        return self.global_model.state_dict()
    
    def set_global_model(self, state_dict):
        """
        Set the global model weights.
        
        Args:
            state_dict: State dict to load
        """
        self.global_model.load_state_dict(state_dict)
    
    def aggregate_weights(self, client_weights, client_sizes):
        """
        Aggregate client model weights using FedAvg.
        
        FedAvg (Federated Averaging):
        w_global = sum_i (n_i / n_total) * w_i
        
        where n_i is the number of samples on client i.
        
        Args:
            client_weights: List of state dicts from clients
            client_sizes: List of dataset sizes for each client
        
        Returns:
            Aggregated state dict
        """
        total_samples = sum(client_sizes)
        
        # Initialize aggregated weights
        aggregated_weights = copy.deepcopy(client_weights[0])
        
        # Zero out for proper weighted averaging
        for key in aggregated_weights.keys():
            aggregated_weights[key] = torch.zeros_like(aggregated_weights[key])
        
        # Weighted average
        for client_weight, client_size in zip(client_weights, client_sizes):
            weight_ratio = client_size / total_samples
            
            for key in client_weight.keys():
                # Handle different tensor types
                if isinstance(client_weight[key], torch.Tensor):
                    aggregated_weights[key] += client_weight[key].float() * weight_ratio
                else:
                    aggregated_weights[key] += client_weight[key] * weight_ratio
        
        return aggregated_weights
    
    def update_global_model(self, client_weights, client_sizes):
        """
        Update global model by aggregating client weights.
        
        Args:
            client_weights: List of state dicts from clients
            client_sizes: List of dataset sizes for each client
        
        Returns:
            Updated global model
        """
        # Aggregate weights
        aggregated_weights = self.aggregate_weights(client_weights, client_sizes)
        
        # Update global model
        self.global_model.load_state_dict(aggregated_weights)
        
        return self.global_model
    
    def log_round(self, round_num, metrics):
        """
        Log metrics for a round.
        
        Args:
            round_num: Current round number
            metrics: Dictionary of metrics from clients
        """
        self.current_round = round_num
        
        # Record history
        self.history['rounds'].append(round_num)
        self.history['avg_client_loss'].append(metrics.get('avg_loss', 0))
        self.history['avg_client_acc'].append(metrics.get('avg_acc', 0))
        self.history['avg_client_f1'].append(metrics.get('avg_f1', 0))
        
        # Print summary
        print(f"\n  Round {round_num} Summary:")
        print(f"    Avg Client Loss: {metrics.get('avg_loss', 0):.4f}")
        print(f"    Avg Client Acc:  {metrics.get('avg_acc', 0):.4f}")
        print(f"    Avg Client F1:   {metrics.get('avg_f1', 0):.4f}")
    
    def broadcast_to_clients(self, clients):
        """
        Broadcast global model weights to all clients.
        
        Args:
            clients: List of FederatedClient instances
        """
        global_weights = self.get_global_model()
        
        for client in clients:
            client.set_model_weights(copy.deepcopy(global_weights))
    
    def collect_from_clients(self, clients):
        """
        Collect model updates from all clients.
        
        Args:
            clients: List of FederatedClient instances
        
        Returns:
            tuple: (list of weights, list of sizes)
        """
        client_weights = []
        client_sizes = []
        
        for client in clients:
            client_weights.append(client.get_model_weights())
            client_sizes.append(len(client.trainloader.dataset))
        
        return client_weights, client_sizes


def fedavg_aggregate(client_weights, client_sizes):
    """
    Standalone FedAvg aggregation function.
    
    This function performs the Federated Averaging (FedAvg) algorithm.
    
    FedAvg Algorithm:
    1. Each client trains locally for E epochs
    2. Server aggregates weights: w = sum_i (n_i/n * w_i)
    
    Args:
        client_weights: List of state dicts from clients
        client_sizes: List of dataset sizes for each client
    
    Returns:
        Aggregated state dict
    """
    total_samples = sum(client_sizes)
    n_clients = len(client_weights)
    
    # Initialize aggregated weights
    aggregated_weights = copy.deepcopy(client_weights[0])
    
    # Zero out for proper weighted averaging
    for key in aggregated_weights.keys():
        if isinstance(aggregated_weights[key], torch.Tensor):
            aggregated_weights[key] = torch.zeros_like(aggregated_weights[key])
    
    # Weighted average
    for client_weight, client_size in zip(client_weights, client_sizes):
        weight_ratio = client_size / total_samples
        
        for key in client_weight.keys():
            if isinstance(client_weight[key], torch.Tensor):
                aggregated_weights[key] += client_weight[key].float() * weight_ratio
            else:
                aggregated_weights[key] += client_weight[key] * weight_ratio
    
    return aggregated_weights


if __name__ == "__main__":
    # Test server functionality
    print("Federated Server module loaded successfully")
    print("This module provides the FederatedServer class for FL coordination")
