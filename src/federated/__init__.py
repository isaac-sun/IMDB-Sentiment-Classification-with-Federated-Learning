"""
Federated Learning Module - Client and Server Implementation

Implements federated learning components using the FedAvg algorithm.
"""

from .server import FederatedServer, fedavg_aggregate
from .client import FederatedClient

__all__ = [
    'FederatedServer',
    'fedavg_aggregate',
    'FederatedClient'
]
