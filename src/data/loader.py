"""
Data Loader Module for IMDB Reviews Dataset

This module handles loading and initial processing of the IMDB dataset
from HuggingFace datasets library.
"""

from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm


def download_imdb_dataset():
    """
    Download the IMDB dataset from HuggingFace.
    
    Returns:
        train_dataset: Training portion of the IMDB dataset
        test_dataset: Test portion of the IMDB dataset
    """
    print("=" * 60)
    print("Downloading IMDB Dataset from HuggingFace...")
    print("=" * 60)
    
    dataset = load_dataset("imdb")
    
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    
    # Print dataset information
    print(f"\nDataset loaded successfully!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Print class distribution for training set
    labels = [example['label'] for example in train_dataset]
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    
    print(f"\nClass Distribution (Training):")
    print(f"  Positive reviews: {pos_count} ({pos_count/len(labels)*100:.1f}%)")
    print(f"  Negative reviews: {neg_count} ({neg_count/len(labels)*100:.1f}%)")
    
    return train_dataset, test_dataset


def split_dataset(train_dataset, val_split=0.1, seed=42):
    """
    Split the training dataset into train and validation sets.
    
    Args:
        train_dataset: Original training dataset
        val_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
    
    Returns:
        train_data: List of training examples
        val_data: List of validation examples
    """
    print("\n" + "=" * 60)
    print("Splitting Dataset into Train/Validation Sets")
    print("=" * 60)
    
    # Convert dataset to list for splitting
    train_data = train_dataset.train_test_split(test_size=val_split, seed=seed)
    
    print(f"Training samples: {len(train_data['train'])}")
    print(f"Validation samples: {len(train_data['test'])}")
    
    return train_data['train'], train_data['test']


def create_client_datasets(train_data, num_clients=5, alpha=0.5, seed=42):
    """
    Create non-IID data distributions for federated learning clients.
    Uses Dirichlet distribution to simulate realistic non-IID data.
    
    Args:
        train_data: Training dataset
        num_clients: Number of clients
        alpha: Dirichlet distribution parameter (lower = more non-IID)
        seed: Random seed
    
    Returns:
        client_data: List of datasets for each client
        client_sizes: List of dataset sizes for each client
    """
    print("\n" + "=" * 60)
    print(f"Creating {num_clients} Client Datasets (Non-IID with alpha={alpha})")
    print("=" * 60)
    
    np.random.seed(seed)
    
    # Convert to list format
    texts = [example['text'] for example in train_data]
    labels = [example['label'] for example in train_data]
    
    n_samples = len(texts)
    
    # Generate Dirichlet distribution for each class
    # This creates different proportions of positive/negative for each client
    class_proportions = np.random.dirichlet([alpha] * num_clients, 2)  # 2 classes
    
    # Initialize client data lists
    client_texts = [[] for _ in range(num_clients)]
    client_labels = [[] for _ in range(num_clients)]
    
    # Assign samples to clients based on class proportions
    # First separate by class
    pos_indices = [i for i, l in enumerate(labels) if l == 1]
    neg_indices = [i for i, l in enumerate(labels) if l == 0]
    
    # Distribute positive-class samples
    pos_per_client = (class_proportions[1] * len(pos_indices)).astype(int)
    pos_per_client = np.maximum(pos_per_client, 1)  # Ensure at least 1 sample
    
    # Adjust to match total
    diff = sum(pos_per_client) - len(pos_indices)
    if diff > 0:
        pos_per_client[np.argmax(pos_per_client)] -= diff
    elif diff < 0:
        pos_per_client[np.argmin(pos_per_client)] -= diff
    
    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)
    
    # Assign positive samples to clients
    start_idx = 0
    for client_id in range(num_clients):
        end_idx = start_idx + pos_per_client[client_id]
        indices = pos_indices[start_idx:end_idx]
        for idx in indices:
            client_texts[client_id].append(texts[idx])
            client_labels[client_id].append(labels[idx])
        start_idx = end_idx
    
    # Distribute and assign negative-class samples
    neg_per_client = (class_proportions[0] * len(neg_indices)).astype(int)
    neg_per_client = np.maximum(neg_per_client, 1)
    
    diff = sum(neg_per_client) - len(neg_indices)
    if diff > 0:
        neg_per_client[np.argmax(neg_per_client)] -= diff
    elif diff < 0:
        neg_per_client[np.argmin(neg_per_client)] -= diff
    
    start_idx = 0
    for client_id in range(num_clients):
        end_idx = start_idx + neg_per_client[client_id]
        indices = neg_indices[start_idx:end_idx]
        for idx in indices:
            client_texts[client_id].append(texts[idx])
            client_labels[client_id].append(labels[idx])
        start_idx = end_idx
    
    # Shuffle each client's data
    for client_id in range(num_clients):
        combined = list(zip(client_texts[client_id], client_labels[client_id]))
        np.random.shuffle(combined)
        client_texts[client_id], client_labels[client_id] = zip(*combined)
        client_texts[client_id] = list(client_texts[client_id])
        client_labels[client_id] = list(client_labels[client_id])
    
    # Print client statistics
    client_sizes = []
    for client_id in range(num_clients):
        n_samples = len(client_labels[client_id])
        n_pos = sum(client_labels[client_id])
        n_neg = n_samples - n_pos
        pos_ratio = n_pos / n_samples * 100
        client_sizes.append(n_samples)
        print(f"  Client {client_id}: {n_samples} samples "
              f"(Pos: {n_pos} ({pos_ratio:.1f}%), Neg: {n_neg})")
    
    return client_texts, client_labels, client_sizes


if __name__ == "__main__":
    # Test data loading
    train_ds, test_ds = download_imdb_dataset()
    print("\nSample review:", train_ds[0]['text'][:200])
    print("Label:", train_ds[0]['label'])
