"""
Dataset Module for IMDB Reviews

Provides the IMDBDataset class and collate_batch function used
by both centralized and federated training pipelines.
"""

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class IMDBDataset(Dataset):
    """
    Custom Dataset for IMDB reviews.

    Pre-encodes all texts at construction time so that DataLoader
    workers only need to do simple index lookups.
    """

    def __init__(self, texts, labels, vocab, max_seq_length, preprocessor):
        """
        Initialize dataset.

        Args:
            texts: List of raw text strings
            labels: List of labels (0 or 1)
            vocab: VocabularyBuilder instance
            max_seq_length: Maximum sequence length (longer texts are truncated)
            preprocessor: TextPreprocessor instance
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.preprocessor = preprocessor

        # Pre-encode all texts once to avoid repeated work in __getitem__
        self.encoded_texts = []
        self.lengths = []

        for text in tqdm(texts, desc="Encoding"):
            tokens = self.preprocessor.preprocess(text)
            indices = vocab.encode(tokens)

            if len(indices) > max_seq_length:
                indices = indices[:max_seq_length]
                length = max_seq_length
            else:
                length = len(indices)
                indices = indices + [vocab.vocab['<pad>']] * (max_seq_length - length)

            self.encoded_texts.append(indices)
            self.lengths.append(length)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encoded_texts[idx], dtype=torch.long),
            torch.tensor(self.lengths[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float),
        )


def collate_batch(batch):
    """
    Collate function for DataLoader.

    Args:
        batch: List of (text_tensor, length_tensor, label_tensor) tuples

    Returns:
        Tuple of stacked tensors: (texts, lengths, labels)
    """
    texts, lengths, labels = zip(*batch)
    return torch.stack(texts), torch.stack(lengths), torch.stack(labels)
