"""
Preprocessing Module for IMDB Reviews

This module handles text preprocessing including:
- Lowercasing
- Punctuation removal
- Stopword removal
- Tokenization
- Vocabulary building
- Sequence padding
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import string
from collections import Counter
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm


# Download required NLTK data
def download_nltk_resources():
    """Download required NLTK resources for text preprocessing."""
    resources = ['stopwords', 'punkt', 'punkt_tab']
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            print(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, quiet=True)


class TextPreprocessor:
    """
    Text preprocessor for IMDB reviews.
    
    Handles lowercasing, punctuation removal, stopword removal,
    and tokenization.
    """
    
    def __init__(self, use_stopwords=True):
        """
        Initialize the text preprocessor.
        
        Args:
            use_stopwords: Whether to remove stopwords
        """
        self.use_stopwords = use_stopwords
        self.stop_words = set(stopwords.words('english')) if use_stopwords else set()
        
    def lowercase(self, text):
        """Convert text to lowercase."""
        return text.lower()
    
    def remove_punctuation(self, text):
        """
        Remove punctuation from text.
        
        Args:
            text: Input text string
        
        Returns:
            Text with punctuation removed
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of tokens
        
        Returns:
            List of tokens with stopwords removed
        """
        if not self.use_stopwords:
            return tokens
        return [token for token in tokens if token not in self.stop_words]
    
    def tokenize(self, text):
        """
        Tokenize text into words.
        
        Args:
            text: Input text string
        
        Returns:
            List of tokens
        """
        # Simple whitespace tokenization
        return text.split()
    
    def preprocess(self, text):
        """
        Full preprocessing pipeline.
        
        Args:
            text: Input text string
        
        Returns:
            List of preprocessed tokens
        """
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        return tokens


class VocabularyBuilder:
    """
    Builds and manages vocabulary for text data.
    
    Attributes:
        vocab: Dictionary mapping tokens to indices
        index_to_word: List mapping indices to tokens
        word_counts: Counter of word frequencies
    """
    
    def __init__(self, max_vocab_size=20000, min_freq=1):
        """
        Initialize vocabulary builder.
        
        Args:
            max_vocab_size: Maximum vocabulary size
            min_freq: Minimum word frequency to include
        """
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.vocab = {}
        self.index_to_word = []
        self.word_counts = Counter()
        
        # Add special tokens
        self.special_tokens = ['<pad>', '<unk>']
        for token in self.special_tokens:
            self.vocab[token] = len(self.index_to_word)
            self.index_to_word.append(token)
    
    def build_vocab(self, texts):
        """
        Build vocabulary from list of texts.
        
        Args:
            texts: List of text strings or list of token lists
        
        Returns:
            Vocabulary dictionary
        """
        print("\n" + "=" * 60)
        print("Building Vocabulary")
        print("=" * 60)
        
        # Count word frequencies
        for text in tqdm(texts, desc="Counting words"):
            if isinstance(text, str):
                tokens = text.lower().split()
            else:
                tokens = text
            self.word_counts.update(tokens)
        
        print(f"Total unique words: {len(self.word_counts)}")
        
        # Add words to vocabulary by frequency
        for word, count in self.word_counts.most_common(self.max_vocab_size - len(self.special_tokens)):
            if count >= self.min_freq:
                self.vocab[word] = len(self.index_to_word)
                self.index_to_word.append(word)
        
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Words excluded: {len(self.word_counts) - len(self.vocab) + len(self.special_tokens)}")
        
        return self.vocab
    
    def encode(self, text, max_length=None):
        """
        Encode text to indices.
        
        Args:
            text: Input text string or token list
            max_length: Maximum sequence length
        
        Returns:
            List of indices
        """
        if isinstance(text, str):
            tokens = text.lower().split()
        else:
            tokens = text
        
        indices = [
            self.vocab.get(token, self.vocab['<unk>']) 
            for token in tokens
        ]
        
        if max_length:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                indices = indices + [self.vocab['<pad>']] * (max_length - len(indices))
        
        return indices
    
    def decode(self, indices):
        """
        Decode indices to text.
        
        Args:
            indices: List of indices
        
        Returns:
            Decoded text string
        """
        tokens = [
            self.index_to_word[idx] for idx in indices 
            if idx != self.vocab['<pad>']
        ]
        return ' '.join(tokens)
    
    def save_vocab(self, path):
        """Save vocabulary to file."""
        with open(path, 'w') as f:
            for word, idx in self.vocab.items():
                f.write(f"{word}\t{idx}\n")
    
    def load_vocab(self, path):
        """Load vocabulary from file."""
        self.vocab = {}
        self.index_to_word = []
        with open(path, 'r') as f:
            for line in f:
                word, idx = line.strip().split('\t')
                idx = int(idx)
                self.vocab[word] = idx
                self.index_to_word.append(word)


def prepare_data(train_texts, val_texts, test_texts, max_vocab_size=20000, max_seq_length=256):
    """
    Prepare data for training.
    
    Args:
        train_texts: Training texts
        val_texts: Validation texts
        test_texts: Test texts
        max_vocab_size: Maximum vocabulary size
        max_seq_length: Maximum sequence length
    
    Returns:
        vocab: Vocabulary dictionary
        trainloader: Training DataLoader
        valloader: Validation DataLoader
        testloader: Test DataLoader
    """
    # Download NLTK resources
    download_nltk_resources()
    
    # Build vocabulary
    builder = VocabularyBuilder(max_vocab_size=max_vocab_size)
    builder.build_vocab(train_texts)
    
    return builder


if __name__ == "__main__":
    # Test preprocessing
    download_nltk_resources()
    
    preprocessor = TextPreprocessor(use_stopwords=True)
    
    sample_text = "I LOVED this movie! It was fantastic, with great acting and beautiful cinematography. <br /><br />The best film of the year!"
    
    print("Original text:")
    print(sample_text)
    print("\nPreprocessed tokens:")
    print(preprocessor.preprocess(sample_text))
