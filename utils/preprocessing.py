# utils/preprocessing.py

import os
import json
import torch
import nltk
import numpy as np
from PIL import Image
from collections import Counter
from tqdm import tqdm
from torchvision import transforms

nltk.download('punkt')

class Vocabulary:
    def __init__(self, min_word_freq=5):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.freq = Counter()
        self.min_word_freq = min_word_freq
    
    def add_captions(self, captions):
        """Add captions to build vocabulary."""
        for caption in captions:
            self.add_caption(caption)
    
    def add_caption(self, caption):
        """Add single caption to build vocabulary."""
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        self.freq.update(tokens)
    
    def build_vocabulary(self):
        """Build vocabulary based on word frequency."""
        words = [word for word, freq in self.freq.items() if freq >= self.min_word_freq]
        
        # Add words to dictionaries
        for idx, word in enumerate(words, start=len(self.word2idx)):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"Built vocabulary with {len(self.word2idx)} words")
    
    def save(self, vocab_file):
        """Save vocabulary to file."""
        with open(vocab_file, 'w') as f:
            json.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word
            }, f)
    
    @classmethod
    def load(cls, vocab_file):
        """Load vocabulary from file."""
        vocab = cls()
        with open(vocab_file, 'r') as f:
            data = json.load(f)
            vocab.word2idx = data['word2idx']
            vocab.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        return vocab
    
    def __len__(self):
        return len(self.word2idx)


def preprocess_captions(captions_file, vocab_file, min_word_freq=5, max_caption_length=30):
    """Preprocess captions and build vocabulary."""
    with open(captions_file, 'r') as f:
        data = json.load(f)
    
    # Extract captions
    all_captions = []
    image_captions = {}
    
    for item in data['annotations']:
        image_id = item['image_id']
        caption = item['caption']
        
        if image_id not in image_captions:
            image_captions[image_id] = []
        
        # Clean and tokenize caption
        caption = caption.lower().strip()
        if len(caption) > 0:
            all_captions.append(caption)
            if len(nltk.tokenize.word_tokenize(caption)) <= max_caption_length:
                image_captions[image_id].append(caption)
    
    # Build vocabulary
    vocab = Vocabulary(min_word_freq)
    vocab.add_captions(all_captions)
    vocab.build_vocabulary()
    vocab.save(vocab_file)
    
    return image_captions, vocab


def get_transform(image_size=224, is_train=True):
    """Get image transformation pipeline."""
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])