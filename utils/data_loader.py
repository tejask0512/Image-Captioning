# utils/data_loader.py

import os
import torch
import json
import numpy as np
from PIL import Image
import nltk
from torch.utils.data import Dataset, DataLoader
from .preprocessing import get_transform

class ImageCaptionDataset(Dataset):
    def __init__(self, root_folder, image_captions, vocab, image_folder, transform=None, max_length=30):
        """
        Dataset for image captioning.
        
        Args:
            root_folder: Root folder containing images and captions
            image_captions: Dictionary mapping image_ids to captions
            vocab: Vocabulary object
            image_folder: Folder containing images
            transform: Image transformations
            max_length: Maximum caption length
        """
        self.root_folder = root_folder
        self.image_folder = os.path.join(root_folder, image_folder)
        self.vocab = vocab
        self.transform = transform
        self.max_length = max_length
        
        # Create dataset entries
        self.entries = []
        for image_id, captions in image_captions.items():
            image_path = os.path.join(self.image_folder, f"{image_id}.jpg")
            if os.path.exists(image_path):
                for caption in captions:
                    self.entries.append((image_id, caption))
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        image_id, caption = self.entries[idx]
        image_path = os.path.join(self.image_folder, f"{image_id}.jpg")
        
        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Tokenize and numericalize caption
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        caption = [self.vocab.word2idx.get(token, self.vocab.word2idx["<UNK>"]) for token in tokens]
        
        # Add <SOS> and <EOS> tokens
        caption = [self.vocab.word2idx["<SOS>"]] + caption + [self.vocab.word2idx["<EOS>"]]
        
        # Pad caption to max_length
        caption = caption[:self.max_length]
        caption_length = len(caption)
        
        # Pad with <PAD> tokens
        caption = caption + [self.vocab.word2idx["<PAD>"]] * (self.max_length - len(caption))
        
        return {
            "image_id": image_id,
            "image": image,
            "caption": torch.LongTensor(caption),
            "caption_length": caption_length
        }


def get_data_loaders(config):
    """Create train and validation data loaders."""
    # Load captions and vocabulary
    with open(os.path.join(config.data_folder, config.captions_file), 'r') as f:
        captions_data = json.load(f)
    
    # Load or create vocabulary
    vocab_path = os.path.join(config.data_folder, config.vocab_file)
    if os.path.exists(vocab_path):
        from .preprocessing import Vocabulary
        vocab = Vocabulary.load(vocab_path)
    else:
        from .preprocessing import preprocess_captions
        image_captions, vocab = preprocess_captions(
            os.path.join(config.data_folder, config.captions_file),
            vocab_path,
            config.min_word_freq,
            config.max_caption_length
        )
    
    # Process image IDs and captions
    image_captions = {}
    for item in captions_data['annotations']:
        image_id = item['image_id']
        caption = item['caption']
        
        if image_id not in image_captions:
            image_captions[image_id] = []
        
        image_captions[image_id].append(caption)
    
    # Split into train and validation sets
    image_ids = list(image_captions.keys())
    np.random.shuffle(image_ids)
    split_idx = int(len(image_ids) * config.train_val_split)
    
    train_image_ids = image_ids[:split_idx]
    val_image_ids = image_ids[split_idx:]
    
    train_captions = {img_id: image_captions[img_id] for img_id in train_image_ids}
    val_captions = {img_id: image_captions[img_id] for img_id in val_image_ids}
    
    # Create datasets
    train_transform = get_transform(config.image_size, is_train=True)
    val_transform = get_transform(config.image_size, is_train=False)
    
    train_dataset = ImageCaptionDataset(
        config.data_folder,
        train_captions,
        vocab,
        config.image_folder,
        transform=train_transform,
        max_length=config.max_caption_length
    )
    
    val_dataset = ImageCaptionDataset(
        config.data_folder,
        val_captions,
        vocab,
        config.image_folder,
        transform=val_transform,
        max_length=config.max_caption_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, vocab