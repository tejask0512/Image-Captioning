# utils/metrics.py

import torch
import nltk
from nltk.translate.bleu_score import corpus_bleu

def decode_caption(caption_ids, vocab, skip_special_tokens=True):
    """Convert caption indices to words."""
    words = []
    for idx in caption_ids:
        word = vocab.idx2word[idx]
        if skip_special_tokens and word in ["<PAD>", "<SOS>", "<EOS>"]:
            continue
        words.append(word)
        if word == "<EOS>":
            break
    return " ".join(words)


def calculate_bleu_score(references, hypotheses):
    """
    Calculate BLEU score.
    
    Args:
        references: List of reference captions (tokenized)
        hypotheses: List of generated captions (tokenized)
        
    Returns:
        bleu1, bleu4: BLEU-1 and BLEU-4 scores
    """
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    return bleu1, bleu4


# train.py

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

from models.encoder import ImageEncoder
from models.decoder import TransformerDecoder
from models.caption_model import ImageCaptioningModel
from utils.data_loader import get_data_loaders
from utils.metrics import decode_caption, calculate_bleu_score
from configs.config import Config

def train_epoch(model, data_loader, criterion, optimizer, device, grad_clip=None):
    """Train for one epoch."""
    model.train()
    losses = []
    
    # Progress bar
    pbar = tqdm(data_loader, desc="Training")
    
    for batch in pbar:
        # Get batch data
        images = batch["image"].to(device)
        captions = batch["caption"].to(device)
        
        # Forward pass
        outputs = model(images, captions)
        
        # Calculate loss (ignore <SOS> token in target)
        target = captions[:, 1:]  # Remove <SOS> token
        output = outputs[:, :-1]  # Remove last prediction
        
        # Reshape for loss calculation
        output = output.reshape(-1, output.shape[2])
        target = target.reshape(-1)
        
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Update weights
        optimizer.step()
        
        # Track loss
        losses.append(loss.item())
        pbar.set_postfix({"loss": np.mean(losses[-100:])})
    
    return np.mean(losses)


def validate(model, data_loader, criterion, vocab, device):
    """Validate the model."""
    model.eval()
    losses = []
    all_references = []
    all_hypotheses = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Validation")
        
        for batch in pbar:
            # Get batch data
            images = batch["image"].to(device)
            captions = batch["caption"].to(device)
            
            # Forward pass
            outputs = model(images, captions)
            
            # Calculate loss (ignore <SOS> token in target)
            target = captions[:, 1:]  # Remove <SOS> token
            output = outputs[:, :-1]  # Remove last prediction
            
            # Reshape for loss calculation
            output = output.reshape(-1, output.shape[2])
            target = target.reshape(-1)
            
            loss = criterion(output, target)
            losses.append(loss.item())
            
            # Generate captions for BLEU calculation
            for i in range(len(images)):
                img = images[i].unsqueeze(0)
                
                # Generate caption with beam search
                generated_ids = model.generate_caption(img)
                
                # Convert to words
                generated_caption = decode_caption(generated_ids, vocab)
                
                # Get reference caption
                reference_ids = captions[i].tolist()
                reference_caption = decode_caption(reference_ids, vocab)
                
                # Tokenize for BLEU calculation
                generated_tokens = generated_caption.split()
                reference_tokens = reference_caption.split()
                
                all_hypotheses.append(generated_tokens)
                all_references.append([reference_tokens])
    
    # Calculate metrics
    bleu1, bleu4 = calculate_bleu_score(all_references, all_hypotheses)
    avg_loss = np.mean(losses)
    
    return avg_loss, bleu1, bleu4


def main():
    """Main training function."""
    config = Config()
    
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, vocab = get_data_loaders(config)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create model
    encoder = ImageEncoder(config.encoder_name, config.embed_dim)
    decoder = TransformerDecoder(
        len(vocab),
        config.embed_dim,
        config.num_heads,
        config.num_decoder_layers,
        config.dim_feedforward,
        config.dropout
    )
    
    model = ImageCaptioningModel(encoder, decoder, len(vocab), device)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Create checkpoint directory
    os.makedirs(config.checkpoint_path, exist_ok=True)
    
    # Training loop
    best_bleu4 = 0.0
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Train for one epoch
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, config.clip_grad
        )
        
        # Validate
        val_loss, bleu1, bleu4 = validate(model, val_loader, criterion, vocab, device)
        
        # Print stats
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, BLEU-1: {bleu1:.4f}, BLEU-4: {bleu4:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'bleu1': bleu1,
            'bleu4': bleu4
        }
        
        torch.save(checkpoint, os.path.join(
            config.checkpoint_path, f"checkpoint_epoch_{epoch+1}.pth"
        ))
        
        # Save best model
        if bleu4 > best_bleu4:
            best_bleu4 = bleu4
            torch.save(model.state_dict(), config.best_model_path)
            print(f"New best model saved with BLEU-4: {bleu4:.4f}")


if __name__ == "__main__":
    main()