# models/encoder.py

import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, encoder_name="resnet50", embed_dim=512):
        """
        Image encoder using a pre-trained CNN.
        
        Args:
            encoder_name: Name of the pre-trained CNN model
            embed_dim: Dimension of the output embedding
        """
        super(ImageEncoder, self).__init__()
        
        # Load pre-trained model
        if encoder_name == "resnet50":
            cnn = models.resnet50(pretrained=True)
            # Remove the final fc layer
            modules = list(cnn.children())[:-1]
            self.cnn = nn.Sequential(*modules)
            self.fc = nn.Linear(2048, embed_dim)
        elif encoder_name == "resnet101":
            cnn = models.resnet101(pretrained=True)
            modules = list(cnn.children())[:-1]
            self.cnn = nn.Sequential(*modules)
            self.fc = nn.Linear(2048, embed_dim)
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")
        
    def forward(self, images):
        """
        Forward pass through the encoder.
        
        Args:
            images: Input images [batch_size, 3, height, width]
            
        Returns:
            features: Image features [batch_size, embed_dim]
        """
        # Extract features with CNN [batch_size, 2048, 1, 1]
        features = self.cnn(images)
        # Reshape to [batch_size, 2048]
        features = features.view(features.size(0), -1)
        # Project to embed_dim [batch_size, embed_dim]
        features = self.fc(features)
        
        return features


# models/decoder.py

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, 
                 num_layers=6, dim_feedforward=2048, dropout=0.1):
        """
        Transformer decoder for caption generation.
        
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout rate
        """
        super(TransformerDecoder, self).__init__()
        
        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights of the model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        """
        Forward pass through the decoder.
        
        Args:
            tgt: Target sequence [seq_len, batch_size]
            memory: Encoder output [1, batch_size, embed_dim]
            tgt_mask: Target sequence mask
            tgt_key_padding_mask: Target key padding mask
            
        Returns:
            output: Decoder output [seq_len, batch_size, vocab_size]
        """
        # Embed target sequence [seq_len, batch_size, embed_dim]
        tgt_embed = self.embedding(tgt)
        tgt_embed = self.positional_encoding(tgt_embed)
        
        # Pass through transformer decoder
        output = self.transformer_decoder(
            tgt_embed, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Project to vocabulary space [seq_len, batch_size, vocab_size]
        output = self.fc_out(output)
        
        return output


# models/caption_model.py

import torch
import torch.nn as nn
import math

class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder, vocab_size, device):
        """
        Complete image captioning model.
        
        Args:
            encoder: Image encoder module
            decoder: Caption decoder module
            vocab_size: Size of the vocabulary
            device: Device (cuda/cpu)
        """
        super(ImageCaptioningModel, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.device = device
        
    def generate_square_subsequent_mask(self, sz):
        """Generate mask for transformer decoder."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)
    
    def forward(self, images, captions):
        """
        Forward pass through the model.
        
        Args:
            images: Input images [batch_size, 3, height, width]
            captions: Input captions [batch_size, caption_length]
            
        Returns:
            outputs: Predicted word scores [batch_size, caption_length, vocab_size]
        """
        # Extract features from images [batch_size, embed_dim]
        image_features = self.encoder(images)
        
        # Reshape image features for transformer decoder
        # [1, batch_size, embed_dim]
        memory = image_features.unsqueeze(0)
        
        # Prepare target for transformer decoder
        # Convert [batch_size, caption_length] -> [caption_length, batch_size]
        tgt = captions.permute(1, 0)
        
        # Create masks
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0))
        tgt_key_padding_mask = (captions == 0).to(self.device)  # <PAD> token has idx 0
        
        # Forward through decoder
        # [caption_length, batch_size, vocab_size]
        outputs = self.decoder(tgt, memory, tgt_mask, tgt_key_padding_mask)
        
        # Convert [caption_length, batch_size, vocab_size] -> [batch_size, caption_length, vocab_size]
        outputs = outputs.permute(1, 0, 2)
        
        return outputs
    
    def generate_caption(self, image, max_length=30, beam_size=5):
        """
        Generate caption for an image using beam search.
        
        Args:
            image: Input image [1, 3, height, width]
            max_length: Maximum caption length
            beam_size: Beam size for beam search
            
        Returns:
            caption: Generated caption
        """
        # Set model to evaluation mode
        self.eval()
        
        with torch.no_grad():
            # Extract features from image
            image_features = self.encoder(image)  # [1, embed_dim]
            memory = image_features.unsqueeze(0)  # [1, 1, embed_dim]
            
            k = beam_size
            
            # Initialize beam with <SOS> token
            sequences = [[1]]  # <SOS> token idx is 1
            sequence_scores = torch.zeros(1).to(self.device)
            
            # Expand to beam size
            for _ in range(max_length - 1):
                candidates = []
                
                for idx, seq in enumerate(sequences):
                    # If sequence ended with <EOS>, add it to candidates
                    if seq[-1] == 2:  # <EOS> token idx is 2
                        candidates.append((sequence_scores[idx].item(), seq))
                        continue
                    
                    # Convert sequence to tensor
                    tgt = torch.LongTensor(seq).unsqueeze(1).to(self.device)  # [seq_len, 1]
                    
                    # Create mask
                    tgt_mask = self.generate_square_subsequent_mask(tgt.size(0))
                    
                    # Forward through decoder
                    output = self.decoder(tgt, memory, tgt_mask)  # [seq_len, 1, vocab_size]
                    
                    # Get prediction for next token
                    logits = output[-1, 0]  # [vocab_size]
                    
                    # Get top k predictions
                    topk_logits, topk_indices = torch.topk(logits, k)
                    
                    # Add to candidates
                    for j in range(k):
                        next_seq = seq + [topk_indices[j].item()]
                        next_score = sequence_scores[idx] + topk_logits[j]
                        candidates.append((next_score.item(), next_seq))
                
                # Select top k candidates
                candidates = sorted(candidates, key=lambda x: x[0], reverse=True)[:k]
                
                # Update sequences and scores
                sequences = [seq for _, seq in candidates]
                sequence_scores = torch.FloatTensor([score for score, _ in candidates]).to(self.device)
                
                # If all sequences end with <EOS>, break
                if all(seq[-1] == 2 for seq in sequences):
                    break
            
            # Select sequence with highest score
            _, best_sequence = max(zip(sequence_scores, sequences), key=lambda x: x[0])
            
            return best_sequence