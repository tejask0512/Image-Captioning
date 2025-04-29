# predict.py

import os
import torch
import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from models.encoder import ImageEncoder
from models.decoder import TransformerDecoder
from models.caption_model import ImageCaptioningModel
from utils.preprocessing import Vocabulary
from configs.config import Config


def load_image(image_path, transform=None):
    """Load and preprocess image."""
    image = Image.open(image_path).convert("RGB")
    
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    image_tensor = transform(image)
    return image, image_tensor.unsqueeze(0)


def generate_caption(model, image_tensor, vocab, max_length=30, beam_size=5):
    """Generate caption for the image."""
    # Generate caption ids
    caption_ids = model.generate_caption(image_tensor, max_length, beam_size)
    
    # Convert ids to words
    words = []
    for idx in caption_ids:
        word = vocab.idx2word[idx]
        if word not in ["<PAD>", "< SOS >", "<EOS>", "<UNK>"]:
            words.append(word)
        if word == "<EOS>":
            break
    
    # Return caption
    return " ".join(words)


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Generate captions for images")
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    parser.add_argument("--model", type=str, default="./outputs/best_model.pth", help="Path to the trained model")
    parser.add_argument("--vocab", type=str, default="./data/vocabulary.json", help="Path to vocabulary file")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for beam search")
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load vocabulary
    vocab = Vocabulary.load(args.vocab)
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
    
    # Load trained weights
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    image, image_tensor = load_image(args.image)
    image_tensor = image_tensor.to(device)
    
    # Generate caption
    with torch.no_grad():
        caption = generate_caption(model, image_tensor, vocab, config.max_caption_length, args.beam_size)
    
    # Display image and caption
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title(caption)
    plt.axis("off")
    plt.savefig(f"./outputs/caption_{os.path.basename(args.image)}")
    plt.show()
    
    print(f"Generated caption: {caption}")


if __name__ == "__main__":
    main()