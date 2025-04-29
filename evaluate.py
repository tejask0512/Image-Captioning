# evaluate.py

import os
import torch
import json
import argparse
import numpy as np
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

from models.encoder import ImageEncoder
from models.decoder import TransformerDecoder
from models.caption_model import ImageCaptioningModel
from utils.data_loader import get_data_loaders
from utils.preprocessing import Vocabulary
from configs.config import Config


def evaluate_model(model, data_loader, vocab, device, beam_size=5):
    """Evaluate model on dataset."""
    model.eval()
    
    references = {}
    hypotheses = {}
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating")
        
        for batch in pbar:
            # Get batch data
            image_ids = batch["image_id"]
            images = batch["image"].to(device)
            captions = batch["caption"].to(device)
            
            # Generate captions for evaluation
            for i, img_id in enumerate(image_ids):
                img = images[i].unsqueeze(0)
                
                # Generate caption with beam search
                generated_ids = model.generate_caption(img, beam_size=beam_size)
                
                # Convert to words (skip special tokens)
                generated_tokens = []
                for idx in generated_ids:
                    word = vocab.idx2word[idx]
                    if word not in ["<PAD>", "< SOS >", "<EOS>", "<UNK>"]:
                        generated_tokens.append(word)
                    if word == "<EOS>":
                        break
                
                # Reference caption
                reference_ids = captions[i].tolist()
                reference_tokens = []
                for idx in reference_ids:
                    word = vocab.idx2word[idx]
                    if word not in ["<PAD>", "< SOS >", "<EOS>", "<UNK>"]:
                        reference_tokens.append(word)
                    if word == "<EOS>":
                        break
                
                # Add to dictionaries
                if img_id not in references:
                    references[img_id] = []
                
                references[img_id].append(" ".join(reference_tokens))
                hypotheses[img_id] = " ".join(generated_tokens)
    
    # Convert format for metrics calculation
    references_for_bleu = [[ref.split()] for ref in list(references.values())]
    hypotheses_for_bleu = [hyp.split() for hyp in list(hypotheses.values())]
    
    # Calculate BLEU scores
    bleu1 = corpus_bleu(references_for_bleu, hypotheses_for_bleu, weights=(1, 0, 0, 0))
    bleu4 = corpus_bleu(references_for_bleu, hypotheses_for_bleu, weights=(0.25, 0.25, 0.25, 0.25))
    
    # Calculate CIDEr score
    cider_scorer = Cider()
    
    # Format for CIDEr
    refs = {idx: [references[idx][0]] for idx in references}
    hyps = {idx: [hypotheses[idx]] for idx in hypotheses}
    
    (cider_score, _) = cider_scorer.compute_score(refs, hyps)
    
    # Calculate METEOR score
    meteor_scorer = Meteor()
    (meteor_score, _) = meteor_scorer.compute_score(refs, hyps)
    
    return {
        "BLEU-1": bleu1,
        "BLEU-4": bleu4,
        "CIDEr": cider_score,
        "METEOR": meteor_score
    }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate image captioning model")
    parser.add_argument("--model", type=str, default="./outputs/best_model.pth", help="Path to the trained model")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for beam search")
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    _, val_loader, vocab = get_data_loaders(config)
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
    
    # Evaluate model
    metrics = evaluate_model(model, val_loader, vocab, device, args.beam_size)
    
    # Print results
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save results
    os.makedirs("./outputs", exist_ok=True)
    with open("./outputs/evaluation_results.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()