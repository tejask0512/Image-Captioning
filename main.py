# main.py

import os
import argparse
import torch
import nltk

def main():
    """Main entry point for the image captioning project."""
    parser = argparse.ArgumentParser(description="Image Captioning with Transformers")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "evaluate", "predict"], 
                        help="Mode: train, evaluate, or predict")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--image", type=str, default=None, help="Path to image for prediction")
    parser.add_argument("--model", type=str, default="./outputs/best_model.pth", help="Path to the model")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for beam search")
    
    args = parser.parse_args()
    
    # Download NLTK data
    nltk.download('punkt')
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs("./outputs", exist_ok=True)
    
    if args.mode == "train":
        from train import main as train_main
        train_main()
    
    elif args.mode == "evaluate":
        from evaluate import main as evaluate_main
        evaluate_main()
    
    elif args.mode == "predict":
        if args.image is None:
            raise ValueError("Image path must be provided for prediction mode")
        
        from predict import main as predict_main
        predict_main()
    
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()