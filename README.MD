# Image Captioning System

A deep learning-based image captioning system that generates natural language descriptions for images using CNNs, Transformers, and NLP techniques.

## Overview

This project implements an end-to-end image captioning system that combines computer vision and natural language processing. The system uses a CNN-based encoder to extract visual features from images and a Transformer-based decoder to generate descriptive captions.

## Features

- **Encoder-Decoder Architecture**: CNN for image encoding and Transformer for text decoding
- **Beam Search**: Advanced decoding strategy for better caption generation
- **Evaluation Metrics**: BLEU, CIDEr, and METEOR scoring
- **Pre-trained Models**: Leverages transfer learning with pre-trained CNN models
- **Comprehensive Pipeline**: Data preprocessing, training, evaluation, and inference

## Project Structure

```
image_captioning/
├── configs/              # Configuration files
│   └── config.py         # Hyperparameters and settings
├── data/                 # Dataset storage
├── models/               # Model architecture files
│   ├── encoder.py        # CNN image encoder
│   ├── decoder.py        # Transformer decoder
│   └── caption_model.py  # Complete captioning model
├── utils/                # Utility functions
│   ├── data_loader.py    # Dataset and dataloader utilities
│   ├── preprocessing.py  # Data preprocessing functions
│   └── metrics.py        # Evaluation metrics calculation
├── outputs/              # Model checkpoints and outputs
├── main.py               # Main entry point
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── predict.py            # Inference script
└── download_dataset.py   # Dataset download utility
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd image-captioning
```

2. Create a virtual environment (recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

Download and prepare a dataset using the provided utility:

```bash
# For Flickr8k dataset (smaller, ~1GB)
python download_dataset.py --dataset flickr8k --output_dir ./data

# For MS COCO dataset (larger, ~20GB)
python download_dataset.py --dataset coco --output_dir ./data
```

## Usage

### Training

Train the model with default parameters:

```bash
python main.py --mode train
```

The training script will:
- Process the dataset and build vocabulary
- Initialize the encoder-decoder architecture
- Train the model with specified hyperparameters
- Validate on a separate subset
- Save model checkpoints

### Evaluation

Evaluate a trained model:

```bash
python main.py --mode evaluate --model ./outputs/best_model.pth
```

This will calculate and report standard evaluation metrics (BLEU-1, BLEU-4, CIDEr, METEOR).

### Inference

Generate captions for new images:

```bash
python main.py --mode predict --image path/to/your/image.jpg --model ./outputs/best_model.pth
```

The generated caption will be displayed along with the image and saved to the outputs directory.

## Model Architecture

### Image Encoder
- Uses a pre-trained CNN (ResNet50 by default)
- Extracts visual features and projects to embedding space

### Caption Decoder
- Transformer-based architecture with self-attention mechanism
- Processes image features and generates text sequentially
- Implements positional encoding for sequence awareness

### Full Pipeline
```
Input Image → CNN Encoder → Feature Embedding → Transformer Decoder → Caption Output
```

## Customization

You can modify the model configuration in `configs/config.py`. Key parameters include:

- `encoder_name`: CNN backbone model (ResNet50, ResNet101)
- `hidden_dim`: Dimension of hidden layers
- `num_heads`: Number of attention heads in Transformer
- `num_encoder_layers`: Number of encoder layers in Transformer
- `num_decoder_layers`: Number of decoder layers in Transformer
- `batch_size`: Batch size for training
- `learning_rate`: Learning rate for optimization
- `beam_size`: Beam size for caption generation

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- transformers
- NLTK
- Pillow
- numpy
- matplotlib
- tqdm
- pycocoevalcap (for evaluation metrics)

## Acknowledgements

This project draws inspiration from:
- "Show, Attend and Tell" by Xu et al.
- "Attention Is All You Need" by Vaswani et al.
- "Bottom-Up and Top-Down Attention for Image Captioning" by Anderson et al.

## License

[MIT License](LICENSE)