# configs/config.py

class Config:
    # Data Configuration
    data_folder = "./data"
    captions_file = "captions.json"
    image_folder = "images"
    vocab_file = "vocabulary.json"
    train_val_split = 0.8
    
    # Image Configuration
    image_size = 224  # ResNet input size
    
    # Model Configuration
    encoder_name = "resnet50"  # CNN encoder
    hidden_dim = 512
    embed_dim = 512
    attention_dim = 512
    decoder_dim = 512
    dropout = 0.5
    
    # Transformer Configuration
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    
    # Training Configuration
    batch_size = 32
    num_workers = 4
    learning_rate = 3e-4
    num_epochs = 20
    clip_grad = 5.0
    
    # Vocabulary Configuration
    min_word_freq = 5
    max_caption_length = 30
    
    # Inference Configuration
    beam_size = 5
    
    # Checkpoint Configuration
    checkpoint_path = "./outputs/checkpoints"
    best_model_path = "./outputs/best_model.pth"
    
    # Device Configuration
    device = "cuda"  # or "cpu" if GPU not available