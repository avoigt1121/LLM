"""Configuration settings for the TinyLLM model."""

import torch

# Model hyperparameters
EMBED_DIM = 256
CONTEXT_SIZE = 512
NUM_HEADS = 4
NUM_LAYERS = 6

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20000
PRINT_INTERVAL = 50

# Data settings
USE_BPE = True  # Set False for character-level tokenization
DATA_FILE = "corpus.txt"

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Special tokens for BPE
SPECIAL_TOKENS = ["[UNK]", "[PAD]"]
