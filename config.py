"""Configuration settings for the TinyLLM model."""

import torch

# Model hyperparameters
EMBED_DIM = 16
CONTEXT_SIZE = 32
NUM_HEADS = 2
NUM_LAYERS = 1

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 2000
PRINT_INTERVAL = 50

# Data settings
USE_BPE = True  # Set False for character-level tokenization
DATA_FILE = "corpus.txt"

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Special tokens for BPE
SPECIAL_TOKENS = ["[UNK]", "[PAD]"]
