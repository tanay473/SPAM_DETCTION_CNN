import random
import numpy as np
import torch

# --- Reproducibility ---
def set_seed(seed_value=42):
    """Set seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

set_seed(42)

# --- CONFIG ---
MAX_LEN = 100  # Maximum sequence length for tokenization
BATCH_SIZE = 32  # Number of samples per batch
EMBED_DIM = 128  # Dimension of word embeddings
NUM_FILTERS = 16  # Number of filters (output channels) for the convolutional layer
KERNEL_SIZE = 5  # Size of the convolutional kernel
EPOCHS = 50  # Keep high to observe regularization effect, but use Early Stopping in practice
LR = 1e-3  # Learning rate for the optimizer

# --- Regularization Config ---
DROPOUT_PROB = 0.5  # Dropout probability
WEIGHT_DECAY = 1e-5  # L2 regularization strength
