import torch.nn as nn
from config import EMBED_DIM, NUM_FILTERS, KERNEL_SIZE, DROPOUT_PROB

class CNNClassifier(nn.Module):
    """A simple CNN for text classification with regularization."""
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, num_filters=NUM_FILTERS,
                 kernel_size=KERNEL_SIZE, num_classes=2, dropout_prob=DROPOUT_PROB):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, num_filters, kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Added Dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # [B, seq_len, embed_dim]
        x = x.permute(0, 2, 1)  # [B, embed_dim, seq_len] for Conv1d
        x = self.relu(self.conv1(x))  # [B, num_filters, seq_len - kernel_size + 1]
        x = self.pool(x).squeeze(-1)  # [B, num_filters]
        
        # Apply Dropout after pooling
        x = self.dropout(x)
        
        logits = self.fc(x)  # [B, num_classes]
        return logits