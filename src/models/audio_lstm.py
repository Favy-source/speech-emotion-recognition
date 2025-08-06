"""
LSTM model for audio-based emotion recognition.
"""
import torch
import torch.nn as nn
from typing import Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioLSTM(nn.Module):
    """LSTM model for processing sequential audio features."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,  # anger, fear, happiness, sadness, calm
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super(AudioLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        logger.info(f"Initialized AudioLSTM: {input_size} -> {num_classes} classes")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling over sequence dimension
        pooled = torch.mean(attended, dim=1)
        
        # Classification
        out = self.dropout(pooled)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    def get_feature_size(self) -> int:
        """Return the expected input feature size."""
        return self.input_size
