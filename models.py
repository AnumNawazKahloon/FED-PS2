import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MODEL
import numpy as np

class LSTMPredictor(nn.Module):
    """LSTM model for SOC prediction"""
    def __init__(self, input_size=None, hidden_size=None, num_layers=2, dropout_rate=None):
        super().__init__()
        self.input_size = input_size or MODEL['input_size']
        self.hidden_size = hidden_size or MODEL.get('lstm_hidden_size', 128)
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate or MODEL['dropout_rate']
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.hidden_size // 2, 1)
        
    def forward(self, x):
        # x shape: (batch_size, input_size) -> reshape to (batch_size, seq_len=1, input_size)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the last output
        out = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class BiLSTMPredictor(nn.Module):
    """Bidirectional LSTM model for SOC prediction"""
    def __init__(self, input_size=None, hidden_size=None, num_layers=2, dropout_rate=None):
        super().__init__()
        self.input_size = input_size or MODEL['input_size']
        self.hidden_size = hidden_size or MODEL.get('lstm_hidden_size', 128)
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate or MODEL['dropout_rate']
        
        # Bidirectional LSTM layers
        self.bilstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers (note: hidden_size * 2 because of bidirectional)
        self.fc1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.fc3 = nn.Linear(self.hidden_size // 2, 1)
        
    def forward(self, x):
        # x shape: (batch_size, input_size) -> reshape to (batch_size, seq_len=1, input_size)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # BiLSTM forward pass
        bilstm_out, (hidden, cell) = self.bilstm(x)
        
        # Take the last output
        out = bilstm_out[:, -1, :]
        
        # Fully connected layers
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out


class CNNLSTMPredictor(nn.Module):
    """CNN-LSTM hybrid model for SOC prediction"""
    def __init__(self, input_size=None, hidden_size=None, num_layers=2, dropout_rate=None):
        super().__init__()
        self.input_size = input_size or MODEL['input_size']
        self.hidden_size = hidden_size or MODEL.get('lstm_hidden_size', 128)
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate or MODEL['dropout_rate']
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=0)
        self.conv_dropout = nn.Dropout(self.dropout_rate)
        
        self.cnn_output_size = 128
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.hidden_size // 2, 1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Reshape for CNN: (batch_size, input_size) -> (batch_size, 1, input_size)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.conv_dropout(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.conv_dropout(x)
        
        # Reshape for LSTM: (batch_size, channels, features) -> (batch_size, features, channels)
        x = x.permute(0, 2, 1)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the last output
        out = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


# Legacy MLP model (keeping for compatibility)
class SOCPredictor(nn.Module):
    """Original MLP model for neural network-based prediction"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(MODEL['input_size'], MODEL['hidden_size'])
        self.dropout = nn.Dropout(MODEL['dropout_rate'])
        self.fc2 = nn.Linear(MODEL['hidden_size'], MODEL['hidden_size'] // 2)
        self.fc3 = nn.Linear(MODEL['hidden_size'] // 2, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Model factory function
def create_model(model_type='lstm', **kwargs):
    """
    Factory function to create models
    
    Args:
        model_type: 'lstm', 'bilstm', 'cnn-lstm', or 'mlp'
        **kwargs: Additional arguments for model initialization
    
    Returns:
        Model instance
    """
    model_type = model_type.lower()
    
    if model_type == 'lstm':
        return LSTMPredictor(**kwargs)
    elif model_type == 'bilstm':
        return BiLSTMPredictor(**kwargs)
    elif model_type in ['cnn-lstm', 'cnnlstm']:
        return CNNLSTMPredictor(**kwargs)
    elif model_type == 'mlp':
        return SOCPredictor()
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from 'lstm', 'bilstm', 'cnn-lstm', 'mlp'")