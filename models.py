import torch.nn as nn
import torch.nn.functional as F
from config import MODEL

class SOCPredictor(nn.Module):
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