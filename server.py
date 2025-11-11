import torch
import numpy as np
from models import create_model
from utils import evaluate_model
from config import FEDERATED_LEARNING

class Server:
    def __init__(self, test_data, model_type='lstm'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type.lower()
        self.test_data = test_data
        
        # Initialize global model based on type
        self.global_model = create_model(model_type).to(self.device)
    
    def aggregate(self, client_updates):
        """
        Aggregate client updates using Federated Averaging (FedAvg)
        Works for all PyTorch models: LSTM, BiLSTM, CNN-LSTM, MLP
        """
        global_dict = self.global_model.state_dict()
        
        # Initialize averaged parameters
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
        
        # Sum all client updates
        for client_dict in client_updates:
            for key in global_dict.keys():
                global_dict[key] += client_dict[key].to(self.device)
        
        # Average the updates
        for key in global_dict.keys():
            global_dict[key] = torch.div(global_dict[key], len(client_updates))
        
        # Update global model
        self.global_model.load_state_dict(global_dict)
        
        return global_dict
    
    def evaluate(self):
        """
        Evaluation of global model on test data
        """
        return evaluate_model(self.global_model, self.test_data, self.device)
    
    def get_global_state(self):
        """Get current global model state"""
        return self.global_model.state_dict()
