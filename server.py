import torch
import numpy as np
from models import SOCPredictor
from utils import evaluate_model
from config import FEDERATED_LEARNING

class Server:
    def __init__(self, test_data):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = SOCPredictor().to(self.device)
        self.test_data = test_data
    
    def aggregate(self, client_updates):
        """
        Aggregate client updates using Federated Averaging
        """
        global_dict = self.global_model.state_dict()
        
        # Initialize averaged parameters
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
        
        # Sum all client updates
        for client_dict in client_updates:
            for key in global_dict.keys():
                global_dict[key] += client_dict[key]
        
        # Average the updates
        for key in global_dict.keys():
            global_dict[key] = torch.div(global_dict[key], len(client_updates))
        
        # Update global model
        self.global_model.load_state_dict(global_dict)
        
        return global_dict
    
    def evaluate(self):
        """
        Evaluate the global model on test data
        """
        return evaluate_model(self.global_model, self.test_data, self.device)