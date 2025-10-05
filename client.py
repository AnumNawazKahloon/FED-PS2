import torch
import torch.optim as optim
from models import create_model
from config import TRAINING, FEDERATED_LEARNING
import numpy as np

class Client:
    def __init__(self, client_id, data, model_type='lstm'):
        self.id = client_id
        self.data = data
        self.model_type = model_type.lower()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model based on type
        self.model = create_model(model_type).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=TRAINING['learning_rate'])
        self.criterion = torch.nn.MSELoss()
    
    def train(self, global_model_state):
        """
        Train client model with FedProx regularization
        Supports LSTM, BiLSTM, CNN-LSTM models
        """
        # Load global model parameters
        self.model.load_state_dict(global_model_state)
        
        # Set model to training mode
        self.model.train()
        
        # Create data loader
        train_loader = torch.utils.data.DataLoader(
            self.data, 
            batch_size=TRAINING['batch_size'], 
            shuffle=True
        )
        
        # Training loop
        for epoch in range(TRAINING['local_epochs']):
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X).squeeze()
                
                # Calculate loss with FedProx regularization
                loss = self.criterion(outputs, batch_y)
                
                # Add FedProx regularization term
                if FEDERATED_LEARNING['mu'] > 0:
                    proximal_term = 0.0
                    for local_param, global_param in zip(
                        self.model.parameters(), 
                        [p for p in global_model_state.values()]
                    ):
                        if isinstance(global_param, torch.Tensor):
                            proximal_term += (local_param - global_param.to(self.device)).norm(2)
                    loss += (FEDERATED_LEARNING['mu'] / 2) * proximal_term
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
        
        return self.model.state_dict()