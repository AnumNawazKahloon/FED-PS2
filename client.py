import torch
import torch.optim as optim
from models import SOCPredictor
from config import TRAINING, FEDERATED_LEARNING

class Client:
    def __init__(self, client_id, data):
        self.id = client_id
        self.data = data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SOCPredictor().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=TRAINING['learning_rate'])
        self.criterion = torch.nn.MSELoss()
    
    def train(self, global_model_state):
        """
        Train client model with FedProx regularization
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
                        self.model.parameters(), global_model_state.values()
                    ):
                        proximal_term += (local_param - global_param).norm(2)
                    loss += (FEDERATED_LEARNING['mu'] / 2) * proximal_term
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
        
        return self.model.state_dict()