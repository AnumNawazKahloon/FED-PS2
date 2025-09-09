import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def load_data():
    """
    Load pre-split CSV files and prepare them for federated learning
    """
    config = {
        'features': [
            "DC_Link_Voltage", "vehicle_speed", "Torque_Measured", 
            "MotorTemperature", "Odometer", "cabin_temperature", 
            "outdoor_temperature", "slope", "DC_Link_Current"
        ],
        'target': "SOC",
        'data_path': './data/'
    }
    
    # Load CSV files
    data_path = '/Users/virk/Parma/FED-PS1/data/'
    X_train = pd.read_csv(f"{data_path}X_train.csv")
    X_test = pd.read_csv(f"{data_path}X_test.csv")
    y_train = pd.read_csv(f"{data_path}y_train.csv")
    y_test = pd.read_csv(f"{data_path}y_test.csv")
    
    
    # Ensure we're using the right features and target
    X_train = X_train[config['features']]
    X_test = X_test[config['features']]
    y_train = y_train[config['target']]
    y_test = y_test[config['target']]
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.FloatTensor(y_train.values)
    y_test_tensor = torch.FloatTensor(y_test.values)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    return train_dataset, test_dataset, scaler

def split_data_for_clients(dataset, num_clients):
    """
    Split data among clients for federated learning
    """
    # Calculate data per client
    data_per_client = len(dataset) // num_clients
    client_datasets = []
    
    for i in range(num_clients):
        start_idx = i * data_per_client
        end_idx = start_idx + data_per_client if i < num_clients - 1 else len(dataset)
        
        # Create subset for this client
        client_data = torch.utils.data.Subset(dataset, range(start_idx, end_idx))
        client_datasets.append(client_data)
    
    return client_datasets