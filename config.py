# Hyperparameters and configuration settings

# Federated Learning Parameters
FEDERATED_LEARNING = {
    'rounds': 50,
    'mu': 0.01,  # FedProx regularization parameter
    'client_ratio': 0.75,  # Ratio of clients to select each round
    'num_clients': 4,  # Number of clients for federated learning
}

# Model Architecture Parameters
MODEL = {
    'hidden_size': 128,
    'dropout_rate': 0.3,
    'input_size': 9,  # Based on your features
    'lstm_hidden_size': 128,  # Hidden size for LSTM models
    'lstm_num_layers': 2,  # Number of LSTM layers
    'cnn_channels': [64, 128],  # CNN channels for CNN-LSTM
}

# Training Parameters
TRAINING = {
    'batch_size': 64,
    'local_epochs': 2,
    'learning_rate': 0.001,  # Lower learning rate for LSTM models
    'lr_decay': 0.995,
    'weight_decay': 1e-5,  
}

# Data Parameters
DATA = {
    'features': [
        "DC_Link_Voltage", "vehicle_speed", "Torque_Measured", 
        "MotorTemperature", "Odometer", "cabin_temperature", 
        "outdoor_temperature", "slope", "DC_Link_Current"
    ],
    'target': "SOC",
    'data_path': './data/',  # Path to your CSV files
}


EVALUATION = {
    'metrics': ['mse', 'mae', 'rmse'],
}