# Hyperparameters and configuration settings

# Federated Learning Parameters
FEDERATED_LEARNING = {
    'rounds': 20,
    'mu': 0.1,  # FedProx regularization parameter
    'client_ratio': 1.0,  # Ratio of clients to select each round
    'num_clients': 4,  # Number of clients for federated learning
}

# Model Architecture Parameters
MODEL = {
    'hidden_size': 64,
    'dropout_rate': 0.2,
    'input_size': 9,  # Based on your features
}

# Training Parameters
TRAINING = {
    'batch_size': 32,
    'local_epochs': 5,
    'learning_rate': 0.001,
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

# Evaluation Parameters
EVALUATION = {
    'metrics': ['mse', 'mae', 'rmse'],
}