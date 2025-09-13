from data_loader import load_data, split_data_for_clients
from client import Client
from server import Server
from config import FEDERATED_LEARNING
import numpy as np
import torch
import torch.nn as nn

def train_global_model(train_data, test_data, model, epochs=50):
    """Train a centralized global model for comparison"""
    print("\nTraining global (centralized) model...")
    
    # Create data loader for training
    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=FEDERATED_LEARNING['batch_size'], 
        shuffle=True
    )
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=FEDERATED_LEARNING['learning_rate'])
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Global Model Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
    
    # Evaluate global model
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
    total_mse, total_mae = 0, 0
    
    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)
            total_mse += nn.functional.mse_loss(output, y, reduction='sum').item()
            total_mae += nn.functional.l1_loss(output, y, reduction='sum').item()
    
    mse = total_mse / len(test_data)
    mae = total_mae / len(test_data)
    rmse = np.sqrt(mse)
    
    return {'mse': mse, 'mae': mae, 'rmse': rmse}

def main():
    print("Starting Federated Learning for SOC Prediction...")
    
    # Load data
    train_data, test_data, scaler = load_data()
    print(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples")
    
    # Initialize server (which contains the global model architecture)
    server = Server(test_data)
    
    # Train and evaluate a centralized global model
    global_model = server.create_model()  # Assuming Server has a method to create a new model instance
    global_metrics = train_global_model(train_data, test_data, global_model)
    print(f"\nGlobal Model Results - MSE: {global_metrics['mse']:.4f}, MAE: {global_metrics['mae']:.4f}, RMSE: {global_metrics['rmse']:.4f}")
    
    # Split data among clients for federated learning
    client_datasets = split_data_for_clients(train_data, FEDERATED_LEARNING['num_clients'])
    print(f"Split data among {FEDERATED_LEARNING['num_clients']} clients")
    
    # Federated learning rounds
    federated_metrics_history = []
    
    for round in range(FEDERATED_LEARNING['rounds']):
        print(f"\n--- Federated Learning Round {round+1}/{FEDERATED_LEARNING['rounds']} ---")
        
        # Select clients for this round
        num_selected = int(FEDERATED_LEARNING['client_ratio'] * FEDERATED_LEARNING['num_clients'])
        selected_clients = np.random.choice(range(FEDERATED_LEARNING['num_clients']), 
                                           num_selected, replace=False)
        
        # Get current global model state
        global_state = server.global_model.state_dict()
        
        # Train selected clients
        client_updates = []
        for client_id in selected_clients:
            print(f"Training client {client_id+1}")
            client = Client(client_id, client_datasets[client_id])
            client_update = client.train(global_state)
            client_updates.append(client_update)
        
        # Aggregate client updates
        server.aggregate(client_updates)
        
        # Evaluate federated model
        metrics = server.evaluate()
        federated_metrics_history.append(metrics)
        print(f"Federated Model Round {round+1} - Test MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
    
    # Print final comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON RESULTS")
    print("="*60)
    print(f"Global (Centralized) Model:")
    print(f"  MSE: {global_metrics['mse']:.4f}, MAE: {global_metrics['mae']:.4f}, RMSE: {global_metrics['rmse']:.4f}")
    
    final_federated_metrics = federated_metrics_history[-1]
    print(f"Federated Model (Final):")
    print(f"  MSE: {final_federated_metrics['mse']:.4f}, MAE: {final_federated_metrics['mae']:.4f}, RMSE: {final_federated_metrics['rmse']:.4f}")
    
    # Calculate improvement/decline
    mse_diff = global_metrics['mse'] - final_federated_metrics['mse']
    mae_diff = global_metrics['mae'] - final_federated_metrics['mae']
    rmse_diff = global_metrics['rmse'] - final_federated_metrics['rmse']
    
    print(f"\nPerformance Difference (Global - Federated):")
    print(f"  MSE: {mse_diff:+.4f} ({'Better' if mse_diff > 0 else 'Worse'})")
    print(f"  MAE: {mae_diff:+.4f} ({'Better' if mae_diff > 0 else 'Worse'})")
    print(f"  RMSE: {rmse_diff:+.4f} ({'Better' if rmse_diff > 0 else 'Worse'})")
    
    print("\nFederated learning completed!")

if __name__ == "__main__":
    main()