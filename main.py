from data_loader import load_data, split_data_for_clients
from client import Client
from server import Server
from config import FEDERATED_LEARNING
import numpy as np

def main():
    print("Starting Federated Learning for SOC Prediction...")
    
    # Load data
    train_data, test_data, scaler = load_data()
    print(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples")
    
    # Split data among clients
    client_datasets = split_data_for_clients(train_data, FEDERATED_LEARNING['num_clients'])
    print(f"Split data among {FEDERATED_LEARNING['num_clients']} clients")
    
    # Initialize server
    server = Server(test_data)
    
    # Federated learning rounds
    for round in range(FEDERATED_LEARNING['rounds']):
        print(f"\n--- Round {round+1}/{FEDERATED_LEARNING['rounds']} ---")
        
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
        
        # Evaluate global model
        metrics = server.evaluate()
        print(f"Round {round+1} - Test MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
    
    print("\nFederated learning completed!")

if __name__ == "__main__":
    main()