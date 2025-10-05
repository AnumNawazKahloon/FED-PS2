from data_loader import load_data, split_data_for_clients
from client import Client
from server import Server
from config import FEDERATED_LEARNING, TRAINING
from models import create_model
import numpy as np
import torch
import torch.optim as optim
from utils import evaluate_model
import time

def train_centralized(train_data, test_data, model_type='lstm'):
    """Train model in centralized manner"""
    print("\n" + "="*60)
    print(f"Training {model_type.upper()} - CENTRALIZED MODE")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=TRAINING['learning_rate'])
    criterion = torch.nn.MSELoss()
    
    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=TRAINING['batch_size'], 
        shuffle=True
    )
    
    start_time = time.time()
    
    # Training loop
    epochs = FEDERATED_LEARNING['rounds'] * TRAINING['local_epochs']
    print(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / batch_count
            metrics = evaluate_model(model, test_data, device)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, "
                  f"Test MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
    
    training_time = time.time() - start_time
    
    # Final evaluation
    final_metrics = evaluate_model(model, test_data, device)
    final_metrics['training_time'] = training_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model, final_metrics


def train_federated(train_data, test_data, model_type='lstm'):
    """Train model using federated learning with FedProx"""
    print("\n" + "="*60)
    print(f"Training {model_type.upper()} - FEDERATED MODE (FedProx)")
    print("="*60)
    
    # Split data among clients
    client_datasets = split_data_for_clients(train_data, FEDERATED_LEARNING['num_clients'])
    print(f"Split data among {FEDERATED_LEARNING['num_clients']} clients")
    
    # Initialize server - FIXED: Remove model_type parameter
    server = Server(test_data)
    
    start_time = time.time()
    
    # Federated learning rounds
    for round in range(FEDERATED_LEARNING['rounds']):
        print(f"\n--- Round {round+1}/{FEDERATED_LEARNING['rounds']} ---")
        
        # Select clients for this round
        num_selected = int(FEDERATED_LEARNING['client_ratio'] * FEDERATED_LEARNING['num_clients'])
        selected_clients = np.random.choice(
            range(FEDERATED_LEARNING['num_clients']), 
            num_selected, 
            replace=False
        )
        
        # Get current global model state - FIXED: Use global_model directly
        global_state = server.global_model.state_dict()
        
        # Train selected clients
        client_updates = []
        for client_id in selected_clients:
            print(f"Training client {client_id+1}")
            client = Client(client_id, client_datasets[client_id], model_type=model_type)
            client_update = client.train(global_state)
            client_updates.append(client_update)
        
        # Aggregate client updates
        server.aggregate(client_updates)
        
        # Evaluate global model
        if (round + 1) % 10 == 0 or round == FEDERATED_LEARNING['rounds'] - 1:
            metrics = server.evaluate()
            print(f"Round {round+1} - Test MSE: {metrics['mse']:.4f}, "
                  f"MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
    
    training_time = time.time() - start_time
    
    # Final evaluation
    final_metrics = server.evaluate()
    final_metrics['training_time'] = training_time
    
    print(f"Federated training completed in {training_time:.2f} seconds")
    
    # FIXED: Return the global_model directly
    return server.global_model, final_metrics


def print_comparison(results):
    """Print comprehensive comparison of all models"""
    print("\n" + "="*100)
    print("FINAL RESULTS COMPARISON - ALL MODELS")
    print("="*100)
    
    # Print table header
    print(f"\n{'Model':<20} {'Mode':<15} {'MSE':<15} {'MAE':<15} {'RMSE':<15} {'Time (s)':<15}")
    print("-" * 100)
    
    for model_name, modes in results.items():
        for mode, metrics in modes.items():
            print(f"{model_name:<20} {mode:<15} "
                  f"{metrics['mse']:<15.6f} {metrics['mae']:<15.6f} "
                  f"{metrics['rmse']:<15.6f} {metrics['training_time']:<15.2f}")
    
    print("\n" + "="*100)
    
    # Find best models overall
    print("\nBEST MODELS (OVERALL):")
    print("-" * 100)
    
    all_results = []
    for model_name, modes in results.items():
        for mode, metrics in modes.items():
            all_results.append({
                'model': model_name,
                'mode': mode,
                'mse': metrics['mse'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'time': metrics['training_time']
            })
    
    # Best by MSE
    best_mse = min(all_results, key=lambda x: x['mse'])
    print(f"Lowest MSE:  {best_mse['model']:<20} ({best_mse['mode']:<15}) - MSE: {best_mse['mse']:.6f}")
    
    # Best by MAE
    best_mae = min(all_results, key=lambda x: x['mae'])
    print(f"Lowest MAE:  {best_mae['model']:<20} ({best_mae['mode']:<15}) - MAE: {best_mae['mae']:.6f}")
    
    # Best by RMSE
    best_rmse = min(all_results, key=lambda x: x['rmse'])
    print(f"Lowest RMSE: {best_rmse['model']:<20} ({best_rmse['mode']:<15}) - RMSE: {best_rmse['rmse']:.6f}")
    
    # Fastest training
    fastest = min(all_results, key=lambda x: x['time'])
    print(f"Fastest:     {fastest['model']:<20} ({fastest['mode']:<15}) - Time: {fastest['time']:.2f}s")
    
    print("\n" + "="*100)


def main():
    print("\n" + "="*100)
    print("COMPREHENSIVE COMPARISON: LSTM, BiLSTM, CNN-LSTM")
    print("CENTRALIZED vs FEDERATED LEARNING FOR SOC PREDICTION")
    print("="*100)
    
    # Load data
    train_data, test_data, scaler = load_data()
    print(f"\nLoaded {len(train_data)} training samples and {len(test_data)} test samples")
    
    # Dictionary to store all results
    results = {}
    
    # Dictionary to store trained models for visualization
    centralized_models = {}
    
    # List of models to train
    models_to_train = ['lstm', 'bilstm', 'cnn-lstm']
    
    for model_type in models_to_train:
        print("\n" + "#"*100)
        print(f"# {model_type.upper()} MODEL - CENTRALIZED & FEDERATED")
        print("#"*100)
        
        # Train centralized
        try:
            centralized_model, centralized_metrics = train_centralized(
                train_data, test_data, model_type=model_type
            )
            print(f"\n✓ {model_type.upper()} Centralized training completed!")
            print(f"  MSE: {centralized_metrics['mse']:.6f}, MAE: {centralized_metrics['mae']:.6f}, "
                  f"RMSE: {centralized_metrics['rmse']:.6f}, Time: {centralized_metrics['training_time']:.2f}s")
            
            # Store model for visualization
            centralized_models[model_type.upper()] = centralized_model
            
        except Exception as e:
            print(f"\n✗ {model_type.upper()} Centralized training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            centralized_metrics = {'mse': float('inf'), 'mae': float('inf'), 
                                  'rmse': float('inf'), 'training_time': 0}
        
        # Train federated
        try:
            federated_model, federated_metrics = train_federated(
                train_data, test_data, model_type=model_type
            )
            print(f"\n✓ {model_type.upper()} Federated training completed!")
            print(f"  MSE: {federated_metrics['mse']:.6f}, MAE: {federated_metrics['mae']:.6f}, "
                  f"RMSE: {federated_metrics['rmse']:.6f}, Time: {federated_metrics['training_time']:.2f}s")
        except Exception as e:
            print(f"\n✗ {model_type.upper()} Federated training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            federated_metrics = {'mse': float('inf'), 'mae': float('inf'), 
                                'rmse': float('inf'), 'training_time': 0}
        
        # Store results
        results[model_type.upper()] = {
            'Centralized': centralized_metrics,
            'Federated': federated_metrics
        }
    
    # Print comprehensive comparison
    print_comparison(results)
    
    # Create visualizations
    try:
        from visualization import create_visualization_report
        print("\n" + "="*100)
        print("GENERATING VISUALIZATIONS")
        print("="*100)
        create_visualization_report(results, centralized_models, test_data, output_dir='./plots/')
    except ImportError:
        print("\n⚠ Visualization module not found. Skipping visualization generation.")
        print("  Make sure visualization.py is in the same directory.")
    except Exception as e:
        print(f"\n⚠ Visualization generation failed: {str(e)}")
        print("  Continuing without visualizations...")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*100)
    print("✓ All training and evaluation completed!")
    print("\nModel Architectures:")
    print("  - LSTM: Unidirectional LSTM with 2 layers")
    print("  - BiLSTM: Bidirectional LSTM with 2 layers")
    print("  - CNN-LSTM: 2 Conv1D layers + LSTM for feature extraction")
    print("\nFederated Learning:")
    print("  - Algorithm: FedProx (Federated Averaging with proximal term)")
    print(f"  - Clients: {FEDERATED_LEARNING['num_clients']}")
    print(f"  - Rounds: {FEDERATED_LEARNING['rounds']}")
    print(f"  - Client selection ratio: {FEDERATED_LEARNING['client_ratio']}")
    print(f"  - Mu (proximal term): {FEDERATED_LEARNING['mu']}")
    print("\nVisualization:")
    print("  - Check ./plots/ directory for all charts and analysis")
    print("="*100 + "\n")

if __name__ == "__main__":
    main()