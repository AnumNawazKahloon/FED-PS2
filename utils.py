import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, test_data, device):
    """
    Evaluate model on test data
    """
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X).squeeze()
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }