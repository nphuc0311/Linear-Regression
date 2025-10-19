import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import csv
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score

# Import our model
from src.models.linear import LinearRegression

def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params

def load_data():
    """Load processed data."""
    with open("data/processed/processed_data.pkl", "rb") as f:
        data = pickle.load(f)
    return data

def train_model(params, data):
    """Train the linear regression model."""
    # Convert data to PyTorch tensors
    X_train = torch.FloatTensor(data["X_train"])
    y_train = torch.FloatTensor(data["y_train"]).reshape(-1, 1)
    X_val = torch.FloatTensor(data["X_val"])
    y_val = torch.FloatTensor(data["y_val"]).reshape(-1, 1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=params["model"]["batch_size"],
        shuffle=True
    )
    
    # Initialize model, loss function, and optimizer
    model = LinearRegression(params["model"]["input_features"])
    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=params["model"]["learning_rate"]
    # )

    optimizer = torch.optim.Adam(model.parameters(), lr=params["model"]["learning_rate"])
    
    # Containers for per-epoch metrics
    epochs = []
    train_losses = []
    val_losses = []
    val_mses = []
    val_r2s = []
    
    # Training loop
    for epoch in range(params["model"]["num_epochs"]):
        model.train()
        running_loss = 0.0
        n_batches = 0
        
        for X_batch, y_batch in train_loader:
            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            n_batches += 1
        
        # average train loss for epoch
        avg_train_loss = running_loss / max(1, n_batches)
        
        # Validation: compute loss and also MSE/R2 on whole validation set
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
        
        # compute validation mse and r2 using numpy arrays
        y_val_np = y_val.numpy().flatten()
        val_pred_np = val_pred.numpy().flatten()
        val_mse = float(mean_squared_error(y_val_np, val_pred_np))
        val_r2 = float(r2_score(y_val_np, val_pred_np))
        
        # store metrics
        epoch_idx = epoch + 1
        epochs.append(epoch_idx)
        train_losses.append(float(avg_train_loss))
        val_losses.append(float(val_loss))
        val_mses.append(val_mse)
        val_r2s.append(val_r2)
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{params['model']['num_epochs']}], "
                  f"Train Loss: {loss.item():.4f}, "
                  f"Val Loss: {val_loss:.4f}")
            
    # Save metrics directory and files (JSON + CSV) for DVC plots
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(exist_ok=True)

    csv_path = metrics_dir / "train.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["epoch", "train_loss", "val_loss", "val_mse", "val_r2"]
        writer.writerow(header)
        for i in range(len(epochs)):
            writer.writerow([
                epochs[i],
                train_losses[i],
                val_losses[i],
                val_mses[i],
                val_r2s[i]
            ])
    
    # Save the model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pth")
    print("Model training completed and saved successfully.")

if __name__ == "__main__":
    params = load_params()
    data = load_data()
    train_model(params, data)