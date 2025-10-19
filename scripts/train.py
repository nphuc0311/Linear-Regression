import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import csv
import yaml
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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


def plot_metrics(epochs, train_losses, val_losses, val_mses, val_r2s, output_dir="plots"):
    """Plot training metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title("Training & Validation Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_mses, color="orange", label="Validation MSE")
    plt.title("Validation MSE per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "val_mse.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_r2s, color="green", label="Validation R²")
    plt.title("Validation R² per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("R² Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "val_r2.png")
    plt.close()

    print(f"Metrics plots saved in {output_dir}/")


def train_model(params, data):
    """Train the linear regression model."""
    X_train = torch.FloatTensor(data["X_train"])
    y_train = torch.FloatTensor(data["y_train"]).reshape(-1, 1)
    X_val = torch.FloatTensor(data["X_val"])
    y_val = torch.FloatTensor(data["y_val"]).reshape(-1, 1)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=params["model"]["batch_size"],
        shuffle=True
    )
    
    model = LinearRegression(params["model"]["input_features"])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["model"]["learning_rate"])
    
    epochs, train_losses, val_losses, val_mses, val_r2s = [], [], [], [], []
    
    for epoch in range(params["model"]["num_epochs"]):
        model.train()
        running_loss = 0.0
        n_batches = 0
        
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = running_loss / max(1, n_batches)
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
        
        y_val_np = y_val.numpy().flatten()
        val_pred_np = val_pred.numpy().flatten()
        val_mse = float(mean_squared_error(y_val_np, val_pred_np))
        val_r2 = float(r2_score(y_val_np, val_pred_np))
        
        epoch_idx = epoch + 1
        epochs.append(epoch_idx)
        train_losses.append(float(avg_train_loss))
        val_losses.append(float(val_loss))
        val_mses.append(val_mse)
        val_r2s.append(val_r2)
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{params['model']['num_epochs']}], "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}")
            
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
    
    plot_metrics(epochs, train_losses, val_losses, val_mses, val_r2s)

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pth")
    print("Model training completed and saved successfully.")


if __name__ == "__main__":
    params = load_params()
    torch.manual_seed(params["model"]["random_seed"])
    data = load_data()
    train_model(params, data)
