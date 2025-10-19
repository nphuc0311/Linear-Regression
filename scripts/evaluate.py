import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import yaml
import torch
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
import json
import csv

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

def evaluate_model(params, data):
    """Evaluate the trained model on test data."""
    # Load test data
    X_test = torch.FloatTensor(data["X_test"])
    y_test = data["y_test"]
    
    # Load model
    model = LinearRegression(params["model"]["input_features"])
    model.load_state_dict(torch.load("models/model.pth"))
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        y_pred = model(X_test).numpy().flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Save metrics
    metrics = {
        "mse": float(mse),
        "r2": float(r2)
    }
    
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(exist_ok=True)
    

    csv_path = metrics_dir / "eval.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["mse", "r2"])
        writer.writerow([float(mse), float(r2)])
    
    print(f"Model Evaluation Results:")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

if __name__ == "__main__":
    params = load_params()
    data = load_data()
    evaluate_model(params, data)