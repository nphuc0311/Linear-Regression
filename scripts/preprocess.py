#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params

def preprocess_data(params):
    """Preprocess the advertising dataset."""
    # Load data
    data_path = Path("data/raw/advertising.csv")
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df[["TV", "Radio", "Newspaper"]].values
    y = df["Sales"].values
    
    # Split data into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"]
    )
    
    val_size_adjusted = params["data"]["val_size"] / (1 - params["data"]["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=params["data"]["random_state"]
    )
    
    # # Scale features
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_val = scaler.transform(X_val)
    # X_test = scaler.transform(X_test)
    
    # Create processed data directory
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save processed data
    processed_data = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
    }
    
    with open(processed_dir / "processed_data.pkl", "wb") as f:
        pickle.dump(processed_data, f)
    
    print("Data preprocessing completed successfully.")

if __name__ == "__main__":
    params = load_params()
    preprocess_data(params)