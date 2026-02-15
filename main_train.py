import os
import sys
import numpy as np
import torch
import pandas as pd
import json

# Adjust path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import load_and_preprocess_data
from utils.data_splitter import split_data_non_iid
from chimp_optimization.choa import ChimpOptimization
from cnn_model.model import CNNIDS
from federated.client import FLClient
from federated.server import FLServer
from evaluation.metrics import evaluate_model, save_confusion_matrix
from torch.utils.data import TensorDataset, DataLoader

# Configuration
DATA_PATH = "data/CICIDS2017.csv"
MAX_SAMPLES = 5000
N_CLIENTS = 3
FL_ROUNDS = 5
LOCAL_EPOCHS = 3
CHOA_AGENTS = 10
CHOA_ITER = 15
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

import joblib

def main():
    print(f"Starting FL-IDS Pipeline on {DEVICE}")
    
    # 1. Load Data
    X, y, classes, scaler = load_and_preprocess_data(DATA_PATH, MAX_SAMPLES)
    
    # Save scaler for frontend
    joblib.dump(scaler, "models/scaler.pkl")
    print("[INFO] Scaler saved to models/scaler.pkl")
    
    n_features_original = X.shape[1]
    n_classes = len(classes)
    
    # Save classes for frontend
    np.save("models/classes.npy", classes)
    
    # 2. Feature Selection (ChOA)
    fs_path = "models/selected_features.npy"
    if os.path.exists(fs_path):
        print("[INFO] Loading selected features from file...")
        selected_mask = np.load(fs_path)
    else:
        print("[INFO] Starting Chimp Optimization for Feature Selection...")
        choa = ChimpOptimization(n_agents=CHOA_AGENTS, max_iter=CHOA_ITER)
        # We need to reshape X for ChOA (it expects 2D, which we have)
        selected_mask, curve = choa.fit(X, y)
        np.save(fs_path, selected_mask)
        np.save("models/choa_convergence.npy", curve)
        print(f"[INFO] ChOA completed. Selected {np.sum(selected_mask)}/{n_features_original} features.")

    # Apply mask
    if np.sum(selected_mask) == 0:
        print("[WARNING] No features selected! Forcing selection of all.")
        selected_mask[:] = 1
        
    X_reduced = X[:, selected_mask == 1]
    n_features_selected = X_reduced.shape[1]
    
    # 3. Split Data for FL
    client_data = split_data_non_iid(X_reduced, y, n_clients=N_CLIENTS)
    
    # Global Test Set (For simplicity, using a portion of the last client or splitting beforehand)
    # Better: Split 70/30 train/test globally first.
    # Re-doing split logic properly:
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, stratify=y, random_state=42)
    
    # Create Test DataLoader
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=256)
    
    # Redistribute Train Data to Clients
    client_data_train = split_data_non_iid(X_train, y_train, n_clients=N_CLIENTS)
    
    # 4. Initialize Global Model
    global_model = CNNIDS(input_dim=n_features_selected, num_classes=n_classes)
    global_model.to(DEVICE)
    
    # Save initial model
    torch.save(global_model.state_dict(), "models/global_init.pth")
    
    # 5. Initialize Server
    server = FLServer(global_model, device=DEVICE)
    
    # Metrics Storage
    history = {'round': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    
    # 6. Federated Learning Loop
    for round_idx in range(FL_ROUNDS):
        print(f"\n--- FL Round {round_idx + 1}/{FL_ROUNDS} ---")
        
        local_weights = []
        local_sizes = []
        
        # Client Training
        for cid in range(N_CLIENTS):
            X_c, y_c = client_data_train[cid]
            client = FLClient(cid, X_c, y_c, device=DEVICE)
            
            # Load global weights to local model
            local_model = CNNIDS(input_dim=n_features_selected, num_classes=n_classes)
            local_model.load_state_dict(global_model.state_dict())
            
            w, size = client.train(local_model, epochs=LOCAL_EPOCHS)
            local_weights.append(w)
            local_sizes.append(size)
        
        # Server Aggregation
        server.aggregate(local_weights, local_sizes)
        
        # Evaluate Global Model
        metrics, _, _ = evaluate_model(global_model, test_loader, device=DEVICE)
        print(f"Global Model Metrics: {metrics}")
        
        # Save metrics
        history['round'].append(round_idx + 1)
        history['accuracy'].append(metrics['Accuracy'])
        history['precision'].append(metrics['Precision'])
        history['recall'].append(metrics['Recall'])
        history['f1'].append(metrics['F1-Score'])
        
        # Save Checkpoint
        torch.save(global_model.state_dict(), f"checkpoints/global_round_{round_idx+1}.pth")
        
        # Save best model
        if round_idx == 0 or metrics['Accuracy'] > max(history['accuracy'][:-1]):
            torch.save(global_model.state_dict(), "models/best_global_model.pth")
            
    # Save Final History
    pd.DataFrame(history).to_csv("evaluation/training_history.csv", index=False)
    
    # Final Evaluation & Confusion Matrix
    print("\n[INFO] Generating Final Evaluation...")
    final_metrics, y_true, y_pred = evaluate_model(global_model, test_loader, device=DEVICE)
    save_confusion_matrix(final_metrics['Confusion Matrix'], classes, "evaluation/confusion_matrix.png")
    
    print("[INFO] Training Complete!")

if __name__ == "__main__":
    main()
