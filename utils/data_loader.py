import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import os

def load_and_preprocess_data(filepath, max_samples=150000):
    """
    Loads CICIDS2017 dataset, cleans, encodes, and normalizes it.
    If file not found, generates synthetic data for demonstration.
    """
    if not os.path.exists(filepath):
        print(f"[WARNING] Dataset not found at {filepath}. Generating SYNTHETIC data for demonstration.")
        return generate_synthetic_data(max_samples)

    print(f"[INFO] Loading dataset from {filepath}...")
    # Load only a subset to save memory/time
    df = pd.read_csv(filepath, nrows=max_samples)

    # 1. Clean Data
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Replace Inf with NaN and drop NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 2. Encode Labels
    # Assuming the target column is 'Label' (standard in CICIDS2017)
    if 'Label' not in df.columns:
        # Fallback if column name differs
        possible_labels = [col for col in df.columns if 'Label' in col or 'class' in col.lower()]
        if possible_labels:
            target_col = possible_labels[0]
        else:
            raise ValueError("Could not find 'Label' column in dataset.")
    else:
        target_col = 'Label'

    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col].astype(str))
    
    # 3. Separate Features and Target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Remove non-numeric columns if any remained
    X = X.select_dtypes(include=[np.number])

    # 4. Normalize Features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"[INFO] Data loaded. Shape: {X_scaled.shape}")
    return X_scaled, y.values, le.classes_, scaler

def generate_synthetic_data(n_samples=1000):
    """Generates synthetic data for testing pipeline."""
    n_features = 78 # Standard CICIDS2017 feature count
    n_classes = 5
    
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    classes = [f"Class_{i}" for i in range(n_classes)]
    
    # Needs to return a dummy scaler too
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X)
    
    return X, y, classes, scaler

