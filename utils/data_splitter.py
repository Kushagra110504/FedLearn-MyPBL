import numpy as np

def split_data_non_iid(X, y, n_clients=3, samples_per_client=50000):
    """
    Splits data into non-IID clients.
    Sorts data by label and divides into shards.
    """
    print(f"[INFO] Splitting data into {n_clients} non-IID clients...")
    
    data_len = len(y)
    idxs = np.arange(data_len)
    
    # Sort by label
    idxs_labels = np.vstack((idxs, y))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    
    # Divide into shards
    # For simplicity, we just chunk the sorted data
    # This creates a pathological non-IID split (highly skewed)
    
    client_data = []
    
    # Logic to ensure we don't exceed samples_per_client
    # And we handle cases where total data < desired total
    
    chunk_size = min(int(data_len / n_clients), samples_per_client)
    
    for i in range(n_clients):
        start = i * chunk_size
        end = start + chunk_size
        if start >= data_len:
            break
            
        client_idxs = idxs[start:end]
        X_client = X[client_idxs]
        y_client = y[client_idxs]
        
        client_data.append((X_client, y_client))
        print(f"[INFO] Client {i}: {len(X_client)} samples")
        
    return client_data
