import torch
import copy

class FLServer:
    def __init__(self, global_model, device='cpu'):
        self.global_model = global_model
        self.device = device

    def aggregate(self, local_weights, local_sizes):
        """
        Federated Averaging (FedAvg).
        w_global = sum( (n_k / n_total) * w_k )
        """
        print("[Server] Aggregating weights...")
        
        total_samples = sum(local_sizes)
        
        # Capture original dtypes
        original_dtypes = {k: v.dtype for k, v in self.global_model.state_dict().items()}
        
        # Initialize global weights accumulator with float zeros
        global_dict = {}
        for key, tensor in self.global_model.state_dict().items():
            global_dict[key] = torch.zeros_like(tensor, dtype=torch.float32)
            
        # Weighted sum
        for weights, size in zip(local_weights, local_sizes):
            weight_factor = size / total_samples
            for key in weights.keys():
                # weights[key] might be Long, multiply by float -> Float
                # ensure we are on the same device
                val = weights[key].to(self.device)
                global_dict[key] += val * weight_factor
                
        # Cast back to original types
        final_dict = {}
        for key, value in global_dict.items():
            if original_dtypes[key] in [torch.int64, torch.long, torch.int32]:
                 final_dict[key] = torch.round(value).long()
            else:
                 final_dict[key] = value
                
        # Update global model
        self.global_model.load_state_dict(final_dict)
        return self.global_model
