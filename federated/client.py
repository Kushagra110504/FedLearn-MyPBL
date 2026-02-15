import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy

class FLClient:
    def __init__(self, client_id, X, y, device='cpu'):
        self.client_id = client_id
        self.device = device
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.dataset = TensorDataset(self.X, self.y)
        self.dataloader = DataLoader(self.dataset, batch_size=256, shuffle=True)
        self.sample_count = len(X)

    def train(self, model, epochs=3):
        """Train local model and return weights."""
        model.to(self.device)
        model.train()
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        print(f"[Client {self.client_id}] Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            avg_loss = total_loss / len(self.dataloader)
            acc = 100 * correct / total
            print(f"  [Client {self.client_id}] Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")
            
        return model.state_dict(), self.sample_count
