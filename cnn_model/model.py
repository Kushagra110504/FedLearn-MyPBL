import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNIDS(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNNIDS, self).__init__()
        
        # 1st Conv Layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        # 2nd Conv Layer
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        # 3rd Conv Layer
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Pooling & Dropout
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        
        # Global Pooling to handle variable feature size from ChOA
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Fully Connected Layer
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input shape: (Batch, Features) -> (Batch, 1, Features)
        x = x.unsqueeze(1)
        
        # Layer 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        
        # Layer 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        
        # Layer 3
        # Note: If feature dim is small, we might skip pooling or handle it. 
        # But AdaptiveMaxPool at the end handles the spatial dim.
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        # Global Pooling -> (Batch, 128, 1)
        x = self.global_pool(x)
        
        # Flatten -> (Batch, 128)
        x = x.view(x.size(0), -1)
        
        # Output
        x = self.fc(x)
        return x
