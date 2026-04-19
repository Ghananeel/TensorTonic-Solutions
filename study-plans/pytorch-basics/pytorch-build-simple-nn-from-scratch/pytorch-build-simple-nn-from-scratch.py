import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    """
    Returns: two-layer MLP output (linear -> ReLU -> linear)
    """

    def __init__(self, in_features, hidden_size, out_features):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_features)
        ])
        
    def forward(self, x):
        x = torch.Tensor(x)
        for layer in self.layers:
            x = layer(x)
        return x
