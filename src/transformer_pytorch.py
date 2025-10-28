import torch
import torch.nn as nn

class PytorchStateValueModel(nn.Module):
    def __init__(self, input_dim: int = 77, hidden_dim: int = 256, output_dim: int = 1):
        super().__init__()
        # this is a very simple placeholder model.
        # in a real scenario, you'd replicate the deepmind transformer architecture here,
        # potentially loading converted weights.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        # input x is expected to be a flattened board representation
        return self.net(x)