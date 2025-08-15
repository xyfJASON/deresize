import torch.nn as nn
from torch import Tensor


class DeresizerHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.LayerNorm(1024),
            nn.GELU(approximate="tanh"),
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.GELU(approximate="tanh"),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(approximate="tanh"),
            nn.Linear(64, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)
