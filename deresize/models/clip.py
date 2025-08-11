import torch
import torch.nn as nn
from torch import Tensor
from transformers import CLIPVisionConfig, CLIPVisionModel


class DeresizerHead(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, 1024),
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


class CLIPBasedDeresizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.head = DeresizerHead(self.clip.config)

    def forward(self, *args, **kwargs) -> Tensor:
        with torch.no_grad():
            output = self.clip(*args, **kwargs)
            output = output.pooler_output
        output = self.head(output)
        return output
