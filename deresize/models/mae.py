import torch
import torch.nn as nn
from torch import Tensor
from transformers import ViTMAEModel

from .head import DeresizerHead


class MAEBasedDeresizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = ViTMAEModel.from_pretrained("facebook/vit-mae-large", mask_ratio=0.)
        self.head = DeresizerHead(self.mae.config.hidden_size)

    def forward(self, *args, **kwargs) -> Tensor:
        with torch.no_grad():
            output = self.mae(*args, **kwargs)
            output = output.last_hidden_state[:, 0, :]
        output = self.head(output)
        return output
