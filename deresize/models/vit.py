import torch
import torch.nn as nn
from torch import Tensor
from transformers import ViTModel

from .head import DeresizerHead


class ViTBasedDeresizer(nn.Module):
    def __init__(self, pretrained_model_name_or_path: str):
        super().__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_name_or_path, add_pooling_layer=False)
        self.head = DeresizerHead(self.vit.config.hidden_size)

    def forward(self, *args, **kwargs) -> Tensor:
        with torch.no_grad():
            output = self.vit(*args, **kwargs)
            output = output.last_hidden_state[:, 0, :]
        output = self.head(output)
        return output
