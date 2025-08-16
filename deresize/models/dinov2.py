import torch
import torch.nn as nn
from torch import Tensor
from transformers import Dinov2WithRegistersModel

from .head import DeresizerHead


class Dinov2BasedDeresizer(nn.Module):
    def __init__(self, pretrained_model_name_or_path: str):
        super().__init__()
        self.dinov2 = Dinov2WithRegistersModel.from_pretrained(pretrained_model_name_or_path)
        self.head = DeresizerHead(self.dinov2.config.hidden_size)

    def forward(self, *args, **kwargs) -> Tensor:
        with torch.no_grad():
            output = self.dinov2(*args, **kwargs)
            output = output.pooler_output
        output = self.head(output)
        return output
