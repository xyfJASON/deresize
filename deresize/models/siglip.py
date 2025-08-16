import torch
import torch.nn as nn
from torch import Tensor
from transformers import SiglipVisionModel

from .head import DeresizerHead


class SiglipBasedDeresizer(nn.Module):
    def __init__(self, pretrained_model_name_or_path: str):
        super().__init__()
        self.siglip = SiglipVisionModel.from_pretrained(pretrained_model_name_or_path)
        self.head = DeresizerHead(self.siglip.config.hidden_size)

    def forward(self, *args, **kwargs) -> Tensor:
        with torch.no_grad():
            output = self.siglip(*args, **kwargs)
            output = output.pooler_output
        output = self.head(output)
        return output
