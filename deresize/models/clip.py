import torch
import torch.nn as nn
from torch import Tensor
from transformers import CLIPVisionModel

from .head import DeresizerHead


class CLIPBasedDeresizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.head = DeresizerHead(self.clip.config.hidden_size)

    def forward(self, *args, **kwargs) -> Tensor:
        with torch.no_grad():
            output = self.clip(*args, **kwargs)
            output = output.pooler_output
        output = self.head(output)
        return output
