from PIL import Image

import torch
from huggingface_hub import hf_hub_download

from .helper import get_model_and_processor, load_weights


class Deresizer:

    REPO = "xyfJASON/deresize"
    HEAD_WEIGHTS = {
        "siglip": "siglip-head.safetensors",
        "clip": "clip-head.safetensors",
    }

    def __init__(self, model_name: str, device: str | torch.device = "cuda", local_head_weights_file: str = None):
        self.model_name = model_name
        self.device = device

        self.model, self.processor = get_model_and_processor(self.model_name)
        head_weights_file = local_head_weights_file or hf_hub_download(self.REPO, self.HEAD_WEIGHTS[self.model_name])
        head_weights = load_weights(head_weights_file)
        self.model.head.load_state_dict(head_weights)

        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> Image.Image:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
        ar = self.model(**inputs).squeeze().exp()
        w, h = image.size
        if ar * h <= w:
            new_w, new_h = int(ar * h), h
        else:
            new_w, new_h = w, int(w / ar)
        resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return resized_image
