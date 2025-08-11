import torch
from safetensors.torch import load_file


def get_model_and_processor(model_name: str):

    if model_name == "siglip":
        from transformers import SiglipImageProcessor
        from .models.siglip import SiglipBasedDeresizer
        processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        model = SiglipBasedDeresizer()

    elif model_name == "clip":
        from transformers import CLIPImageProcessor
        from .models.clip import CLIPBasedDeresizer
        processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", size=224)
        model = CLIPBasedDeresizer()

    else:
        raise ValueError(f"Model {model_name} is not supported.")

    return model, processor


def load_weights(weights_path: str):
    if weights_path.endswith(".safetensors"):
        weights = load_file(weights_path, device="cpu")
    else:
        weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "model" in weights:
        weights = weights["model"]
    return weights
