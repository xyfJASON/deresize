import os
import numpy as np
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset


def find_all_images(directory):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    return [str(p) for p in Path(directory).rglob('*') if p.suffix.lower() in image_extensions]


class ImageDataset(Dataset):
    def __init__(self, root: str):
        self.root = os.path.expanduser(root)
        self.image_paths = find_all_images(self.root)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        # crop image to a random aspect ratio
        ar = np.exp(np.clip(np.random.randn(), -1.2, 1.2))
        width, height = image.size
        if width / height > ar:
            new_width = int(height * ar)
            left = (width - new_width) // 2
            image = image.crop((left, 0, left + new_width, height))
        else:
            new_height = int(width / ar)
            top = (height - new_height) // 2
            image = image.crop((0, top, width, top + new_height))
        ar = image.width / image.height
        # resize image to square
        image = image.resize((384, 384), Image.Resampling.LANCZOS)
        return image, ar


def collate_fn(batch):
    images, ars = zip(*batch)
    return list(images), torch.tensor(ars, dtype=torch.float32)
