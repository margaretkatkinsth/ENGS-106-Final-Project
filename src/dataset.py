import os
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset


class CelebrityFacesDataset(Dataset):
    def __init__(self, image_folder, img_mode="L", transform=None):
        self.mode = img_mode
        self.transform = transform

        valid_exts = (".jpg", ".jpeg", ".png", ".webp")

        self.image_files = [
            os.path.join(root, f)
            for root, _, files in os.walk(image_folder)
            for f in files
            if f.lower().endswith(valid_exts)
        ]

        print("Images found:", len(self.image_files))
        print("First few image paths:", self.image_files[:5])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = self.image_files[idx]

        try:
            with Image.open(path) as img:
                img = img.convert(self.mode)
        except (UnidentifiedImageError, OSError):
            print("Skipping:", path)
            raise ValueError

        if self.transform:
            img = self.transform(img)

        return img, 0

    def to_numpy(self):
        images = []

        for idx in range(len(self)):
            img, _ = self[idx]

            if isinstance(img, torch.Tensor):
                img_array = img.squeeze().cpu().numpy()
            else:
                img_array = np.array(img, dtype=np.float32)

            images.append(img_array.astype(np.float32))

        return np.stack(images)


class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, _ = self.subset[idx]

        if self.transform:
            img = self.transform(img)

        return img, 0

    def to_numpy(self):
        images = []

        for idx in range(len(self)):
            img, _ = self[idx]

            if isinstance(img, torch.Tensor):
                img_array = img.squeeze().cpu().numpy()
            else:
                img_array = np.array(img, dtype=np.float32)

            images.append(img_array.astype(np.float32))

        return np.stack(images)
