import os
import numpy as np
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


class CelebrityFacesDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform

        valid_exts = (".jpg", ".jpeg", ".png", ".webp")
        self.image_files = []

        for root, _, files in os.walk(image_folder):
            for f in files:
                if f.lower().endswith(valid_exts):
                    self.image_files.append(os.path.join(root, f))

        self.image_files.sort()
        print("Images found:", len(self.image_files))
        print("First few image paths:", self.image_files[:5])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert("L")
        except (UnidentifiedImageError, OSError):
            return self.__getitem__((idx + 1) % len(self.image_files))

        if self.transform:
            image = self.transform(image)

        return image, 0


class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        pil_img, label = self.subset[idx]
        if self.transform:
            pil_img = self.transform(pil_img)
        return pil_img, label

    def to_numpy(self):
        """Convert all images in the subset to a NumPy array."""
        images = []
        for idx in range(len(self.subset)):
            img, _ = self[idx]
            img_array = np.array(img, dtype=np.float32)
            images.append(img_array)
        return np.stack(images)  # (num_images, height, width)


class CelebrityFacesDatasetCached(Dataset):
    def __init__(self, image_folder, transform=None):
        self.transform = transform
        self.images = []

        valid_exts = (".jpg", ".jpeg", ".png", ".webp")
        for root, _, files in os.walk(image_folder):
            for f in files:
                if f.lower().endswith(valid_exts):
                    path = os.path.join(root, f)
                    try:
                        img = Image.open(path).convert("L")
                        self.images.append(img)
                    except UnidentifiedImageError, OSError:
                        # Skip corrupted images
                        continue

        print("Cached images:", len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, 0

    def to_numpy(self):
        """Convert cached images to a NumPy array."""
        images = []
        for img in self.images:
            img_array = np.array(img, dtype=np.float32)
            images.append(img_array)
        return np.stack(images)  # (num_images, height, width)
