import torch
import torch.utils.data
import os
from PIL import Image
import cv2
import numpy as np


def pil_loader(file):
    image = Image.open(file)
    return image.convert("RGB")


def cv2_loader(file):
    image_rgb = cv2.imread(file, cv2.IMREAD_COLOR_RGB)
    return image_rgb


class ImageDataset(torch.utils.data.Dataset):
    ALLOWED_EXT = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

    def __init__(self, folder, transform=None, loader=pil_loader, albumentations=False):
        # Get all files recursively
        self.files = []
        for path, curr_dir, files in os.walk(folder):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext.lower() in self.ALLOWED_EXT:
                    self.files.append(os.path.join(path, file))

        self.transform = transform
        self.loader = loader
        self.albumentations = albumentations

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        try:
            image = self.loader(file)
        except:
            if self.albumentations:
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                image = Image.new("RGB", (224, 224))

        if self.transform is not None:
            if self.albumentations:
                image = self.transform(image=image)["image"]
            else:
                image = self.transform(image)

        return image, torch.tensor(0).long()
