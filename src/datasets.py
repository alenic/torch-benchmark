import torch
import torch.utils.data
import os
from PIL import Image
import cv2

def pil_loader(file):
    image = Image.open(file)
    return image.convert("RGB")

def cv2_loader(file):
    image_bgr = cv2.imread(file)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform=None, loader=pil_loader):
        # Get all files recursively
        self.files = []
        for path, curr_dir, files in os.walk(folder):
            for file in files:
                self.files.append(os.path.join(path, file))
        
        self.transform = transform
        self.loader = loader
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file = self.files[index]
        image = self.loader(file)
    
        if self.transform is not None:
            image = self.transform(image)
        
        return image, torch.tensor(0).long()
        