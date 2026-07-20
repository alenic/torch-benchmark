import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def pil_aug(image_size, eval=False):
    if eval:
        return T.Compose([
            T.Resize((image_size,image_size)),
            T.ToTensor(),
            T.Normalize(mean=((0.485, 0.456, 0.406)), std=(0.229, 0.224, 0.225)),
        ])
    else:
        return T.Compose([
            T.RandomResizedCrop((image_size, image_size), scale=(0.8, 1)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.1, 0.1, 0.1, 0.1),
            T.ToTensor(),
            T.Normalize(mean=((0.485, 0.456, 0.406)), std=(0.229, 0.224, 0.225)),
        ])

def alb_aug(image_size, eval=False):
    if eval:
        return A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=((0.485, 0.456, 0.406)), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    else:
        return A.Compose([
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8,1)),
            A.HorizontalFlip(),
            A.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1),
            A.Normalize(mean=((0.485, 0.456, 0.406)), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

