import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def pil_aug(image_size):
    return T.Compose([
        T.Resize((image_size,image_size)),
        T.ToTensor()
    ])

def alb_aug(image_size):
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        ToTensorV2()
    ])

