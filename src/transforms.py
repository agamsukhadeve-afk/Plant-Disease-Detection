import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.config import Config

def get_train_transforms():
    """Returns albumentations transforms for training with data augmentation."""
    return A.Compose([
        A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625), rotate=(-30, 30), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], # ImageNet means
            std=[0.229, 0.224, 0.225],  # ImageNet stds
        ),
        ToTensorV2()
    ])

def get_valid_transforms():
    """Returns albumentations transforms for validation/testing."""
    return A.Compose([
        A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2()
    ])
