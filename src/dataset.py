import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from src.transforms import get_train_transforms, get_valid_transforms
from src.config import Config

class PlantDiseaseDataset(Dataset):
    """Custom PyTorch Dataset for Plant Disease images."""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.JPEG')):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls_name])
                    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Read image using OpenCV
        image = cv2.imread(img_path)
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, torch.tensor(label, dtype=torch.long)

def get_dataloaders():
    """Returns train, validation, and test dataloaders."""
    
    train_dataset = PlantDiseaseDataset(Config.TRAIN_DIR, transform=get_train_transforms())
    val_dataset = PlantDiseaseDataset(Config.VAL_DIR, transform=get_valid_transforms())
    test_dataset = PlantDiseaseDataset(Config.TEST_DIR, transform=get_valid_transforms())
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.classes
