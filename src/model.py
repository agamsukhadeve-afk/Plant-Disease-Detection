import torch
import torch.nn as nn
import torchvision.models as models
from src.config import Config

class PlantDiseaseModel(nn.Module):
    """
    EfficientNet-B0 Model for Plant Disease Detection.
    Uses pretrained weights and modifies the classifier head.
    """
    
    def __init__(self, num_classes=Config.NUM_CLASSES, freeze_backbone=True):
        super(PlantDiseaseModel, self).__init__()
        
        # Load pretrained EfficientNet-B0
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Freeze backbone parameters if specified
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
                
        # Modify the classifier head
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)
        
    def unfreeze_backbone(self):
        """Unfreezes the backbone for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
