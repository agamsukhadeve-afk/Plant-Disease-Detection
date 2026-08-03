import torch
import cv2
import numpy as np
import os
from src.model import PlantDiseaseModel
from src.config import Config
from src.transforms import get_valid_transforms

class PlantDiseasePredictor:
    def __init__(self, model_path, class_names):
        self.device = Config.DEVICE
        self.class_names = class_names
        
        # Initialize and load model
        self.model = PlantDiseaseModel(num_classes=len(class_names), freeze_backbone=False)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint) # fallback
        else:
            print(f"Warning: Model path {model_path} does not exist. Using untrained weights.")
            
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = get_valid_transforms()
        
    def predict(self, image):
        """
        Predicts the disease from an OpenCV image (RGB).
        
        Args:
            image: numpy array (RGB)
            
        Returns:
            dict: top_predictions containing class names and probabilities
        """
        # Apply transforms
        tensor_img = self.transform(image=image)['image'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor_img)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        # Get top 3 predictions
        top_prob, top_catid = torch.topk(probabilities, 3)
        
        results = []
        for i in range(top_prob.size(0)):
            results.append({
                'class': self.class_names[top_catid[i]],
                'probability': top_prob[i].item()
            })
            
        return results
