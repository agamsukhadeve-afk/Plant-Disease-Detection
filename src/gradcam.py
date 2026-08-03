import cv2
import numpy as np
import torch
import torch.nn.functional as F
from src.config import Config
from src.transforms import get_valid_transforms

class GradCAMVisualizer:
    def __init__(self, model):
        self.model = model
        # Target layer for EfficientNet-B0
        self.target_layer = model.model.features[-1]
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        self.transform = get_valid_transforms()

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, rgb_img):
        """
        Generates a Grad-CAM heatmap overlay for the given RGB image.
        """
        self.model.eval()
        # Enable gradients for the target layer
        for param in self.target_layer.parameters():
            param.requires_grad = True
            
        tensor_img = self.transform(image=rgb_img)['image'].unsqueeze(0).to(Config.DEVICE)
        tensor_img.requires_grad = True
        
        # Forward pass
        output = self.model(tensor_img)
        target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Global average pooling on gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1).squeeze().detach().cpu().numpy()
        
        # ReLU to keep only positive influence
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        # Resize to original image size
        cam = cv2.resize(cam, (rgb_img.shape[1], rgb_img.shape[0]))
        
        # Convert to heatmap (JET colormap)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Superimpose
        rgb_img_float = np.float32(rgb_img) / 255
        heatmap_float = np.float32(heatmap) / 255
        
        visualization = heatmap_float * 0.5 + rgb_img_float * 0.5
        visualization = np.clip(visualization, 0, 1)
        
        return np.uint8(255 * visualization)
