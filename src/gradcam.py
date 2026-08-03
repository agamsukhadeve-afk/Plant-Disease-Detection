import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.config import Config
from src.transforms import get_valid_transforms

class GradCAMVisualizer:
    def __init__(self, model):
        self.model = model
        # Target layer for EfficientNet-B0
        self.target_layers = [model.model.features[-1]]
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)
        self.transform = get_valid_transforms()

    def generate_heatmap(self, rgb_img):
        """
        Generates a Grad-CAM heatmap overlay for the given RGB image.
        """
        tensor_img = self.transform(image=rgb_img)['image'].unsqueeze(0).to(Config.DEVICE)
        
        # The GradCAM library expects gradients to be enabled for the target layers
        # EfficientNet-b0 features requires grad should be True for GradCAM to work.
        for param in self.model.model.features[-1].parameters():
            param.requires_grad = True

        # You can also pass target_category to generate cam for a specific class
        grayscale_cam = self.cam(input_tensor=tensor_img, targets=None)[0, :]
        
        # Normalize original image to [0, 1] for the overlay
        rgb_img_float = np.float32(cv2.resize(rgb_img, (Config.IMG_SIZE, Config.IMG_SIZE))) / 255
        
        visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
        return visualization
