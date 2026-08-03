import os
import torch

class Config:
    # Paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "val")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    
    # Model parameters
    MODEL_NAME = "efficientnet_b0"
    NUM_CLASSES = 38
    IMG_SIZE = 224
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        DEVICE = "mps"
        
    # Misc
    SEED = 42
    NUM_WORKERS = min(4, os.cpu_count() or 1)

    @classmethod
    def setup_dirs(cls):
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
