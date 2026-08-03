import os
import random
import logging
import numpy as np
import torch

def seed_everything(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_logger(name: str = "PlantDiseaseDetection", log_file: str = "train.log") -> logging.Logger:
    """Sets up and returns a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid adding handlers multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Stream handler
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        
        # File handler
        if log_file:
            from src.config import Config
            Config.setup_dirs()
            fh = logging.FileHandler(os.path.join(Config.LOG_DIR, log_file))
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
    return logger
