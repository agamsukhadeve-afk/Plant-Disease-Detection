import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.config import Config
from src.utils import get_logger
from src.evaluate import calculate_metrics

class Trainer:
    """Trainer class to handle the training and validation loops."""
    
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = get_logger("Trainer")
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=3)
        self.scaler = GradScaler()
        self.writer = SummaryWriter(log_dir=Config.LOG_DIR)
        
        self.best_val_f1 = 0.0
        self.early_stop_counter = 0
        self.early_stop_patience = 1
        self.best_epoch = 0
        self.best_metrics = {}
        
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{Config.NUM_EPOCHS} [Train]")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed Precision Training
            with autocast(enabled=(self.device == 'cuda')):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item() * images.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(self.train_loader.dataset)
        metrics = calculate_metrics(all_labels, all_preds)
        
        self.writer.add_scalar('Loss/Train', epoch_loss, epoch)
        self.writer.add_scalar('Accuracy/Train', metrics['accuracy'], epoch)
        
        self.logger.info(f"Train - Loss: {epoch_loss:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        return epoch_loss, metrics['accuracy']

    def evaluate(self, epoch=None, loader=None):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        if loader is None:
            loader = self.val_loader
            desc = "Validation"
        else:
            desc = "Test"
            
        pbar = tqdm(loader, desc=f"{desc}")
        
        with torch.no_grad():
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                with autocast(enabled=(self.device == 'cuda')):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                running_loss += loss.item() * images.size(0)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        epoch_loss = running_loss / len(loader.dataset)
        metrics = calculate_metrics(all_labels, all_preds)
        
        if epoch is not None:
            self.writer.add_scalar('Loss/Validation', epoch_loss, epoch)
            self.writer.add_scalar('Accuracy/Validation', metrics['accuracy'], epoch)
            self.logger.info(f"Val - Loss: {epoch_loss:.4f}, Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
            
        return epoch_loss, metrics, all_labels, all_preds

    def save_checkpoint(self, epoch, metrics, filename="best_model.pth"):
        filepath = os.path.join(Config.CHECKPOINT_DIR, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1,
            'epoch': epoch,
            'metrics': metrics
        }, filepath)
        self.logger.info(f"Checkpoint saved to {filepath} (Epoch {epoch})")
        
    def fit(self, start_epoch=1):
        self.logger.info("Starting training...")
        
        if start_epoch <= 5:
            self.logger.info("Phase 1: Training classifier head only")
        else:
            self.logger.info(f"Resuming at epoch {start_epoch}. Backbone should be unfrozen.")
            self.model.unfreeze_backbone()
        
        for epoch in range(start_epoch, Config.NUM_EPOCHS + 1):
            if epoch == 5:
                self.logger.info("Phase 2: Unfreezing backbone for fine-tuning")
                self.model.unfreeze_backbone()
                # Optional: reduce learning rate for fine-tuning
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = Config.LEARNING_RATE * 0.1
            
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_metrics, _, _ = self.evaluate(epoch)
            val_f1 = val_metrics['f1_score']
            
            self.scheduler.step(val_f1)
            
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_epoch = epoch
                self.best_metrics = val_metrics
                self.save_checkpoint(epoch, val_metrics)
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                
            if self.early_stop_counter >= self.early_stop_patience:
                self.logger.info(f"Early stopping triggered! No improvement in Validation F1-score for {self.early_stop_patience} consecutive epochs.")
                break
                
        self.writer.close()
        self.logger.info(f"Training complete! Best Epoch: {self.best_epoch}, Best Val F1: {self.best_val_f1:.4f}")
        
        # Restore best model checkpoint at the end of training
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        if os.path.exists(best_model_path):
            self.logger.info("Restoring best model checkpoint...")
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
