import os
from src.config import Config
from src.utils import seed_everything, get_logger
from src.dataset import get_dataloaders
from src.model import PlantDiseaseModel
from src.trainer import Trainer

def main():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup_dirs()
    logger = get_logger("Train_Entry")
    
    logger.info(f"Using device: {Config.DEVICE}")
    
    # 2. Load Data
    logger.info("Loading dataloaders...")
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    logger.info(f"Found {len(classes)} classes.")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # 3. Initialize Model
    logger.info("Initializing EfficientNet-B0 model...")
    model = PlantDiseaseModel(num_classes=len(classes), freeze_backbone=True)
    
    # 4. Initialize Trainer & Load Checkpoint
    trainer = Trainer(model, train_loader, val_loader, Config.DEVICE)
    import torch
    
    start_epoch = 1
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        logger.info(f"Found existing checkpoint at {best_model_path}. Resuming training...")
        checkpoint = torch.load(best_model_path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Infer start_epoch based on checkpoint or default to 6 as requested
        start_epoch = checkpoint.get('epoch', 5) + 1
        trainer.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        
        # Re-apply LR change if we passed epoch 5
        if start_epoch > 5:
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = Config.LEARNING_RATE * 0.1
                
        logger.info(f"Resuming from Epoch {start_epoch} (Best Val F1 so far: {trainer.best_val_f1:.4f})")
    
    # 5. Start Training
    trainer.fit(start_epoch=start_epoch)
    
    # 6. Evaluate on Test Set
    logger.info("Evaluating best model on test set...")
    import torch
    
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss, test_metrics, all_labels, all_preds = trainer.evaluate(loader=test_loader)
        logger.info(f"Test Results - Loss: {test_loss:.4f}, Accuracy: {test_metrics['accuracy']:.4f}, Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1_score']:.4f}")
        
        from src.evaluate import plot_confusion_matrix
        cm_path = os.path.join(Config.LOG_DIR, "confusion_matrix.png")
        plot_confusion_matrix(all_labels, all_preds, classes, cm_path)
        logger.info(f"Confusion matrix saved to {cm_path}")
    else:
        logger.error("Best model checkpoint not found for testing.")

if __name__ == "__main__":
    main()
