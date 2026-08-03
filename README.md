# Plant Disease Detection 🌿

An end-to-end Machine Learning pipeline for Plant Disease Detection using PyTorch (EfficientNet-B0) and Streamlit. This project includes data preparation, custom dataset loading, model training with mixed precision, evaluation, and a web application with Grad-CAM visualizations.

## Features
- **Transfer Learning**: EfficientNet-B0 backbone fine-tuned for 38 classes.
- **Data Augmentation**: Robust augmentations via `albumentations` (Flip, Rotate, Color Jitter).
- **Advanced Training**: Mixed Precision (AMP), AdamW, ReduceLROnPlateau, Early Stopping.
- **Interpretability**: Grad-CAM integration to visualize the model's focus areas.
- **Web App**: Interactive Streamlit app for fast inference.
- **Production-Ready**: Clean modular code following PEP-8 best practices.

## Project Structure
```text
Plant_Disease_Detection/
├── data/                    # Dataset (created automatically by preparation script)
├── src/                     # Source modules
│   ├── config.py            # Hyperparameters and paths
│   ├── utils.py             # Logging and seeding
│   ├── transforms.py        # Data augmentations
│   ├── dataset.py           # Custom PyTorch Dataset
│   ├── model.py             # EfficientNet-B0 definition
│   ├── trainer.py           # Training & Validation loop
│   ├── evaluate.py          # Metrics calculation (Accuracy, F1, etc.)
│   ├── predict.py           # Inference wrapper
│   └── gradcam.py           # Grad-CAM visualization
├── scripts/
│   └── prepare_data.py      # Script to download and split Kaggle dataset
├── checkpoints/             # Saved model weights
├── logs/                    # Training logs and TensorBoard runs
├── app.py                   # Streamlit web application
├── train.py                 # Main training script
├── requirements.txt         # Project dependencies
└── README.md                # This file
```

## Setup Instructions

### 1. Environment
Ensure you have Python 3.11+ installed.
```bash
# Clone the repository (if applicable)
cd Plant_Disease_Detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data
Ensure you have your Kaggle API key configured (`~/.kaggle/kaggle.json`). The script will download the `emmarex/plantdisease` dataset and split it automatically into 80% train, 10% val, 10% test.
```bash
python scripts/prepare_data.py
```

### 3. Training
To train the model from scratch, simply run:
```bash
python train.py
```
Checkpoints will be saved in `checkpoints/best_model.pth`.
You can track the progress using TensorBoard:
```bash
tensorboard --logdir=logs
```

### 4. Running the Web App
Once the model is trained (or if you already have a checkpoint), you can launch the Streamlit app:
```bash
streamlit run app.py
```
Upload any leaf image to get the Top-3 disease predictions along with a Grad-CAM heatmap showing what parts of the leaf the network analyzed to make its decision.

## Performance
- **Target metric**: Accuracy > 85% on validation set.
- **Evaluation**: The model computes Accuracy, Precision, Recall, and F1-score.

## Author
*AI Developer*
