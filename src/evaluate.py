import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    Calculates various evaluation metrics.
    
    Args:
        y_true (list or np.array): True labels
        y_pred (list or np.array): Predicted labels
        
    Returns:
        dict: A dictionary containing accuracy, precision, recall, and f1_score
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def get_confusion_matrix(y_true, y_pred):
    """
    Returns the confusion matrix.
    """
    return confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    
    cm = confusion_matrix(y_true, y_pred)
    # Figure size depends on number of classes
    fig, ax = plt.subplots(figsize=(20, 20))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
    
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
