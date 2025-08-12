"""
Evaluation utilities for Belgian Bird Species Classification.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm


def evaluate_model(model, dataloader, device, class_names):
    """
    Evaluate model performance on a dataset.
    
    Args:
        model: Trained PyTorch model
        dataloader: Data loader for evaluation
        device: Device (CPU or GPU)
        class_names: List of class names
        
    Returns:
        Tuple of (predictions, targets, confusion_matrix)
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate accuracy
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_targets))
    print(f"Overall Accuracy: {accuracy:.2f}%")
    
    # Create confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    for i, acc in enumerate(per_class_acc):
        print(f"Class {class_names[i]}: {acc:.2f}%")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names))
    
    return all_preds, all_targets, cm


def plot_confusion_matrix(cm, class_names, save_path='results/confusion_matrix.pdf'):
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """
    fig_size = max(12, len(class_names) * 0.6)
    
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'shrink': 0.8}, annot_kws={'size': 8})
    plt.xlabel('Predicted', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=14, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, ha='right', fontsize=9)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight', 
                pad_inches=0.1, dpi=300, transparent=True)
    plt.show()


def plot_training_history(history, title="Training History", save_path='results/training_history.pdf'):
    """
    Plot training and validation metrics.
    
    Args:
        history: Dictionary containing training history
        title: Plot title
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    
    # Plot loss
    ax1.plot(history['train_loss'], color='#1f77b4', linewidth=2, 
             marker='o', markersize=4, label='Train Loss')
    ax1.plot(history['val_loss'], color='#ff7f0e', linewidth=2, 
             marker='s', markersize=4, label='Val Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curve', fontsize=14)
    ax1.legend(frameon=False)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], color='#2ca02c', linewidth=2, 
             marker='o', markersize=4, label='Train Accuracy')
    ax2.plot(history['val_acc'], color='#d62728', linewidth=2, 
             marker='s', markersize=4, label='Val Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy Curve', fontsize=14)
    ax2.legend(frameon=False)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    
    plt.savefig(save_path, format='pdf', bbox_inches='tight', 
                pad_inches=0, dpi=300, transparent=True)
    plt.show()
