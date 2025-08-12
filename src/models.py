"""
Model definitions and utilities for Belgian Bird Species Classification.
"""

import torch.nn as nn
import timm


def create_model(num_classes, model_name="efficientnet_b3", pretrained=True):
    """
    Create a model with pretrained weights.
    
    Args:
        num_classes: Number of output classes
        model_name: Name of the model architecture
        pretrained: Whether to use pretrained weights
        
    Returns:
        PyTorch model
    """
    model = timm.create_model(model_name, pretrained=pretrained)
    
    # Replace classifier head
    if 'efficientnet' in model_name:
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    else:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    
    return model


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience=7, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return True
        
        if self.mode == 'min':
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.counter = 0
                return True
            else:
                self.counter += 1
        else:  # mode == 'max'
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.counter = 0
                return True
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.early_stop = True
            
        return False
