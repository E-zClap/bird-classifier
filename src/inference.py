"""
Inference utilities for Belgian Bird Species Classification.
"""

import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .models import create_model
from .dataset import IMAGENET_MEAN, IMAGENET_STD


def test_time_augmentation(model, image_tensor, device, num_augmentations=10):
    """
    Apply test-time augmentation for more robust predictions.
    
    Args:
        model: Trained PyTorch model
        image_tensor: Input image tensor
        device: Device (CPU or GPU)
        num_augmentations: Number of augmentations to apply
        
    Returns:
        Tuple of (predicted_class, probabilities)
    """
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Original prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
    
    # Horizontal flip
    flipped = torch.flip(image_tensor, [3])
    with torch.no_grad():
        flip_outputs = model(flipped)
        flip_probabilities = torch.softmax(flip_outputs, dim=1)
    
    # Average the predictions
    avg_probabilities = (probabilities + flip_probabilities) / 2
    
    # Get top prediction
    _, predicted_class = torch.max(avg_probabilities, 1)
    
    return predicted_class.item(), avg_probabilities.cpu().numpy()[0]


def load_model_and_predict(image_path, model_path='models/bird_species_classifier.pt'):
    """
    Load a trained model and make predictions on an image.
    
    Args:
        image_path: Path to the input image
        model_path: Path to the saved model
        
    Returns:
        Tuple of (predicted_class_name, confidence)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model info
    model_info = torch.load(model_path, map_location=device)
    
    # Create model
    model = create_model(len(model_info['class_names']), model_name=model_info['model_name'])
    model.load_state_dict(model_info['state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    transform = A.Compose([
        A.Resize(model_info['img_size'], model_info['img_size']),
        A.Normalize(mean=model_info['mean'], std=model_info['std']),
        ToTensorV2(),
    ])
    img_tensor = transform(image=np.array(img))['image']
    
    # Apply TTA for better prediction
    class_idx, probs = test_time_augmentation(model, img_tensor, device)
    
    # Get predicted class name and confidence
    predicted_class = model_info['class_names'][class_idx]
    confidence = probs[class_idx]
    
    return predicted_class, confidence


def export_model(model, class_names, filename='models/bird_species_classifier.pt', 
                img_size=224, model_name='efficientnet_b3'):
    """
    Export model for inference.
    
    Args:
        model: Trained PyTorch model
        class_names: List of class names
        filename: Output filename
        img_size: Image size used during training
        model_name: Model architecture name
    """
    model_info = {
        'state_dict': model.state_dict(),
        'class_names': class_names,
        'model_name': model_name,
        'img_size': img_size,
        'mean': IMAGENET_MEAN,
        'std': IMAGENET_STD
    }
    torch.save(model_info, filename)
    print(f"Model exported to {filename}")
