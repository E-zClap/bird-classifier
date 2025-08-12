# Model Architecture

## Overview

The Belgian Bird Species Classifier uses **EfficientNet-B3** as the backbone architecture with transfer learning from ImageNet pre-trained weights.

## Architecture Details

### Base Model: EfficientNet-B3

- **Input Size**: 224×224×3 RGB images
- **Parameters**: ~12M total parameters
- **Pre-training**: ImageNet-1k dataset
- **Architecture**: Compound scaling of depth, width, and resolution

### Model Modifications

1. **Classifier Head Replacement**
   - Original: 1000 classes (ImageNet)
   - Modified: 42 classes (Belgian bird species)
   - Layer: Single Linear layer with dropout

2. **Feature Extraction**
   - Input: 224×224×3 images
   - Features: 1536-dimensional feature vector
   - Output: 42-class probability distribution

## Training Strategy

### Two-Phase Training Approach

#### Phase 1: Classifier Head Training
- **Duration**: 5 epochs
- **Frozen Layers**: All EfficientNet backbone layers
- **Trainable**: Only the final classification layer
- **Learning Rate**: 1e-3
- **Optimizer**: AdamW with weight decay
- **Scheduler**: StepLR (step_size=3, gamma=0.5)

#### Phase 2: Fine-tuning
- **Duration**: 30 epochs
- **Unfrozen Layers**: Last 2 blocks + classifier
- **Learning Rate**: 2e-4
- **Scheduler**: CosineAnnealingLR
- **Early Stopping**: Patience=7 epochs

### Loss Function and Regularization

- **Loss**: CrossEntropyLoss with label smoothing (0.1)
- **Weight Decay**: 1e-4
- **Early Stopping**: Monitors validation accuracy
- **Data Augmentation**: Comprehensive augmentation pipeline

## Performance Optimizations

### Data Augmentation

**Training Augmentations:**
- Random resized crop (scale: 0.7-1.0)
- Horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation, hue)
- Coarse dropout (max 32×32 patches)

**Validation/Test Augmentations:**
- Resize to 256×256
- Center crop to 224×224
- Normalization (ImageNet statistics)

### Test-Time Augmentation (TTA)

- Original image prediction
- Horizontal flip prediction
- Average of both predictions
- Improves robustness and accuracy

## Model Export

The final model is exported with:
- Model state dictionary
- Class names mapping
- Image preprocessing parameters
- Model architecture information

This allows for easy loading and inference without requiring the training code.
