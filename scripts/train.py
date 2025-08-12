#!/usr/bin/env python3
"""
Train the Belgian Bird Species Classifier.
"""

import argparse
import random
import numpy as np
import torch
from pathlib import Path

from src.dataset import BirdSpeciesDataset, get_data_loaders, get_train_transforms, get_val_transforms
from src.models import create_model
from src.training import train_phase1, train_phase2
from src.evaluation import plot_training_history
from src.inference import export_model


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Train Belgian Bird Species Classifier')
    parser.add_argument('--data-root', type=str, default='data/BelgianSpecies',
                        help='Path to dataset root directory')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--epochs-phase1', type=int, default=5,
                        help='Number of epochs for phase 1 training')
    parser.add_argument('--epochs-phase2', type=int, default=30,
                        help='Number of epochs for phase 2 training')
    parser.add_argument('--lr-phase1', type=float, default=1e-3,
                        help='Learning rate for phase 1')
    parser.add_argument('--lr-phase2', type=float, default=2e-4,
                        help='Learning rate for phase 2')
    parser.add_argument('--model-name', type=str, default='efficientnet_b3',
                        help='Model architecture name')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset paths
    data_root = Path(args.data_root)
    train_dir = data_root / 'train'
    valid_dir = data_root / 'valid'
    test_dir = data_root / 'test'
    
    # Get class names
    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    
    # Create datasets
    train_dataset = BirdSpeciesDataset(train_dir, class_names, get_train_transforms(args.img_size))
    valid_dataset = BirdSpeciesDataset(valid_dir, class_names, get_val_transforms(args.img_size))
    test_dataset = BirdSpeciesDataset(test_dir, class_names, get_val_transforms(args.img_size)) if test_dir.exists() else None
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")
    if test_dataset:
        print(f"Test dataset size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader, val_loader, test_loader, _ = get_data_loaders(
        train_dataset, valid_dataset, test_dataset,
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    # Create model
    model = create_model(num_classes, model_name=args.model_name)
    model = model.to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Phase 1: Train classifier head
    print("\nStarting Phase 1 Training: Classifier head only")
    model, history_phase1 = train_phase1(
        model, train_loader, val_loader,
        num_epochs=args.epochs_phase1,
        lr=args.lr_phase1
    )
    
    # Plot Phase 1 results
    plot_training_history(history_phase1, "Phase 1 Training: Classifier Head Only",
                         'results/phase1_training_history.pdf')
    
    # Phase 2: Fine-tune model
    print("\nStarting Phase 2 Training: Fine-tuning")
    model, history_phase2 = train_phase2(
        model, train_loader, val_loader,
        num_epochs=args.epochs_phase2,
        lr=args.lr_phase2
    )
    
    # Plot Phase 2 results
    plot_training_history(history_phase2, "Phase 2 Training: Fine-tuning",
                         'results/phase2_training_history.pdf')
    
    # Export final model
    export_model(model, class_names, 'models/bird_species_classifier.pt',
                args.img_size, args.model_name)
    
    print("Training completed successfully!")


if __name__ == '__main__':
    main()
