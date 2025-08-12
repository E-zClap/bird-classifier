#!/usr/bin/env python3
"""
Evaluate the Belgian Bird Species Classifier.
"""

import argparse
import torch

from src.dataset import setup_dataset, get_data_loaders
from src.models import create_model
from src.evaluation import evaluate_model, plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser(description='Evaluate Belgian Bird Species Classifier')
    parser.add_argument('--model-path', type=str, default='models/bird_species_classifier.pt',
                        help='Path to trained model')
    parser.add_argument('--data-root', type=str, default='data/BelgianSpecies',
                        help='Path to dataset root directory (will download if not exists)')
    parser.add_argument('--repo-id', type=str, default='Ez-Clap/bird-species',
                        help='Hugging Face repository ID for dataset')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'],
                        help='Dataset split to evaluate on')
    
    args = parser.parse_args()
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model info
    model_info = torch.load(args.model_path, map_location=device)
    class_names = model_info['class_names']
    
    # Create model
    model = create_model(len(class_names), model_name=model_info['model_name'])
    model.load_state_dict(model_info['state_dict'])
    model = model.to(device)
    
    # Setup dataset (will download if needed)
    train_dataset, valid_dataset, test_dataset, dataset_class_names = setup_dataset(
        data_root=args.data_root,
        repo_id=args.repo_id,
        img_size=model_info['img_size']
    )
    
    # Verify class names match
    if class_names != dataset_class_names:
        print("Warning: Model class names don't match dataset class names!")
        print(f"Model classes: {class_names}")
        print(f"Dataset classes: {dataset_class_names}")
    
    # Select evaluation dataset
    if args.split == 'train':
        eval_dataset = train_dataset
    elif args.split == 'valid':
        eval_dataset = valid_dataset
        if eval_dataset is None:
            print("No validation dataset found, using test dataset instead")
            eval_dataset = test_dataset
    else:  # test
        eval_dataset = test_dataset
        if eval_dataset is None:
            print("No test dataset found, using validation dataset instead")
            eval_dataset = valid_dataset
    
    if eval_dataset is None:
        print(f"No {args.split} dataset available!")
        return
    
    print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # Create data loader
    _, eval_loader, _, _ = get_data_loaders(
        None, eval_dataset, None,
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    # Evaluate model
    print(f"Evaluating model on {args.split} set...")
    all_preds, all_targets, cm = evaluate_model(model, eval_loader, device, class_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names, f'results/confusion_matrix_{args.split}.pdf')
    
    print("Evaluation completed successfully!")


if __name__ == '__main__':
    main()
