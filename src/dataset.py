"""
Dataset utilities for Belgian Bird Species Classification.
"""

import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
from huggingface_hub import snapshot_download


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def download_dataset(local_dir="../data/BelgianSpecies", repo_id="Ez-Clap/bird-species"):
    """
    Download dataset from Hugging Face Hub.
    
    Args:
        local_dir: Local directory to save the dataset
        repo_id: Hugging Face repository ID
        
    Returns:
        Path to the downloaded dataset
    """
    print(f"Downloading dataset from Hugging Face: {repo_id}")
    
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=["train/**", "test/**", "valid/**", "validation/**", "val/**"],
        revision="main"
    )
    
    data_root = Path(local_dir)
    print(f"Dataset downloaded to: {data_root}")
    
    return data_root


def get_dataset_paths(data_root):
    """
    Get train, test, and validation directory paths.
    
    Args:
        data_root: Path to dataset root directory
        
    Returns:
        Tuple of (train_dir, test_dir, valid_dir)
    """
    data_root = Path(data_root)
    train_dir = data_root / 'train'
    test_dir = data_root / 'test'
    
    # Check which validation directory exists
    valid_dir = data_root / 'valid'
    if not valid_dir.exists():
        if (data_root / 'validation').exists():
            valid_dir = data_root / 'validation'
        elif (data_root / 'val').exists():
            valid_dir = data_root / 'val'
    
    return train_dir, test_dir, valid_dir


def get_train_transforms(img_size=224):
    """Get training data transforms with augmentation."""
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        A.CoarseDropout(max_holes=1, max_height=32, max_width=32, p=0.2),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=224):
    """Get validation/test data transforms without augmentation."""
    return A.Compose([
        A.Resize(img_size + 32, img_size + 32),
        A.CenterCrop(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class BirdSpeciesDataset(Dataset):
    """Custom dataset for bird species classification."""
    
    def __init__(self, root_dir, class_names=None, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Get class names if not provided
        if class_names is None:
            class_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        
        self.class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
        self.class_names = class_names
        
        # Load all image paths and labels
        for class_folder in sorted(self.root_dir.iterdir()):
            if class_folder.is_dir() and class_folder.name in self.class_to_idx:
                class_idx = self.class_to_idx[class_folder.name]
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    for img_path in class_folder.glob(ext):
                        self.image_paths.append(img_path)
                        self.labels.append(class_idx)
        
        print(f"Found {len(self.image_paths)} images in {len(self.class_names)} classes")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        label = self.labels[idx]
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label


def get_data_loaders(train_dataset, valid_dataset=None, test_dataset=None,
                    batch_size=32, num_workers=4, k_fold=None, fold_index=0):
    """
    Create data loaders for training, validation, and test sets.
    
    Args:
        train_dataset: Training dataset
        valid_dataset: Validation dataset (optional)
        test_dataset: Test dataset (optional)
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        k_fold: Number of folds for cross-validation (if valid_dataset is None)
        fold_index: Current fold index for cross-validation
        
    Returns:
        Tuple of (train_loader, valid_loader, test_loader, fold_info)
    """
    if valid_dataset is not None:
        # Standard train/val/test split
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = None
        if test_dataset:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
        
        return train_loader, valid_loader, test_loader, None
    
    else:
        # K-fold cross-validation
        if k_fold is None:
            k_fold = 5
            
        indices = list(range(len(train_dataset)))
        kfold = KFold(n_splits=k_fold, shuffle=True, random_state=42)
        
        # Get train/val indices for the current fold
        train_folds = []
        val_folds = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(indices)):
            train_folds.append(train_idx)
            val_folds.append(val_idx)
        
        # Use the specified fold
        train_idx = train_folds[fold_index]
        val_idx = val_folds[fold_index]
        
        # Create samplers
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        )
        
        valid_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, valid_loader, None, (train_folds, val_folds)


def setup_dataset(data_root=None, repo_id="Ez-Clap/bird-species", img_size=224):
    """
    Setup complete dataset pipeline with Hugging Face download.
    
    Args:
        data_root: Local directory for dataset (will download if not exists)
        repo_id: Hugging Face repository ID
        img_size: Image size for transforms
        
    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset, class_names)
    """
    if data_root is None:
        data_root = "../data/BelgianSpecies"
    
    data_root = Path(data_root)
    
    # Download dataset if it doesn't exist
    if not data_root.exists():
        data_root = download_dataset(str(data_root), repo_id)
    
    # Get dataset paths
    train_dir, test_dir, valid_dir = get_dataset_paths(data_root)
    
    # Get class names
    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    
    # Create transforms
    train_transform = get_train_transforms(img_size)
    val_transform = get_val_transforms(img_size)
    
    # Create datasets
    train_dataset = BirdSpeciesDataset(train_dir, class_names, train_transform)
    
    valid_dataset = None
    if valid_dir.exists():
        valid_dataset = BirdSpeciesDataset(valid_dir, class_names, val_transform)
    
    test_dataset = None
    if test_dir.exists():
        test_dataset = BirdSpeciesDataset(test_dir, class_names, val_transform)
    
    return train_dataset, valid_dataset, test_dataset, class_names
