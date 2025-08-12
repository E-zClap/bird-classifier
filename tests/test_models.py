"""
Unit tests for the Belgian Bird Species Classifier.
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
from PIL import Image

# Import modules to test
from src.models import create_model, EarlyStopping
from src.dataset import BirdSpeciesDataset, get_train_transforms, get_val_transforms
from src.inference import test_time_augmentation


class TestModels(unittest.TestCase):
    """Test model creation and utilities."""
    
    def test_create_model(self):
        """Test model creation with different parameters."""
        model = create_model(num_classes=10, model_name="efficientnet_b0")
        self.assertIsInstance(model, torch.nn.Module)
        
        # Check output size
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        self.assertEqual(output.shape[1], 10)
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        early_stopping = EarlyStopping(patience=3, mode='max')
        
        # Test improving scores
        self.assertTrue(early_stopping(0.5))
        self.assertTrue(early_stopping(0.6))
        self.assertFalse(early_stopping.early_stop)
        
        # Test non-improving scores
        self.assertFalse(early_stopping(0.5))
        self.assertFalse(early_stopping(0.4))
        self.assertFalse(early_stopping(0.3))
        self.assertTrue(early_stopping.early_stop)


class TestDataset(unittest.TestCase):
    """Test dataset functionality."""
    
    def setUp(self):
        """Set up temporary dataset for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.class_names = ['class1', 'class2']
        
        # Create fake dataset structure
        for class_name in self.class_names:
            class_dir = Path(self.test_dir) / class_name
            class_dir.mkdir()
            
            # Create fake images
            for i in range(3):
                img = Image.new('RGB', (224, 224), color='red')
                img.save(class_dir / f'image_{i}.jpg')
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.test_dir)
    
    def test_dataset_creation(self):
        """Test dataset creation and loading."""
        transform = get_val_transforms()
        dataset = BirdSpeciesDataset(self.test_dir, self.class_names, transform)
        
        self.assertEqual(len(dataset), 6)  # 3 images per class * 2 classes
        self.assertEqual(len(dataset.class_names), 2)
        
        # Test data loading
        image, label = dataset[0]
        self.assertIsInstance(image, torch.Tensor)
        self.assertIn(label, [0, 1])
    
    def test_transforms(self):
        """Test data transforms."""
        train_transform = get_train_transforms()
        val_transform = get_val_transforms()
        
        # Create dummy image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Test transforms
        train_result = train_transform(image=img)
        val_result = val_transform(image=img)
        
        self.assertIsInstance(train_result['image'], torch.Tensor)
        self.assertIsInstance(val_result['image'], torch.Tensor)
        self.assertEqual(train_result['image'].shape, (3, 224, 224))
        self.assertEqual(val_result['image'].shape, (3, 224, 224))


class TestInference(unittest.TestCase):
    """Test inference functionality."""
    
    def test_test_time_augmentation(self):
        """Test TTA functionality."""
        model = create_model(num_classes=5, model_name="efficientnet_b0")
        model.eval()
        
        # Create dummy input
        image_tensor = torch.randn(3, 224, 224)
        device = torch.device('cpu')
        
        predicted_class, probabilities = test_time_augmentation(model, image_tensor, device)
        
        self.assertIsInstance(predicted_class, int)
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(len(probabilities), 5)
        self.assertAlmostEqual(probabilities.sum(), 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
