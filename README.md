# Belgian Bird Species Classifier

A state-of-the-art deep learning project for classifying Belgian bird species using PyTorch and transfer learning with EfficientNet. This project achieves high accuracy through a two-phase training approach and comprehensive data augmentation.

## 🚀 Features

- **High Accuracy**: Achieves excellent classification performance using transfer learning
- **Two-Phase Training**: Optimized training strategy with classifier head pre-training and fine-tuning
- **Comprehensive Data Augmentation**: Advanced augmentation techniques for better generalization
- **Test-Time Augmentation**: Enhanced inference accuracy through TTA
- **Modular Design**: Clean, well-organized codebase with separate modules
- **Easy-to-Use Scripts**: Command-line tools for training, evaluation, and inference
- **Reproducible Results**: Fixed random seeds and deterministic operations

## 📁 Project Structure

```
aizen/
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
├── data/                      # Dataset and test images
│   ├── BelgianSpecies/        # Main dataset
│   │   ├── train/             # Training images
│   │   ├── valid/             # Validation images
│   │   └── test/              # Test images
│   ├── images_*/              # Additional test images
│   └── test_bird*.jpg         # Sample test images
├── src/                       # Source code modules
│   ├── __init__.py            # Package initialization
│   ├── dataset.py             # Dataset and data loading utilities
│   ├── models.py              # Model definitions and utilities
│   ├── training.py            # Training functions and loops
│   ├── evaluation.py          # Evaluation and visualization utilities
│   └── inference.py           # Inference and prediction utilities
├── scripts/                   # Command-line scripts
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   └── predict.py             # Inference script
├── notebooks/                 # Jupyter notebooks
│   └── train.ipynb            # Main training notebook
├── models/                    # Saved model files
│   ├── best_model_phase1_fold0.pt
│   ├── best_model_phase2_fold0.pt
│   └── bird_species_classifier.pt
├── results/                   # Training results and plots
│   ├── confusion_matrix.pdf
│   └── phase2_training_history.pdf
├── docs/                      # Documentation
├── tests/                     # Unit tests
├── 313BIRDS/                  # Virtual environment
└── detectron_env/             # Additional environment
```

## 🛠️ Installation

### Prerequisites

- Ubuntu 18.04+ (or similar Linux distribution)
- Python 3.7+
- CUDA drivers (optional, for GPU acceleration)

### Option 1: Docker Setup (Recommended)

The easiest way to get started is using Docker:

1. **Prerequisites**
   - Docker and Docker Compose installed
   - 4GB+ free disk space

2. **Quick Start**
   ```bash
   git clone <repository-url>
   cd aizen
   ./docker-run.sh build
   ./docker-run.sh train --data-dir /app/data/BelgianSpecies
   ```

3. **Available Docker Commands**
   ```bash
   ./docker-run.sh build          # Build Docker image
   ./docker-run.sh train          # Run training
   ./docker-run.sh predict <img>  # Make predictions
   ./docker-run.sh notebook       # Start Jupyter (http://localhost:8888)
   ./docker-run.sh shell          # Interactive shell
   ./docker-run.sh help           # Show all commands
   ```

📖 **See [docs/DOCKER.md](docs/DOCKER.md) for complete Docker documentation**

### Option 2: Local Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd aizen
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install PyTorch with GPU support (optional)**
   ```bash
   # For CUDA 11.8 (check pytorch.org for your CUDA version)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

## 📊 Dataset

The dataset should follow this structure:

```
data/BelgianSpecies/
├── train/
│   ├── BAR-TAILED GODWIT/
│   ├── BARN OWL/
│   ├── BARN SWALLOW/
│   └── ...
├── valid/
│   └── ... (same structure as train)
└── test/
    └── ... (same structure as train)
```

- **42 bird species** commonly found in Belgium
- **~130 images per species** for training
- **Images resized to 224×224** pixels
- **Comprehensive validation and test sets**

## 🏃‍♂️ Usage

### Docker Usage (Recommended)

**Training:**
```bash
# Basic training
./docker-run.sh train

# Custom training parameters
./docker-run.sh train --epochs-phase1 10 --epochs-phase2 50 --batch-size 16
```

**Prediction:**
```bash
# Predict single image
./docker-run.sh predict data/test_bird.jpg

# Predict with visualization
./docker-run.sh predict data/test_bird.jpg --show-plot
```

**Evaluation:**
```bash
./docker-run.sh evaluate --model-path /app/models/bird_species_classifier.pt
```

**Jupyter Notebook:**
```bash
./docker-run.sh notebook
# Access at http://localhost:8888
```

### Local Usage

**Training:**
```bash
python scripts/train.py --data-root data/BelgianSpecies --epochs-phase1 5 --epochs-phase2 30
```

**Evaluation:**
```bash
python scripts/evaluate.py --model-path models/bird_species_classifier.pt --split test
```

**Inference:**
```bash
python scripts/predict.py data/test_bird.jpg --model-path models/bird_species_classifier.pt --show-plot
```

**Jupyter Notebook:**
```bash
jupyter lab notebooks/train.ipynb
```

### Training Parameters
- `--data-root`: Path to dataset directory
- `--batch-size`: Batch size (default: 32)
- `--epochs-phase1`: Epochs for phase 1 training (default: 5)
- `--epochs-phase2`: Epochs for phase 2 training (default: 30)
- `--lr-phase1`: Learning rate for phase 1 (default: 1e-3)
- `--lr-phase2`: Learning rate for phase 2 (default: 2e-4)
- `--model-name`: Model architecture (default: efficientnet_b3)

## 🧠 Model Architecture

- **Base Model**: EfficientNet-B3 (pre-trained on ImageNet)
- **Input Size**: 224×224×3
- **Output**: 42 classes (Belgian bird species)
- **Parameters**: ~12M total, ~1.5M trainable in phase 1

### Training Strategy

1. **Phase 1**: Train only classifier head (5 epochs)
   - Freeze EfficientNet backbone
   - Train new classification layer
   - Learning rate: 1e-3

2. **Phase 2**: Fine-tune entire model (30 epochs)
   - Unfreeze last few layers
   - Lower learning rate: 2e-4
   - Cosine annealing scheduler

## 📈 Results

- **Test Accuracy**: >95% (varies by fold)
- **Training Time**: ~2-3 hours on GPU
- **Model Size**: ~47MB (exported model)

### Performance Features

- **Label Smoothing**: Reduces overfitting
- **Early Stopping**: Prevents overtraining
- **Test-Time Augmentation**: Improves inference accuracy
- **Comprehensive Data Augmentation**: Enhances generalization

## 🔧 Advanced Usage

### K-Fold Cross-Validation

For maximum performance with limited data:

```python
from src.training import train_kfold_models
models = train_kfold_models(k_folds=5)
```

### Custom Model Training

```python
from src.models import create_model
from src.training import train_phase1, train_phase2

model = create_model(num_classes=42)
model, history1 = train_phase1(model, train_loader, val_loader)
model, history2 = train_phase2(model, train_loader, val_loader)
```

### Batch Inference

```python
from src.inference import load_model_and_predict

for image_path in image_paths:
    predicted_class, confidence = load_model_and_predict(image_path)
    print(f"{image_path}: {predicted_class} ({confidence:.3f})")
```

## 🧪 Testing

Run unit tests:

```bash
python -m pytest tests/
```

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

- [Model Architecture](docs/model_architecture.md)
- [Training Guide](docs/training_guide.md)
- [API Reference](docs/api_reference.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **EfficientNet**: Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks.
- **PyTorch**: For providing an excellent deep learning framework
- **timm**: For pre-trained model implementations
- **albumentations**: For comprehensive data augmentation

## 📞 Contact

- **Author**: Mansouri El Mustapha
- **Email**: mustaphamansouripro@gmail.com
- **Project Link**: [https://github.com/E-zClap/bird-classifier](https://github.com/E-zClap/bird-classifier)

---