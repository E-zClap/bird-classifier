# Docker Setup and Usage Guide

This guide explains how to use the dockerized version of the Belgian Bird Species Classifier.

## Prerequisites

- Docker and Docker Compose installed on your system
- At least 4GB of free disk space
- Internet connection for downloading dependencies

## Quick Start

1. **Build the Docker image:**
   ```bash
   ./docker-run.sh build
   ```

2. **Train a model:**
   ```bash
   ./docker-run.sh train --data-dir /app/data/BelgianSpecies --epochs 10
   ```

3. **Make predictions:**
   ```bash
   ./docker-run.sh predict /app/data/test_bird.jpg
   ```

4. **Start Jupyter notebook:**
   ```bash
   ./docker-run.sh notebook
   ```
   Then open http://localhost:8888 in your browser.

## Available Commands

The `docker-run.sh` script provides the following commands:

### Core Operations
- `build` - Build the Docker image
- `train [args]` - Run training with optional arguments
- `predict <image>` - Predict bird species from an image
- `evaluate [args]` - Run model evaluation
- `notebook` - Start Jupyter notebook server on port 8888
- `shell` - Start an interactive bash shell inside the container
- `python <command>` - Run a Python command inside the container

### Management
- `stop` - Stop all running Docker services
- `clean` - Remove Docker images and containers (with confirmation)
- `logs [service]` - Show logs for a specific service or all services
- `help` - Show detailed usage information

## Usage Examples

### Training Examples
```bash
# Basic training
./docker-run.sh train

# Training with custom parameters
./docker-run.sh train --epochs 20 --batch-size 16 --lr 0.001

# Training with specific data directory
./docker-run.sh train --data-dir /app/data/BelgianSpecies --output-dir /app/models
```

### Prediction Examples
```bash
# Predict a single image
./docker-run.sh predict /app/data/test_bird.jpg

# Predict with visualization
./docker-run.sh predict /app/data/test_bird.jpg --show-plot

# Predict with custom model
./docker-run.sh predict /app/data/test_bird.jpg --model-path /app/models/custom_model.pt
```

### Development Examples
```bash
# Start interactive shell for debugging
./docker-run.sh shell

# Run Python commands
./docker-run.sh python --version
./docker-run.sh python -c "import torch; print(torch.__version__)"

# Install additional packages (temporary)
./docker-run.sh shell
# Inside container: pip install new-package
```

### Jupyter Notebook
```bash
# Start notebook server
./docker-run.sh notebook

# Access at http://localhost:8888
# No password required for development setup
```

## Directory Structure in Docker

The Docker setup mounts the following directories:

```
Host Directory    →    Container Directory
./data           →    /app/data
./models         →    /app/models
./results        →    /app/results
./notebooks      →    /app/notebooks
```

## Data Setup

1. **Prepare your dataset:**
   - Place your dataset in the `data/` directory
   - For the Belgian Species dataset, structure should be:
     ```
     data/BelgianSpecies/
     ├── train/
     ├── valid/
     └── test/
     ```

2. **Test images:**
   - Place test images in `data/` or any subdirectory
   - Supported formats: JPG, JPEG, PNG

## GPU Support (Optional)

To use GPU acceleration, modify the `docker-compose.yml`:

1. **Install NVIDIA Docker runtime** on your host system

2. **Update docker-compose.yml** to add GPU support:
   ```yaml
   services:
     bird-classifier:
       # ... existing configuration ...
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]
   ```

3. **Use CUDA-enabled PyTorch** by updating requirements.txt:
   ```
   torch>=1.9.0+cu111
   torchvision>=0.10.0+cu111
   ```

## Troubleshooting

### Common Issues

1. **Docker not running:**
   ```
   Error: Docker is not running
   Solution: Start Docker service and try again
   ```

2. **Permission denied:**
   ```bash
   chmod +x docker-run.sh
   ```

3. **Port 8888 already in use:**
   ```bash
   # Stop existing services
   ./docker-run.sh stop
   # Or change port in docker-compose.yml
   ```

4. **Out of disk space:**
   ```bash
   # Clean up Docker resources
   ./docker-run.sh clean
   ```

### Viewing Logs
```bash
# View all logs
./docker-run.sh logs

# View specific service logs
./docker-run.sh logs bird-classifier-train

# Follow logs in real-time
docker-compose logs -f bird-classifier-train
```

### Debugging
```bash
# Start interactive shell
./docker-run.sh shell

# Check Python environment
./docker-run.sh python -c "import sys; print(sys.path)"

# Check installed packages
./docker-run.sh python -m pip list
```

## Performance Tips

1. **Use .dockerignore** to exclude unnecessary files from Docker context
2. **Mount only necessary directories** to reduce I/O overhead
3. **Use specific Docker tags** instead of 'latest' for reproducibility
4. **Clean up regularly** with `./docker-run.sh clean`

## Security Considerations

- The Jupyter notebook runs without authentication (development setup)
- Container runs as root (for simplicity)
- For production, consider:
  - Adding authentication to Jupyter
  - Running as non-root user
  - Using secrets for sensitive data

## Docker Compose Services

The setup includes several pre-configured services:

- `bird-classifier` - Main interactive service
- `bird-classifier-train` - Training service
- `bird-classifier-predict` - Prediction service
- `bird-classifier-eval` - Evaluation service
- `bird-classifier-notebook` - Jupyter notebook service

Each service can be run independently using Docker Compose commands.
