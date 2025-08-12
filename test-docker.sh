#!/bin/bash

# Docker setup verification script
# Run this script to test the Docker setup

set -e

echo "🔍 Testing Docker setup for Belgian Bird Species Classifier..."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker and try again."
    exit 1
fi

echo "✅ Docker is running"

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: docker-compose is not installed"
    exit 1
fi

echo "✅ Docker Compose is available"

# Test building the image
echo "🔨 Building Docker image (this may take a few minutes)..."
if ./docker-run.sh build > /dev/null 2>&1; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Error: Failed to build Docker image"
    exit 1
fi

# Test Python environment
echo "🐍 Testing Python environment..."
if ./docker-run.sh python --version > /dev/null 2>&1; then
    VERSION=$(./docker-run.sh python --version 2>&1)
    echo "✅ Python is working: $VERSION"
else
    echo "❌ Error: Python environment test failed"
    exit 1
fi

# Test PyTorch
echo "🔥 Testing PyTorch installation..."
if ./docker-run.sh python -c "import torch; print(f'PyTorch {torch.__version__}')" > /dev/null 2>&1; then
    TORCH_VERSION=$(./docker-run.sh python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>&1)
    echo "✅ PyTorch is working: $TORCH_VERSION"
else
    echo "❌ Error: PyTorch test failed"
    exit 1
fi

# Test source code import
echo "📦 Testing source code imports..."
if ./docker-run.sh python -c "from src import models, dataset, training; print('All imports successful')" > /dev/null 2>&1; then
    echo "✅ Source code imports working"
else
    echo "❌ Error: Source code import test failed"
    exit 1
fi

# Test script availability
echo "📋 Testing script availability..."
if ./docker-run.sh python scripts/train.py --help > /dev/null 2>&1; then
    echo "✅ Training script is accessible"
else
    echo "❌ Error: Training script test failed"
    exit 1
fi

if ./docker-run.sh python scripts/predict.py --help > /dev/null 2>&1; then
    echo "✅ Prediction script is accessible"
else
    echo "❌ Error: Prediction script test failed"
    exit 1
fi

# Test Jupyter availability
echo "📓 Testing Jupyter availability..."
if ./docker-run.sh python -c "import jupyter; print('Jupyter available')" > /dev/null 2>&1; then
    echo "✅ Jupyter is available"
else
    echo "❌ Error: Jupyter test failed"
    exit 1
fi

echo ""
echo "🎉 All tests passed! Docker setup is working correctly."
echo ""
echo "Next steps:"
echo "1. Place your dataset in the data/ directory"
echo "2. Run: ./docker-run.sh train"
echo "3. Run: ./docker-run.sh predict <image_path>"
echo "4. Run: ./docker-run.sh notebook (for Jupyter)"
echo ""
echo "For help: ./docker-run.sh help"
