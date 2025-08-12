#!/bin/bash

# Belgian Bird Species Classifier Docker Setup Script
# This script provides easy commands to work with the dockerized application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to build the Docker image
build() {
    print_info "Building Docker image..."
    docker-compose build
    print_success "Docker image built successfully!"
}

# Function to run training
train() {
    print_info "Starting training..."
    docker-compose run --rm bird-classifier-train python scripts/train.py "$@"
}

# Function to run prediction
predict() {
    if [ $# -eq 0 ]; then
        print_error "Please provide an image path"
        echo "Usage: $0 predict <image_path> [additional_args]"
        exit 1
    fi
    
    print_info "Making prediction on: $1"
    docker-compose run --rm bird-classifier-predict python scripts/predict.py "$@"
}

# Function to run evaluation
evaluate() {
    print_info "Running evaluation..."
    docker-compose run --rm bird-classifier-eval python scripts/evaluate.py "$@"
}

# Function to start Jupyter notebook
notebook() {
    print_info "Starting Jupyter notebook server..."
    print_info "Notebook will be available at: http://localhost:8888"
    docker-compose up bird-classifier-notebook
}

# Function to run interactive shell
shell() {
    print_info "Starting interactive shell..."
    docker-compose run --rm bird-classifier bash
}

# Function to run custom Python command
python() {
    print_info "Running Python command..."
    docker-compose run --rm bird-classifier python "$@"
}

# Function to stop all services
stop() {
    print_info "Stopping all services..."
    docker-compose down
    print_success "All services stopped!"
}

# Function to clean up Docker resources
clean() {
    print_warning "This will remove Docker images and containers. Continue? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_info "Cleaning up Docker resources..."
        docker-compose down --rmi all --volumes --remove-orphans
        print_success "Cleanup completed!"
    else
        print_info "Cleanup cancelled."
    fi
}

# Function to show logs
logs() {
    docker-compose logs -f "$@"
}

# Function to show help
show_help() {
    echo "Belgian Bird Species Classifier Docker Helper"
    echo ""
    echo "Usage: $0 <command> [arguments]"
    echo ""
    echo "Commands:"
    echo "  build                Build Docker image"
    echo "  train [args]         Run training with optional arguments"
    echo "  predict <image>      Predict bird species from image"
    echo "  evaluate [args]      Run evaluation with optional arguments"
    echo "  notebook             Start Jupyter notebook server"
    echo "  shell                Start interactive shell in container"
    echo "  python <command>     Run Python command in container"
    echo "  stop                 Stop all running services"
    echo "  clean                Remove Docker images and containers"
    echo "  logs [service]       Show logs for service"
    echo "  help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 train --epochs 10 --batch-size 32"
    echo "  $0 predict data/test_bird.jpg"
    echo "  $0 evaluate --model-path models/best_model.pt"
    echo "  $0 notebook"
    echo "  $0 shell"
    echo "  $0 python --version"
}

# Main script logic
main() {
    check_docker
    
    case "${1:-help}" in
        "build")
            build
            ;;
        "train")
            shift
            train "$@"
            ;;
        "predict")
            shift
            predict "$@"
            ;;
        "evaluate")
            shift
            evaluate "$@"
            ;;
        "notebook")
            notebook
            ;;
        "shell")
            shell
            ;;
        "python")
            shift
            python "$@"
            ;;
        "stop")
            stop
            ;;
        "clean")
            clean
            ;;
        "logs")
            shift
            logs "$@"
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
