"""
Training utilities and functions for Belgian Bird Species Classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device (CPU or GPU)
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    for batch_idx, (inputs, targets) in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Calculate statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{running_loss / (batch_idx + 1):.4f}",
            'acc': f"{100. * correct / total:.2f}%"
        })
    
    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: PyTorch model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device (CPU or GPU)
        
    Returns:
        Tuple of (average_loss, accuracy, predictions, targets)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation")
        for batch_idx, (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Calculate statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store predictions and targets for later analysis
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{running_loss / (batch_idx + 1):.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })
    
    return running_loss / len(dataloader), 100. * correct / total, all_preds, all_targets


def train_phase1(model, train_loader, val_loader, num_epochs=5, lr=1e-3, weight_decay=1e-4):
    """
    Phase 1: Train only the classifier head.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        lr: Learning rate
        weight_decay: Weight decay
        
    Returns:
        Tuple of (trained_model, history)
    """
    from .models import EarlyStopping
    
    device = next(model.parameters()).device
    
    # Freeze all layers except the classifier
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the classifier
    if hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        for param in model.fc.parameters():
            param.requires_grad = True
    
    # Define loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=5, mode='max')
    
    # Training loop
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 10)
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validate
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # Update learning rate
        scheduler.step()
        
        # Check if we should save the model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/best_model_phase1.pt')
            print(f"Saved best model with accuracy: {val_acc:.2f}%")
        
        # Check early stopping
        if early_stopping(val_acc) and early_stopping.early_stop:
            print("Early stopping triggered!")
            break
    
    return model, history


def train_phase2(model, train_loader, val_loader, num_epochs=30, lr=2e-4, weight_decay=1e-4):
    """
    Phase 2: Fine-tune the model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        lr: Learning rate
        weight_decay: Weight decay
        
    Returns:
        Tuple of (trained_model, history)
    """
    from .models import EarlyStopping
    
    device = next(model.parameters()).device
    
    # Unfreeze layers for fine-tuning
    for name, param in model.named_parameters():
        if 'blocks.6' in name or 'blocks.5' in name or 'classifier' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # Define loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=7, mode='max')
    
    # Training loop
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 10)
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validate
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # Update learning rate
        scheduler.step()
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.7f}")
        
        # Check if we should save the model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/best_model_phase2.pt')
            print(f"Saved best model with accuracy: {val_acc:.2f}%")
        
        # Check early stopping
        if early_stopping(val_acc) and early_stopping.early_stop:
            print("Early stopping triggered!")
            break
    
    return model, history
