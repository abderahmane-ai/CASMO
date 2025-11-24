"""
DP-SGD Benchmark: CIFAR-10 with True Differential Privacy (Opacus)
Benchmark ID: B4

Tests CASMO's performance in a rigorous Differential Privacy setting using Opacus.
This benchmark compares CASMO against DP-SGD and DP-Adam under the same privacy budget.

Task:
    Dataset: CIFAR-10
    Model: ResNet18 (Adapted for CIFAR-10)
    Mechanism: Opacus PrivacyEngine (Per-sample clipping + Gaussian Noise)
    
    Baselines:
    - DP-SGD (Standard SGD + DP)
    - DP-Adam (AdamW + DP)
    
    Metrics:
    - Accuracy vs Epsilon (Privacy Budget)
    - Convergence Speed
"""

import sys
import os
import argparse
import time
import random
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import casmo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from models import ResNet18CIFAR
from baselines import get_optimizer, make_private, OPACUS_AVAILABLE

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_agar_confidence(optimizer):
    # Unwrap if wrapped by Opacus
    # Opacus wraps the optimizer in DPOptimizer
    if hasattr(optimizer, 'original_optimizer'):
        optimizer = optimizer.original_optimizer
        
    if not hasattr(optimizer, '_group_states'):
        return None, None
    group_state = optimizer._group_states.get(0, {})
    return group_state.get('current_agar'), group_state.get('current_confidence')

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------

def run_benchmark(optimizer_name, device, train_loader, test_loader, 
                 num_epochs=20, lr=1e-3, max_grad_norm=1.0, noise_multiplier=1.0, seed=42):
    
    print(f"\n{'='*70}")
    print(f"Running: {optimizer_name.upper()} | Noise: {noise_multiplier} | Clip: {max_grad_norm}")
    print(f"{'='*70}\n")
    
    set_seed(seed)
    
    model = ResNet18CIFAR().to(device)
    
    # Opacus requires the model to be in training mode for replacement of layers (e.g. BatchNorm -> GroupNorm)
    # However, ResNet18 uses BatchNorm. Opacus's ModuleValidator can fix this.
    try:
        from opacus.validators import ModuleValidator
        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict=False)
    except ImportError:
        pass # Should be handled by OPACUS_AVAILABLE check later if critical
        
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = get_optimizer(model, optimizer_name, lr)
    
    # --- MAKE PRIVATE ---
    model, optimizer, train_loader, privacy_engine = make_private(
        model, optimizer, train_loader, 
        noise_multiplier=noise_multiplier, 
        max_grad_norm=max_grad_norm,
        epochs=num_epochs
    )
    # --------------------
    
    results = {
        'train_accs': [], 'val_accs': [],
        'train_losses': [], 'val_losses': [],
        'epsilons': [], 'steps': [],
        'agar_values': [], 'confidence_values': []
    }
    
    step = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        epoch_loss = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Log CASMO metrics
            agar, confidence = get_agar_confidence(optimizer)
            if agar is not None and step % 10 == 0:
                results['agar_values'].append(agar)
                results['confidence_values'].append(confidence)
            
            step += 1
            
        train_acc = 100. * correct / total
        train_loss = epoch_loss / len(train_loader)
        
        # Calculate Epsilon
        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        results['epsilons'].append(epsilon)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        val_acc = 100. * correct / total
        val_loss /= len(test_loader)
        
        results['train_accs'].append(train_acc)
        results['val_accs'].append(val_acc)
        results['train_losses'].append(train_loss)
        results['val_losses'].append(val_loss)
        results['steps'].append(step)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Eps: {epsilon:.2f} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Loss: {train_loss:.4f}")
            
    total_time = time.time() - start_time
    print(f"Finished in {total_time:.1f}s")
    return results

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    if not OPACUS_AVAILABLE:
        print("Error: Opacus is not installed. Please install it with 'pip install opacus'.")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20) # Reduced default epochs as DP is slow
    parser.add_argument('--noise', type=float, default=1.0, help='DP Noise Multiplier (sigma)')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient Clipping Norm')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16) # Reduced to 16 for Opacus GPU memory
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    # Opacus requires drop_last=False usually, but for strict DP accounting, we need to be careful. 
    # Opacus handles Poisson sampling if configured, but standard DataLoader is fine for basic epsilon tracking.
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Run Benchmarks
    casmo_res = run_benchmark('casmo', device, trainloader, testloader, 
                             num_epochs=args.epochs, lr=args.lr, 
                             max_grad_norm=args.clip, noise_multiplier=args.noise)
                             
    adam_res = run_benchmark('adamw', device, trainloader, testloader, 
                            num_epochs=args.epochs, lr=args.lr, 
                            max_grad_norm=args.clip, noise_multiplier=args.noise)
                            
    sgd_res = run_benchmark('sgd', device, trainloader, testloader, 
                           num_epochs=args.epochs, lr=0.1, # SGD usually needs higher LR
                           max_grad_norm=args.clip, noise_multiplier=args.noise)
    
    # Plot
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    plt.figure(figsize=(18, 6))
    
    # 1. Accuracy vs Epoch
    plt.subplot(1, 3, 1)
    plt.plot(casmo_res['val_accs'], label='CASMO', color='green', linewidth=2)
    plt.plot(adam_res['val_accs'], label='DP-AdamW', color='orange', linewidth=2)
    plt.plot(sgd_res['val_accs'], label='DP-SGD', color='blue', linewidth=2)
    plt.title(f'Accuracy (Noise={args.noise})')
    plt.xlabel('Epoch')
    plt.ylabel('Val Accuracy (%)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 2. Accuracy vs Epsilon (Privacy-Utility Tradeoff)
    plt.subplot(1, 3, 2)
    plt.plot(casmo_res['epsilons'], casmo_res['val_accs'], label='CASMO', color='green', marker='o')
    plt.plot(adam_res['epsilons'], adam_res['val_accs'], label='DP-AdamW', color='orange', marker='x')
    plt.plot(sgd_res['epsilons'], sgd_res['val_accs'], label='DP-SGD', color='blue', marker='^')
    plt.title('Privacy-Utility Tradeoff')
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('Val Accuracy (%)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 3. Confidence
    plt.subplot(1, 3, 3)
    if casmo_res['confidence_values']:
        # Plot only a subset to avoid clutter
        steps = np.arange(len(casmo_res['confidence_values']))
        plt.plot(steps, casmo_res['confidence_values'], color='green', alpha=0.6, label='CASMO Confidence')
        plt.title('CASMO Confidence Adaptation')
        plt.xlabel('Step (x10)')
        plt.ylabel('Confidence')
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'dp_benchmark_results.png'))
    print(f"Results saved to {results_dir}")

if __name__ == '__main__':
    main()
