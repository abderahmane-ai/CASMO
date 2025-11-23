"""
Long-Tailed Recognition Benchmark: CIFAR-100 with Class Imbalance

Tests CASMO's ability to handle imbalanced datasets where standard optimizers
overfit to majority classes and ignore minority (tail) classes.

Task:
    Dataset: CIFAR-100 with exponential long-tail distribution
    Imbalance Ratio: 100:1 (most frequent class has 100x samples of rarest)
    
    Hypothesis:
    - AdamW will overfit to majority classes (high head accuracy, low tail accuracy).
    - CASMO will balance learning across all classes (better tail accuracy).
    
    Reasoning:
    - Majority class gradients: Consistent (high AGAR) → normal learning rate
    - Minority class gradients: Noisy due to few samples (low AGAR) → CASMO prevents overfitting
"""

import sys
import os
import argparse

# Add parent directory to path to import casmo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import time
import random
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from casmo import CASMO


# -----------------------------------------------------------------------------
# 1. Dataset: Long-Tailed CIFAR-100
# -----------------------------------------------------------------------------

class LongTailCIFAR100(Dataset):
    """
    CIFAR-100 with exponential long-tail distribution.
    
    Creates imbalanced dataset where:
    - Head classes (many-shot): >100 samples
    - Medium classes (medium-shot): 20-100 samples  
    - Tail classes (few-shot): <20 samples
    """
    def __init__(self, root='./data', train=True, imbalance_factor=100, seed=42):
        self.train = train
        self.imbalance_factor = imbalance_factor
        
        # Load full CIFAR-100
        self.dataset = torchvision.datasets.CIFAR100(
            root=root, 
            train=train, 
            download=True,
            transform=None  # We'll apply transforms later
        )
        
        if train:
            # Create long-tail distribution
            self.data, self.targets, self.class_counts = self._create_long_tail(seed)
        else:
            # Keep test set balanced
            self.data = self.dataset.data
            self.targets = self.dataset.targets
            self.class_counts = self._count_classes()
        
        # Define transforms
        if train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
    
    def _create_long_tail(self, seed):
        """Create exponentially imbalanced dataset."""
        np.random.seed(seed)
        
        num_classes = 100
        # Get samples per class in original dataset
        original_data = np.array(self.dataset.data)
        original_targets = np.array(self.dataset.targets)
        
        # Calculate target samples per class (exponential decay)
        max_samples = 500  # Maximum samples for most frequent class
        samples_per_class = []
        for i in range(num_classes):
            # Exponential decay: n_i = n_max * (imb_factor)^(-i/(C-1))
            n_samples = int(max_samples * (self.imbalance_factor ** (-i / (num_classes - 1))))
            samples_per_class.append(max(n_samples, 5))  # At least 5 samples per class
        
        # Sample from each class
        new_data = []
        new_targets = []
        class_counts = {}
        
        for class_idx in range(num_classes):
            class_mask = original_targets == class_idx
            class_data = original_data[class_mask]
            
            n_samples = samples_per_class[class_idx]
            n_available = len(class_data)
            
            if n_samples > n_available:
                # Oversample if needed
                indices = np.random.choice(n_available, n_samples, replace=True)
            else:
                # Subsample
                indices = np.random.choice(n_available, n_samples, replace=False)
            
            sampled_data = class_data[indices]
            new_data.append(sampled_data)
            new_targets.extend([class_idx] * n_samples)
            class_counts[class_idx] = n_samples
        
        new_data = np.concatenate(new_data, axis=0)
        new_targets = np.array(new_targets)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(new_targets))
        new_data = new_data[shuffle_idx]
        new_targets = new_targets[shuffle_idx]
        
        return new_data, new_targets.tolist(), class_counts
    
    def _count_classes(self):
        """Count samples per class."""
        class_counts = {}
        for target in self.targets:
            class_counts[target] = class_counts.get(target, 0) + 1
        return class_counts
    
    def get_class_groups(self):
        """Categorize classes into many/medium/few shot."""
        many_shot = []  # >100 samples
        medium_shot = []  # 20-100 samples
        few_shot = []  # <20 samples
        
        for class_idx, count in self.class_counts.items():
            if count > 100:
                many_shot.append(class_idx)
            elif count >= 20:
                medium_shot.append(class_idx)
            else:
                few_shot.append(class_idx)
        
        return many_shot, medium_shot, few_shot
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, target


# -----------------------------------------------------------------------------
# 2. Model: ResNet-32 for CIFAR-100
# -----------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet for CIFAR-100."""
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet32(num_classes=100):
    """ResNet-32 for CIFAR-100."""
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)


# -----------------------------------------------------------------------------
# 3. Utilities
# -----------------------------------------------------------------------------

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_agar_confidence(optimizer):
    """Extract current AGAR and confidence from CASMO optimizer."""
    if not hasattr(optimizer, '_group_states'):
        return None, None
    group_state = optimizer._group_states.get(0, {})
    return group_state.get('current_agar'), group_state.get('current_confidence')


def compute_per_group_accuracy(model, loader, many_shot_classes, medium_shot_classes, few_shot_classes, device):
    """Compute accuracy for many/medium/few shot classes."""
    model.eval()
    
    many_correct = 0
    many_total = 0
    medium_correct = 0
    medium_total = 0
    few_correct = 0
    few_total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            for pred, target in zip(predicted, targets):
                target_item = target.item()
                correct = (pred == target).item()
                
                if target_item in many_shot_classes:
                    many_total += 1
                    many_correct += correct
                elif target_item in medium_shot_classes:
                    medium_total += 1
                    medium_correct += correct
                elif target_item in few_shot_classes:
                    few_total += 1
                    few_correct += correct
    
    many_acc = 100. * many_correct / many_total if many_total > 0 else 0
    medium_acc = 100. * medium_correct / medium_total if medium_total > 0 else 0
    few_acc = 100. * few_correct / few_total if few_total > 0 else 0
    
    return many_acc, medium_acc, few_acc


# -----------------------------------------------------------------------------
# 4. Training Loop
# -----------------------------------------------------------------------------

def run_benchmark(optimizer_name, device, train_loader, test_loader, train_dataset, 
                 num_epochs=200, lr=0.1, weight_decay=5e-4, seed=42):
    
    print(f"\n{'='*70}")
    print(f"Running: {optimizer_name.upper()}")
    print(f"{'='*70}\n")
    
    set_seed(seed)
    
    model = ResNet32(num_classes=100)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'casmo':
        total_steps = len(train_loader) * num_epochs
        tau_init_steps = max(50, int(0.05 * total_steps))
        
        optimizer = CASMO(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            granularity='group',
            log_level=1,
            tau_init_steps=tau_init_steps,
            tau_dead_zone=0.2,
            c_min=0.1
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Get class groups for per-group accuracy
    many_shot, medium_shot, few_shot = train_dataset.get_class_groups()
    
    results = {
        'train_accs': [],
        'val_accs': [],
        'train_losses': [],
        'val_losses': [],
        'many_shot_accs': [],
        'medium_shot_accs': [],
        'few_shot_accs': [],
        'agar_values': [],
        'confidence_values': [],
        'steps': []
    }
    
    step = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Log AGAR and confidence
            agar, confidence = get_agar_confidence(optimizer)
            if agar is not None and step % 10 == 0:
                results['agar_values'].append(agar)
                results['confidence_values'].append(confidence)
                results['steps'].append(step)
            
            step += 1
        
        train_acc = 100. * correct / total
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
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
        
        # Compute per-group accuracy
        many_acc, medium_acc, few_acc = compute_per_group_accuracy(
            model, test_loader, many_shot, medium_shot, few_shot, device
        )
        
        results['train_accs'].append(train_acc)
        results['val_accs'].append(val_acc)
        results['train_losses'].append(train_loss)
        results['val_losses'].append(val_loss)
        results['many_shot_accs'].append(many_acc)
        results['medium_shot_accs'].append(medium_acc)
        results['few_shot_accs'].append(few_acc)
        
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | "
                  f"Many: {many_acc:.1f}% | Medium: {medium_acc:.1f}% | Few: {few_acc:.1f}%")
    
    total_time = time.time() - start_time
    print(f"Finished in {total_time:.1f}s")
    
    # Print final AGAR/Confidence statistics
    if optimizer_name == 'casmo' and results['agar_values']:
        print(f"\n{'='*70}")
        print(f"Final AGAR Statistics (last 100 steps):")
        print(f"{'='*70}")
        agar_final = results['agar_values'][-100:] if len(results['agar_values']) >= 100 else results['agar_values']
        print(f"  Mean: {np.mean(agar_final):.4f}")
        print(f"  Std:  {np.std(agar_final):.4f}")
        print(f"  Min:  {min(agar_final):.4f}")
        print(f"  Max:  {max(agar_final):.4f}")
        
        print(f"\n{'='*70}")
        print(f"Final Confidence Statistics (last 100 steps):")
        print(f"{'='*70}")
        conf_final = results['confidence_values'][-100:] if len(results['confidence_values']) >= 100 else results['confidence_values']
        print(f"  Mean: {np.mean(conf_final):.4f}")
        print(f"  Std:  {np.std(conf_final):.4f}")
        print(f"  Min:  {min(conf_final):.4f}")
        print(f"  Max:  {max(conf_final):.4f}")
        print(f"{'='*70}\n")
    
    return results


# -----------------------------------------------------------------------------
# 5. Main Execution
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Long-Tailed CIFAR-100 Benchmark')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--imbalance', type=int, default=100, help='Imbalance factor (e.g., 100 for 100:1 ratio)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    args = parser.parse_args()
    
    print("="*70)
    print(f"CASMO vs AdamW Benchmark - Long-Tailed CIFAR-100")
    print(f"Imbalance Factor: {args.imbalance}:1")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Setup Data
    train_dataset = LongTailCIFAR100(root='./data', train=True, imbalance_factor=args.imbalance)
    test_dataset = LongTailCIFAR100(root='./data', train=False, imbalance_factor=args.imbalance)
    
    # Print dataset statistics
    many_shot, medium_shot, few_shot = train_dataset.get_class_groups()
    print(f"\nDataset Statistics:")
    print(f"  Total training samples: {len(train_dataset)}")
    print(f"  Many-shot classes (>100 samples): {len(many_shot)}")
    print(f"  Medium-shot classes (20-100 samples): {len(medium_shot)}")
    print(f"  Few-shot classes (<20 samples): {len(few_shot)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Run Benchmarks
    print("\nStarting CASMO run...")
    casmo_results = run_benchmark('casmo', device, train_loader, test_loader, train_dataset,
                                  num_epochs=args.epochs, lr=args.lr, weight_decay=5e-4)
    
    print("\nStarting AdamW run...")
    adamw_results = run_benchmark('adamw', device, train_loader, test_loader, train_dataset,
                                  num_epochs=args.epochs, lr=args.lr, weight_decay=5e-4)
    
    # Plotting
    print("\nPlotting results...")
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Overall Accuracy Curves
    axes[0, 0].plot(casmo_results['train_accs'], label='CASMO Train', linestyle='--', alpha=0.5, color='green')
    axes[0, 0].plot(casmo_results['val_accs'], label='CASMO Val', linewidth=2, color='green')
    axes[0, 0].plot(adamw_results['train_accs'], label='AdamW Train', linestyle='--', alpha=0.5, color='orange')
    axes[0, 0].plot(adamw_results['val_accs'], label='AdamW Val', linewidth=2, color='orange')
    axes[0, 0].set_title(f'Overall Accuracy (Imbalance: {args.imbalance}:1)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Per-Group Accuracy
    epochs = range(len(casmo_results['many_shot_accs']))
    axes[0, 1].plot(epochs, casmo_results['many_shot_accs'], label='CASMO Many-shot', color='green', linestyle='-', linewidth=2)
    axes[0, 1].plot(epochs, casmo_results['medium_shot_accs'], label='CASMO Medium-shot', color='green', linestyle='--', linewidth=2)
    axes[0, 1].plot(epochs, casmo_results['few_shot_accs'], label='CASMO Few-shot', color='green', linestyle=':', linewidth=2)
    axes[0, 1].plot(epochs, adamw_results['many_shot_accs'], label='AdamW Many-shot', color='orange', linestyle='-', linewidth=2, alpha=0.7)
    axes[0, 1].plot(epochs, adamw_results['medium_shot_accs'], label='AdamW Medium-shot', color='orange', linestyle='--', linewidth=2, alpha=0.7)
    axes[0, 1].plot(epochs, adamw_results['few_shot_accs'], label='AdamW Few-shot', color='orange', linestyle=':', linewidth=2, alpha=0.7)
    axes[0, 1].set_title('Per-Group Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. AGAR Evolution (CASMO only)
    if casmo_results['agar_values']:
        axes[1, 0].plot(casmo_results['steps'], casmo_results['agar_values'], color='green', alpha=0.6)
        axes[1, 0].set_title('CASMO: AGAR Signal Strength')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('AGAR')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Confidence Evolution (CASMO only)
    if casmo_results.get('confidence_values'):
        axes[1, 1].plot(casmo_results['steps'], casmo_results['confidence_values'], color='blue', alpha=0.6)
        axes[1, 1].set_title('CASMO: Confidence Score')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Confidence')
        axes[1, 1].set_ylim(0, 1.05)
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'long_tail_metrics.png')
    plt.savefig(save_path, dpi=150)
    print(f"✅ Plot saved to {save_path}")
    
    # Print final comparison
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS COMPARISON")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'CASMO':>15} {'AdamW':>15}")
    print(f"{'-'*70}")
    print(f"{'Overall Val Accuracy':<30} {casmo_results['val_accs'][-1]:>14.2f}% {adamw_results['val_accs'][-1]:>14.2f}%")
    print(f"{'Many-shot Accuracy':<30} {casmo_results['many_shot_accs'][-1]:>14.2f}% {adamw_results['many_shot_accs'][-1]:>14.2f}%")
    print(f"{'Medium-shot Accuracy':<30} {casmo_results['medium_shot_accs'][-1]:>14.2f}% {adamw_results['medium_shot_accs'][-1]:>14.2f}%")
    print(f"{'Few-shot Accuracy':<30} {casmo_results['few_shot_accs'][-1]:>14.2f}% {adamw_results['few_shot_accs'][-1]:>14.2f}%")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
