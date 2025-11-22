"""
ResNet-32 Noisy CIFAR-10 Benchmark with Checkpoint Support

Tests CASMO's ability to handle 40% label noise by detecting gradient quality.
Uses the new universal sigmoid-based confidence mapping.
"""

import sys
import os

# Add parent directory to path to import casmo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from casmo import CASMO


# ResNet-32 for CIFAR-10 (standard architecture for noisy label research)
class BasicBlock(nn.Module):
    """Basic residual block for ResNet-32."""
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
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet for CIFAR-10."""
    def __init__(self, block, num_blocks, num_classes=10):
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
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.nn.functional.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet32(num_classes=10):
    """ResNet-32 for CIFAR-10 (5+5+5 blocks = 32 layers)."""
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NoisyCIFAR10(Dataset):
    """CIFAR-10 with symmetric label noise."""

    def __init__(self, base_dataset, noise_rate=0.4, seed=42):
        self.base_dataset = base_dataset
        self.noise_rate = noise_rate

        # Create noisy labels
        np.random.seed(seed)
        self.noisy_labels = []
        self.is_clean = []

        for idx in range(len(base_dataset)):
            _, true_label = base_dataset[idx]
            if np.random.random() < noise_rate:
                # Corrupt label: random class (excluding true class)
                noisy_label = np.random.choice([i for i in range(10) if i != true_label])
                self.noisy_labels.append(noisy_label)
                self.is_clean.append(False)
            else:
                # Keep clean label
                self.noisy_labels.append(true_label)
                self.is_clean.append(True)

        clean_count = sum(self.is_clean)
        print(f"Dataset: {len(self)} samples, {clean_count} clean ({100*clean_count/len(self):.1f}%), "
              f"{len(self)-clean_count} noisy ({100*(len(self)-clean_count)/len(self):.1f}%)")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, _ = self.base_dataset[idx]
        noisy_label = self.noisy_labels[idx]
        return image, noisy_label


def get_agar_confidence(optimizer):
    """Extract current AGAR and confidence from CASMO optimizer."""
    if not hasattr(optimizer, '_group_states'):
        return None, None
    group_state = optimizer._group_states.get(0, {})
    return group_state.get('current_agar'), group_state.get('current_confidence')


def get_distribution_stats(optimizer):
    """Extract distribution statistics from CASMO optimizer."""
    if not hasattr(optimizer, '_group_states'):
        return None, None, None
    group_state = optimizer._group_states.get(0, {})
    return (group_state.get('agar_mean'),
            group_state.get('agar_std'),
            group_state.get('c_min'))


def save_checkpoint(checkpoint_path, epoch, model, optimizer, scheduler, results, best_acc):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'results': results,
        'best_acc': best_acc,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"  💾 Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """Load training checkpoint."""
    if not os.path.exists(checkpoint_path):
        return None, None, None

    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    results = checkpoint['results']
    best_acc = checkpoint['best_acc']

    print(f"✅ Resumed from epoch {checkpoint['epoch']}, best_acc={best_acc:.2f}%\n")
    return start_epoch, results, best_acc


def run_benchmark(optimizer_name, device, train_loader, test_loader, num_epochs=100,
                 lr=1e-3, weight_decay=5e-4, seed=42, checkpoint_dir='./checkpoints',
                 checkpoint_freq=10, resume=True):
    """Run training benchmark for one optimizer."""

    print(f"\n{'='*70}")
    print(f"Running: {optimizer_name.upper()}")
    print(f"{'='*70}\n")

    set_seed(seed)

    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'{optimizer_name}_checkpoint.pth')

    # Load ResNet-32 (standard for CIFAR-10 noisy label research)
    model = resnet32(num_classes=10)
    print(f"Using ResNet-32 (standard for noisy label research)")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Create optimizer
    if optimizer_name == 'casmo':
        total_steps = len(train_loader) * num_epochs
        tau_init_steps = max(100, int(0.05 * total_steps))  # 5% calibration, min 100
        optimizer = CASMO(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            granularity='group',
            log_level=2,
            tau_init_steps=tau_init_steps,
            tau_dead_zone=1.0  # Frozen after calibration
        )
        print(f"CASMO tau_init_steps: {tau_init_steps}")
        print(f"Using universal sigmoid-based confidence mapping")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Initialize results
    results = {
        'optimizer': optimizer_name,
        'train_losses': [],
        'train_accs': [],
        'test_losses': [],
        'test_accs': [],
        'agar_values': [],
        'confidence_values': [],
        'agar_per_epoch': [],
        'calibration_samples': [],
        'calibration_mu': None,
        'calibration_sigma': None,
        'calibration_c_min': None,
    }

    start_epoch = 0
    best_acc = 0
    start_time = time.time()

    # Try to resume from checkpoint
    if resume and os.path.exists(checkpoint_path):
        loaded = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
        if loaded[0] is not None:
            start_epoch, results, best_acc = loaded
            if start_epoch >= num_epochs:
                print(f"⚠️  Training already complete (epoch {start_epoch}/{num_epochs})")
                return results

    try:
        for epoch in range(start_epoch, num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # Training
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            epoch_agars = []

            for batch_idx, (inputs, targets) in enumerate(train_loader):
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

                # Track AGAR/confidence
                agar, conf = get_agar_confidence(optimizer)
                if agar is not None:
                    results['agar_values'].append(agar)
                    results['confidence_values'].append(conf)
                    epoch_agars.append(agar)

                # Capture calibration data (once)
                if optimizer_name == 'casmo' and results['calibration_mu'] is None:
                    group_state = optimizer._group_states.get(0, {})
                    if group_state.get('tau_initialized', False):
                        results['calibration_mu'] = group_state.get('agar_mean')
                        results['calibration_sigma'] = group_state.get('agar_std')
                        results['calibration_c_min'] = group_state.get('c_min')
                    elif len(group_state.get('agar_buffer', [])) > 0:
                        results['calibration_samples'] = list(group_state['agar_buffer'])

                if (batch_idx + 1) % 100 == 0:
                    msg = f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%"
                    if agar is not None:
                        msg += f", AGAR: {agar:.4f}, Conf: {conf:.4f}"
                    print(msg)

            if epoch_agars:
                results['agar_per_epoch'].append({
                    'epoch': epoch + 1,
                    'mean': np.mean(epoch_agars),
                    'std': np.std(epoch_agars),
                    'min': np.min(epoch_agars),
                    'max': np.max(epoch_agars)
                })

            train_loss /= len(train_loader)
            train_acc = 100. * correct / total
            results['train_losses'].append(train_loss)
            results['train_accs'].append(train_acc)

            # Testing (on clean test set)
            model.eval()
            test_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            test_loss /= len(test_loader)
            test_acc = 100. * correct / total
            results['test_losses'].append(test_loss)
            results['test_accs'].append(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc

            scheduler.step()

            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"  Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%, Best={best_acc:.2f}%")

            # Print AGAR and distribution stats for CASMO
            if epoch_agars and (epoch + 1) % 10 == 0:
                mu, sigma, c_min = get_distribution_stats(optimizer)
                print(f"  AGAR: mean={np.mean(epoch_agars):.4f}, std={np.std(epoch_agars):.4f}, "
                      f"range=[{np.min(epoch_agars):.4f}, {np.max(epoch_agars):.4f}]")
                if mu is not None:
                    print(f"  Distribution: μ={mu:.4f}, σ={sigma:.4f}, c_min={c_min:.2f}")

            # Save checkpoint periodically
            if (epoch + 1) % checkpoint_freq == 0 or (epoch + 1) == num_epochs:
                save_checkpoint(checkpoint_path, epoch, model, optimizer, scheduler, results, best_acc)

            print()

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted! Saving checkpoint...")
        save_checkpoint(checkpoint_path, epoch, model, optimizer, scheduler, results, best_acc)
        print("✅ Checkpoint saved. You can resume training later.")
        raise

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("Saving checkpoint before exit...")
        save_checkpoint(checkpoint_path, epoch, model, optimizer, scheduler, results, best_acc)
        raise

    results['best_acc'] = best_acc
    results['total_time'] = time.time() - start_time

    print(f"Complete: {results['total_time']:.1f}s ({results['total_time']/60:.1f} min)")
    print(f"Best Test Accuracy: {best_acc:.2f}%\n")

    # Print final calibration info for CASMO
    if optimizer_name == 'casmo' and results['calibration_mu'] is not None:
        print(f"Final CASMO calibration:")
        print(f"  μ (mean AGAR): {results['calibration_mu']:.4f}")
        print(f"  σ (std AGAR): {results['calibration_sigma']:.4f}")
        print(f"  c_min (adaptive): {results['calibration_c_min']:.2f}")
        cv = results['calibration_sigma'] / (results['calibration_mu'] + 1e-8)
        print(f"  CV (coefficient of variation): {cv:.4f}")
        if cv < 0.3:
            print(f"  → Low variance detected: Using high c_min for pervasive noise")
        elif cv < 0.5:
            print(f"  → Medium variance: Moderate discrimination")
        else:
            print(f"  → High variance: Strong discrimination for bimodal distribution")
        print()

    # Clean up checkpoint after successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"🗑️  Removed checkpoint (training complete)\n")

    return results


def main():
    """Main benchmark execution."""
    print("="*70)
    print("CASMO vs AdamW Benchmark - Noisy CIFAR-10 (40% Label Corruption)")
    print("Architecture: ResNet-32 (standard for noisy label research)")
    print("="*70)

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Data loading
    print("\nLoading CIFAR-10 dataset...")
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load base datasets
    base_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # Create noisy training set (40% label corruption)
    print("\nCreating noisy training set (40% label corruption)...")
    noisy_train = NoisyCIFAR10(base_train, noise_rate=0.4, seed=42)

    train_loader = DataLoader(noisy_train, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    print(f"\nTrain: {len(noisy_train)} (noisy), Test: {len(test_dataset)} (clean)")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"\n⚠️  Training on NOISY labels, testing on CLEAN labels")
    print(f"This tests the optimizer's ability to learn despite label noise.")
    print(f"\n💾 Checkpoints will be saved every 10 epochs\n")

    # Run benchmarks (100 epochs)
    casmo_results = run_benchmark('casmo', device, train_loader, test_loader,
                                  num_epochs=100, seed=42, checkpoint_freq=10, resume=True)
    adamw_results = run_benchmark('adamw', device, train_loader, test_loader,
                                  num_epochs=100, seed=42, checkpoint_freq=10, resume=True)

    # Comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)

    casmo_acc = casmo_results['best_acc']
    adamw_acc = adamw_results['best_acc']
    acc_diff = (casmo_acc - adamw_acc) / adamw_acc * 100

    print(f"\nBest Test Accuracy (on clean labels):")
    print(f"  CASMO:  {casmo_acc:.2f}%")
    print(f"  AdamW:  {adamw_acc:.2f}%")
    print(f"  Diff:   {acc_diff:+.2f}% {'(CASMO wins!)' if acc_diff > 0 else '(AdamW wins)'}")

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    epochs = list(range(1, len(casmo_results['test_accs']) + 1))

    # Training loss
    axes[0, 0].plot(epochs, casmo_results['train_losses'], label='CASMO', alpha=0.7)
    axes[0, 0].plot(epochs, adamw_results['train_losses'], label='AdamW', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss (on noisy labels)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Test accuracy
    axes[0, 1].plot(epochs, casmo_results['test_accs'], 'o-', label='CASMO', linewidth=2, markersize=3)
    axes[0, 1].plot(epochs, adamw_results['test_accs'], 's-', label='AdamW', linewidth=2, markersize=3)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Test Accuracy (on clean labels)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Training accuracy
    axes[0, 2].plot(epochs, casmo_results['train_accs'], label='CASMO', alpha=0.7)
    axes[0, 2].plot(epochs, adamw_results['train_accs'], label='AdamW', alpha=0.7)
    axes[0, 2].axhline(y=60, color='r', linestyle='--', alpha=0.5, label='60% (expected with 40% noise)')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy (%)')
    axes[0, 2].set_title('Training Accuracy (on noisy labels)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # AGAR evolution
    if casmo_results['agar_values']:
        axes[1, 0].plot(casmo_results['agar_values'], color='green', alpha=0.5)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('AGAR')
        axes[1, 0].set_title('CASMO: AGAR Evolution')
        axes[1, 0].grid(True, alpha=0.3)

        # Add calibration line if available
        if casmo_results['calibration_mu'] is not None:
            axes[1, 0].axhline(y=casmo_results['calibration_mu'], color='red',
                              linestyle='--', alpha=0.7, label=f"μ={casmo_results['calibration_mu']:.4f}")
            axes[1, 0].legend()

    # Confidence evolution
    if casmo_results['confidence_values']:
        axes[1, 1].plot(casmo_results['confidence_values'], color='blue', alpha=0.5)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Confidence')
        axes[1, 1].set_title('CASMO: Confidence Evolution (Sigmoid Mapping)')
        axes[1, 1].set_ylim([0, 1.0])
        axes[1, 1].grid(True, alpha=0.3)

        # Add c_min line if available
        if casmo_results['calibration_c_min'] is not None:
            axes[1, 1].axhline(y=casmo_results['calibration_c_min'], color='red',
                              linestyle='--', alpha=0.7, label=f"c_min={casmo_results['calibration_c_min']:.2f}")
            axes[1, 1].legend()

    # AGAR statistics per epoch
    if casmo_results['agar_per_epoch']:
        agar_epochs = [x['epoch'] for x in casmo_results['agar_per_epoch']]
        agar_means = [x['mean'] for x in casmo_results['agar_per_epoch']]
        agar_stds = [x['std'] for x in casmo_results['agar_per_epoch']]
        axes[1, 2].plot(agar_epochs, agar_means, 'o-', label='Mean AGAR', linewidth=2)
        axes[1, 2].fill_between(
            agar_epochs,
            [m - s for m, s in zip(agar_means, agar_stds)],
            [m + s for m, s in zip(agar_means, agar_stds)],
            alpha=0.3, label='±1 std'
        )
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('AGAR')
        axes[1, 2].set_title('CASMO: AGAR Statistics per Epoch')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('noisy_cifar10_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved: noisy_cifar10_comparison.png")
    plt.show()

    print("\n✅ Benchmark complete!")


if __name__ == '__main__':
    main()