"""
Grokking Benchmark: Modular Arithmetic with Label Noise
Benchmark ID: B2

Tests CASMO's ability to "grok" (generalize) faster than AdamW on a task 
known for delayed generalization, specifically in the presence of label noise.

Task:
    Dataset: Modular arithmetic (a + b) mod 97
    Model: 1-layer Transformer
    Noise: 30% of training labels are corrupted to random values
    
    Hypothesis:
    - AdamW will memorize the noise (High Train Acc, Low Val Acc)
    - CASMO will ignore the noise (Lower Train Acc, High Val Acc)
"""

import sys
import os
import argparse

# Add parent directory to path to import casmo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from casmo import CASMO


# -----------------------------------------------------------------------------
# 1. Dataset: Modular Arithmetic with Noise
# -----------------------------------------------------------------------------

class NoisyModularDataset(Dataset):
    """
    Dataset for (a + b) mod p with label noise.
    """
    def __init__(self, p=97, split='train', train_fraction=0.5, noise_rate=0.3, seed=42):
        self.p = p
        self.data = []
        self.targets = []
        self.clean_targets = [] # For tracking if we learned the true rule on training set
        self.is_noisy = []
        
        # Generate all possible pairs
        all_pairs = []
        for i in range(p):
            for j in range(p):
                all_pairs.append((i, j))
        
        # Shuffle and split
        random.seed(seed)
        random.shuffle(all_pairs)
        
        num_train = int(len(all_pairs) * train_fraction)
        
        if split == 'train':
            self.pairs = all_pairs[:num_train]
            # Apply noise only to training set
            np.random.seed(seed)
            for a, b in self.pairs:
                true_target = (a + b) % p
                self.data.append(torch.tensor([a, b, self.p]))
                self.clean_targets.append(true_target)
                
                if np.random.random() < noise_rate:
                    # Corrupt label
                    noisy_target = np.random.randint(0, p)
                    # Ensure it's actually different
                    while noisy_target == true_target:
                        noisy_target = np.random.randint(0, p)
                    
                    self.targets.append(noisy_target)
                    self.is_noisy.append(True)
                else:
                    self.targets.append(true_target)
                    self.is_noisy.append(False)
                    
            print(f"Train Set: {len(self.data)} samples. Noise Rate: {noise_rate:.2f}")
            print(f"  Clean: {len(self.data) - sum(self.is_noisy)}")
            print(f"  Noisy: {sum(self.is_noisy)}")
            
        else:
            self.pairs = all_pairs[num_train:]
            for a, b in self.pairs:
                self.data.append(torch.tensor([a, b, self.p]))
                self.targets.append((a + b) % p)
                self.clean_targets.append((a + b) % p)
                self.is_noisy.append(False)
            print(f"Test Set: {len(self.data)} samples (All clean)")
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# -----------------------------------------------------------------------------
# 2. Model: Simple Transformer
# -----------------------------------------------------------------------------

class Transformer(nn.Module):
    """
    Simple 1-layer Transformer for Grokking.
    """
    def __init__(self, vocab_size, d_model=128, num_heads=4, d_ff=512, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)
        
        self.transformer_block = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=d_ff, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.transformer_block, num_layers=1)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        emb = self.embedding(x) + self.pos_encoding
        out = self.transformer(emb)
        last_token_out = out[:, -1, :] 
        logits = self.fc_out(last_token_out)
        return logits


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

# -----------------------------------------------------------------------------
# 4. Training Loop
# -----------------------------------------------------------------------------

def run_benchmark(optimizer_name, device, train_loader, test_loader, num_epochs=1000,
                 lr=1e-3, weight_decay=1.0, seed=42):
    
    print(f"\n{'='*70}")
    print(f"Running: {optimizer_name.upper()}")
    print(f"{'='*70}\n")
    
    set_seed(seed)
    
    # Model parameters
    P = 97
    VOCAB_SIZE = P + 1
    
    model = Transformer(vocab_size=VOCAB_SIZE, d_model=128, num_heads=4, d_ff=512, dropout=0.0)
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
        
    results = {
        'train_accs': [],
        'val_accs': [],
        'train_losses': [],
        'val_losses': [],
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
        
        results['train_accs'].append(train_acc)
        results['val_accs'].append(val_acc)
        results['train_losses'].append(train_loss)
        results['val_losses'].append(val_loss)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}% | Loss: {train_loss:.4f}")
            
    total_time = time.time() - start_time
    print(f"Finished in {total_time:.1f}s")
    
    return results


# -----------------------------------------------------------------------------
# 5. Main Execution
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Noisy Grokking Benchmark')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--noise', type=float, default=0.3, help='Label noise rate (0.0-1.0)')
    args = parser.parse_args()

    print("="*70)
    print(f"CASMO vs AdamW Benchmark - Noisy Grokking (Noise: {args.noise*100:.0f}%)")
    print("Task: (a + b) mod 97")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Setup Data
    P = 97
    train_dataset = NoisyModularDataset(p=P, split='train', train_fraction=0.5, noise_rate=args.noise)
    test_dataset = NoisyModularDataset(p=P, split='test', train_fraction=0.5)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    # Run Benchmarks
    EPOCHS = args.epochs
    
    print("\nStarting CASMO run...")
    casmo_results = run_benchmark('casmo', device, train_loader, test_loader, num_epochs=EPOCHS, lr=1e-3, weight_decay=1.0)
    
    print("\nStarting AdamW run...")
    adamw_results = run_benchmark('adamw', device, train_loader, test_loader, num_epochs=EPOCHS, lr=1e-3, weight_decay=1.0)
    
    # Plotting
    print("\nPlotting results...")
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Accuracy Curves
    axes[0, 0].plot(casmo_results['train_accs'], label='CASMO Train', linestyle='--', alpha=0.5, color='green')
    axes[0, 0].plot(casmo_results['val_accs'], label='CASMO Val', linewidth=2, color='green')
    axes[0, 0].plot(adamw_results['train_accs'], label='AdamW Train', linestyle='--', alpha=0.5, color='orange')
    axes[0, 0].plot(adamw_results['val_accs'], label='AdamW Val', linewidth=2, color='orange')
    
    # Expected max train acc = (1 - noise_rate) * 100 + noise_rate * (1/P) * 100 approx (1-noise)*100
    expected_max_train = (1.0 - args.noise) * 100
    axes[0, 0].axhline(y=expected_max_train, color='gray', linestyle=':', label=f'Max Clean Acc ({expected_max_train:.0f}%)')
    
    axes[0, 0].set_title(f'Accuracy (Noise: {args.noise*100:.0f}%)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Loss Curves
    axes[0, 1].plot(casmo_results['train_losses'], label='CASMO Train', color='green')
    axes[0, 1].plot(adamw_results['train_losses'], label='AdamW Train', color='orange')
    axes[0, 1].set_title('Training Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Log Loss')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
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
    save_path = os.path.join(results_dir, 'noisy_grokking_metrics.png')
    plt.savefig(save_path)
    print(f"✅ Plot saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    main()
