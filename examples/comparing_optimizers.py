"""
Comparing CASMO vs AdamW

This example demonstrates the difference between CASMO and AdamW
on a simple task with noisy labels.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from casmo import CASMO


def create_noisy_dataset(n_samples=1000, noise_rate=0.3):
    """Create a binary classification dataset with label noise."""
    # Generate linearly separable data
    X = torch.randn(n_samples, 10)
    y_true = (X[:, 0] + X[:, 1] > 0).long()
    
    # Add label noise
    n_noisy = int(n_samples * noise_rate)
    noisy_indices = torch.randperm(n_samples)[:n_noisy]
    y_noisy = y_true.clone()
    y_noisy[noisy_indices] = 1 - y_noisy[noisy_indices]
    
    return X, y_noisy, y_true


def train_model(optimizer_name, X_train, y_train, X_test, y_test, num_epochs=50):
    """Train a model with specified optimizer."""
    # Create model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    
    # Create optimizer
    if optimizer_name == 'casmo':
        optimizer = CASMO(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.01,
            granularity='group'
        )
    else:  # adamw
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.01
        )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    train_losses = []
    test_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        
        # Evaluate on clean test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = test_outputs.max(1)
            test_acc = (predicted == y_test).float().mean().item() * 100
            test_accs.append(test_acc)
    
    return train_losses, test_accs


def main():
    print("Comparing CASMO vs AdamW on Noisy Labels\n")
    
    # Create datasets
    print("Creating dataset with 30% label noise...")
    X_train, y_train_noisy, y_train_clean = create_noisy_dataset(1000, noise_rate=0.3)
    X_test, y_test, _ = create_noisy_dataset(200, noise_rate=0.0)  # Clean test set
    
    # Train with both optimizers
    print("\nTraining with AdamW...")
    adamw_losses, adamw_accs = train_model('adamw', X_train, y_train_noisy, X_test, y_test)
    
    print("Training with CASMO...")
    casmo_losses, casmo_accs = train_model('casmo', X_train, y_train_noisy, X_test, y_test)
    
    # Print results
    print(f"\nFinal Results:")
    print(f"AdamW  - Test Accuracy: {adamw_accs[-1]:.2f}%")
    print(f"CASMO  - Test Accuracy: {casmo_accs[-1]:.2f}%")
    print(f"Improvement: {casmo_accs[-1] - adamw_accs[-1]:+.2f}%")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Training loss
    ax1.plot(adamw_losses, label='AdamW', color='orange')
    ax1.plot(casmo_losses, label='CASMO', color='green')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss (30% Label Noise)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Test accuracy
    ax2.plot(adamw_accs, label='AdamW', color='orange')
    ax2.plot(casmo_accs, label='CASMO', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Test Accuracy (Clean Labels)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=150)
    print("\n✓ Plot saved to optimizer_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
