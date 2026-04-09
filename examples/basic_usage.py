"""
Basic CASMO Usage Example

This example demonstrates the simplest way to use CASMO as a drop-in
replacement for Adam/AdamW in a standard PyTorch training loop.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import CASMO
from casmo import CASMO


def main():
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    
    # Create dummy dataset
    X = torch.randn(1000, 10)
    y = torch.randint(0, 2, (1000,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize CASMO optimizer (drop-in replacement for AdamW)
    optimizer = CASMO(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        granularity='group'  # Recommended for efficiency
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        # Print epoch results
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
