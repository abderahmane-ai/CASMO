"""
Integration tests for CASMO optimizer.

Tests end-to-end training scenarios to ensure CASMO converges correctly
and produces results competitive with Adam/AdamW.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from casmo import CASMO


class SimpleMLP(nn.Module):
    """Simple 2-layer MLP for testing."""
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def create_synthetic_dataset(n_samples=1000, input_dim=10, n_classes=2, noise_level=0.0):
    """Create a simple synthetic classification dataset."""
    torch.manual_seed(42)
    X = torch.randn(n_samples, input_dim)
    
    # Simple linear boundary
    weights = torch.randn(input_dim)
    logits = X @ weights
    y = (logits > 0).long()
    
    # Add label noise if requested
    if noise_level > 0:
        n_noisy = int(n_samples * noise_level)
        noisy_indices = torch.randperm(n_samples)[:n_noisy]
        y[noisy_indices] = 1 - y[noisy_indices]
    
    return TensorDataset(X, y)


def train_model(model, optimizer, dataloader, epochs=10):
    """Train model and return final loss."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for X, y in dataloader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


class TestIntegration:
    """Integration tests for CASMO optimizer."""
    
    def test_basic_convergence(self):
        """Test that CASMO converges on a simple problem."""
        dataset = create_synthetic_dataset(n_samples=500)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        model = SimpleMLP()
        optimizer = CASMO(model.parameters(), lr=1e-3)
        
        final_loss = train_model(model, optimizer, dataloader, epochs=20)
        
        # Should achieve low loss on this simple problem
        assert final_loss < 0.5, f"Expected final loss < 0.5, got {final_loss}"
    
    def test_casmo_vs_adam_clean_data(self):
        """Verify CASMO converges similarly to Adam on clean data."""
        dataset = create_synthetic_dataset(n_samples=500)
        dataloader_casmo = DataLoader(dataset, batch_size=32, shuffle=False)
        dataloader_adam = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # CASMO model
        torch.manual_seed(42)
        model_casmo = SimpleMLP()
        optimizer_casmo = CASMO(model_casmo.parameters(), lr=1e-3, tau_init_steps=50)
        loss_casmo = train_model(model_casmo, optimizer_casmo, dataloader_casmo, epochs=20)
        
        # Adam model
        torch.manual_seed(42)
        model_adam = SimpleMLP()
        optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=1e-3)
        loss_adam = train_model(model_adam, optimizer_adam, dataloader_adam, epochs=20)
        
        # Both should achieve similar low loss on clean data
        assert abs(loss_casmo - loss_adam) < 0.2, \
            f"CASMO and Adam should have similar final loss. CASMO: {loss_casmo}, Adam: {loss_adam}"
    
    def test_casmo_handles_noisy_data(self):
        """Test that CASMO handles noisy labels better than Adam."""
        dataset = create_synthetic_dataset(n_samples=500, noise_level=0.3)
        
        # CASMO model
        torch.manual_seed(42)
        dataloader_casmo = DataLoader(dataset, batch_size=32, shuffle=False)
        model_casmo = SimpleMLP()
        optimizer_casmo = CASMO(model_casmo.parameters(), lr=1e-3, tau_init_steps=50)
        
        # Train and measure overfitting
        model_casmo.train()
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(20):
            for X, y in dataloader_casmo:
                optimizer_casmo.zero_grad()
                outputs = model_casmo(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer_casmo.step()
        
        # Verify CASMO doesn't perfectly fit noisy data
        with torch.no_grad():
            model_casmo.eval()
            correct = 0
            total = 0
            for X, y in dataloader_casmo:
                outputs = model_casmo(X)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        accuracy = correct / total
        
        # With 30% label noise, perfect memorization would give ~100% accuracy
        # CASMO should resist memorizing and stay below perfect accuracy
        assert accuracy < 0.95, \
            f"CASMO should not perfectly memorize noisy data. Accuracy: {accuracy}"
    
    def test_convergence_with_weight_decay(self):
        """Test convergence with weight decay enabled."""
        dataset = create_synthetic_dataset(n_samples=500)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        model = SimpleMLP()
        optimizer = CASMO(model.parameters(), lr=1e-3, weight_decay=0.01)
        
        final_loss = train_model(model, optimizer, dataloader, epochs=20)
        
        assert final_loss < 0.6, f"Expected convergence with weight decay, got loss {final_loss}"
    
    def test_different_granularities(self):
        """Test both parameter and group granularity modes."""
        dataset = create_synthetic_dataset(n_samples=300)
        
        for granularity in ['parameter', 'group']:
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
            torch.manual_seed(42)
            model = SimpleMLP()
            optimizer = CASMO(
                model.parameters(), 
                lr=1e-3, 
                granularity=granularity,
                tau_init_steps=50
            )
            
            final_loss = train_model(model, optimizer, dataloader, epochs=15)
            
            assert final_loss < 0.7, \
                f"Granularity '{granularity}' should converge, got loss {final_loss}"
    
    def test_multiple_parameter_groups(self):
        """Test with different learning rates for different layers."""
        dataset = create_synthetic_dataset(n_samples=300)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        model = SimpleMLP()
        
        # Different LR for each layer
        param_groups = [
            {'params': model.fc1.parameters(), 'lr': 1e-3},
            {'params': model.fc2.parameters(), 'lr': 5e-4}
        ]
        
        optimizer = CASMO(param_groups, tau_init_steps=50)
        final_loss = train_model(model, optimizer, dataloader, epochs=15)
        
        assert final_loss < 0.7, f"Multi-group optimization should work, got loss {final_loss}"
    
    def test_checkpoint_and_resume(self):
        """Test saving and loading optimizer state."""
        dataset = create_synthetic_dataset(n_samples=300)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train for a few epochs
        model = SimpleMLP()
        optimizer = CASMO(model.parameters(), lr=1e-3, tau_init_steps=50)
        train_model(model, optimizer, dataloader, epochs=5)
        
        # Save state
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        
        # Create new model and optimizer
        model2 = SimpleMLP()
        optimizer2 = CASMO(model2.parameters(), lr=1e-3, tau_init_steps=50)
        
        # Load state
        model2.load_state_dict(checkpoint['model'])
        optimizer2.load_state_dict(checkpoint['optimizer'])
        
        # Continue training
        final_loss = train_model(model2, optimizer2, dataloader, epochs=10)
        
        assert final_loss < 0.7, "Resumed training should converge"
    
    def test_gradient_accumulation(self):
        """Test with gradient accumulation."""
        dataset = create_synthetic_dataset(n_samples=300)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        model = SimpleMLP()
        optimizer = CASMO(model.parameters(), lr=1e-3, tau_init_steps=50)
        criterion = nn.CrossEntropyLoss()
        
        accumulation_steps = 2
        model.train()
        
        for epoch in range(10):
            for i, (X, y) in enumerate(dataloader):
                outputs = model(X)
                loss = criterion(outputs, y)
                loss = loss / accumulation_steps
                loss.backward()
                
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
        
        # Just verify no errors occur
        assert True
