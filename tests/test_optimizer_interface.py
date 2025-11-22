"""
Unit tests for CASMO optimizer interface and PyTorch compatibility.

Tests verify that CASMO correctly implements the PyTorch Optimizer interface.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import pytest
from casmo import CASMO


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestOptimizerInterface:
    """Test PyTorch Optimizer interface compatibility."""
    
    def test_basic_initialization(self):
        """Test basic optimizer initialization."""
        model = SimpleModel()
        optimizer = CASMO(model.parameters(), lr=1e-3)
        
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]['lr'] == 1e-3
    
    def test_parameter_groups(self):
        """Test multiple parameter groups."""
        model = SimpleModel()
        
        optimizer = CASMO([
            {'params': model.fc1.parameters(), 'lr': 1e-3},
            {'params': model.fc2.parameters(), 'lr': 1e-4}
        ])
        
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]['lr'] == 1e-3
        assert optimizer.param_groups[1]['lr'] == 1e-4
    
    def test_step_and_zero_grad(self):
        """Test basic optimization step and gradient zeroing."""
        model = SimpleModel()
        optimizer = CASMO(model.parameters(), lr=1e-3)
        
        # Forward pass
        x = torch.randn(4, 10)
        y = torch.tensor([0, 1, 0, 1])
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        for p in model.parameters():
            assert p.grad is not None
        
        # Optimization step
        optimizer.step()
        optimizer.zero_grad()
        
        # Check gradients are zeroed
        for p in model.parameters():
            if p.grad is not None:
                assert torch.all(p.grad == 0)
    
    def test_training_loop(self):
        """Test complete training loop."""
        model = SimpleModel()
        optimizer = CASMO(model.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()
        
        # Generate dummy data
        torch.manual_seed(42)
        x_train = torch.randn(20, 10)
        y_train = torch.randint(0, 2, (20,))
        
        initial_loss = None
        final_loss = None
        
        # Training loop
        for epoch in range(10):
            optimizer.zero_grad()
            output = model(x_train)
            loss = criterion(output, y_train)
            
            if epoch == 0:
                initial_loss = loss.item()
            if epoch == 9:
                final_loss = loss.item()
            
            loss.backward()
            optimizer.step()
        
        # Loss should decrease
        assert final_loss < initial_loss, "Loss should decrease during training"
    
    def test_learning_rate_scheduling(self):
        """Test compatibility with PyTorch learning rate schedulers."""
        model = SimpleModel()
        optimizer = CASMO(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Run 10 epochs
        for epoch in range(10):
            # Dummy forward/backward
            x = torch.randn(4, 10)
            y = torch.randint(0, 2, (4,))
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            scheduler.step()
        
        # Learning rate should have decreased
        final_lr = optimizer.param_groups[0]['lr']
        assert final_lr < initial_lr
    
    def test_weight_decay(self):
        """Test weight decay (AdamW-style) functionality."""
        model = SimpleModel()
        optimizer = CASMO(model.parameters(), lr=1e-2, weight_decay=0.01)
        
        # Get initial weights
        initial_weights = {name: p.clone() for name, p in model.named_parameters()}
        
        # Train for a few steps
        for _ in range(10):
            x = torch.randn(4, 10)
            y = torch.randint(0, 2, (4,))
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Weights should have decayed (changed)
        for name, p in model.named_parameters():
            assert not torch.allclose(p, initial_weights[name]), f"Weight {name} should have changed"
    
    def test_closure_support(self):
        """Test optimizer with closure (for compatibility with some advanced optimizers)."""
        model = SimpleModel()
        optimizer = CASMO(model.parameters(), lr=1e-3)
        
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        
        def closure():
            optimizer.zero_grad()
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            return loss
        
        # Step with closure
        loss = optimizer.step(closure)
        assert loss is not None
        assert isinstance(loss.item(), float)


class TestParameterValidation:
    """Test parameter validation and error handling."""
    
    def test_invalid_learning_rate(self):
        """Test that invalid learning rates raise errors."""
        model = SimpleModel()
        
        with pytest.raises(ValueError):
            CASMO(model.parameters(), lr=-0.1)
    
    def test_invalid_betas(self):
        """Test that invalid beta values raise errors."""
        model = SimpleModel()
        
        with pytest.raises(ValueError):
            CASMO(model.parameters(), lr=1e-3, betas=(1.5, 0.999))
        
        with pytest.raises(ValueError):
            CASMO(model.parameters(), lr=1e-3, betas=(0.9, -0.1))
    
    def test_invalid_weight_decay(self):
        """Test that invalid weight decay raises errors."""
        model = SimpleModel()
        
        with pytest.raises(ValueError):
            CASMO(model.parameters(), lr=1e-3, weight_decay=-0.01)
    
    def test_invalid_c_min(self):
        """Test that invalid c_min raises errors."""
        model = SimpleModel()
        
        with pytest.raises(ValueError):
            CASMO(model.parameters(), lr=1e-3, c_min=1.5)
        
        with pytest.raises(ValueError):
            CASMO(model.parameters(), lr=1e-3, c_min=-0.1)
    
    def test_invalid_tau_init_steps(self):
        """Test that tau_init_steps validation works."""
        model = SimpleModel()
        
        with pytest.raises(ValueError):
            CASMO(model.parameters(), lr=1e-3, tau_init_steps=10)  # Too small
    
    def test_invalid_granularity(self):
        """Test that invalid granularity raises errors."""
        model = SimpleModel()
        
        with pytest.raises(ValueError):
            CASMO(model.parameters(), lr=1e-3, granularity='invalid')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
