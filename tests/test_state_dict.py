"""
Unit tests for CASMO state dict save/load functionality.

Tests verify checkpoint saving and loading for training resumption.
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
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


class TestStateDictSaveLoad:
    """Test state dict save/load functionality."""
    
    def test_save_and_load_state_dict(self):
        """Test basic state dict save and load."""
        model = SimpleModel()
        optimizer = CASMO(model.parameters(), lr=1e-3)
        
        # Run some optimization steps
        for _ in range(20):
            x = torch.randn(4, 10)
            y = torch.randint(0, 2, (4,))
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Save state
        state_dict = optimizer.state_dict()
        
        # Create new optimizer and load state
        new_optimizer = CASMO(model.parameters(), lr=1e-3)
        new_optimizer.load_state_dict(state_dict)
        
        # Verify state matches
        assert new_optimizer._step_count == optimizer._step_count
    
    def test_state_dict_contains_moments(self):
        """Test that state dict contains exp_avg and exp_avg_sq."""
        model = SimpleModel()
        optimizer = CASMO(model.parameters(), lr=1e-3)
        
        # Run optimization
        for _ in range(10):
            x = torch.randn(4, 10)
            y = torch.randint(0, 2, (4,))
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        state_dict = optimizer.state_dict()
        
        # Check that state contains parameter states
        assert 'state' in state_dict
        assert len(state_dict['state']) > 0
        
        # Check first parameter state
        first_param_state = list(state_dict['state'].values())[0]
        assert 'exp_avg' in first_param_state
        assert 'exp_avg_sq' in first_param_state
        assert 'step' in first_param_state
    
    def test_state_dict_contains_group_states(self):
        """Test that state dict contains CASMO-specific group states."""
        model = SimpleModel()
        optimizer = CASMO(model.parameters(), lr=1e-3, tau_init_steps=50)
        
        # Run enough steps to calibrate tau
        for _ in range(60):
            x = torch.randn(4, 10)
            y = torch.randint(0, 2, (4,))
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        state_dict = optimizer.state_dict()
        
        # Check CASMO-specific state
        assert '_group_states' in state_dict
        assert '_step_count' in state_dict
    
    def test_resume_training_continuity(self):
        """Test that training can be resumed from checkpoint."""
        torch.manual_seed(42)
        
        # First training session
        model1 = SimpleModel()
        optimizer1 = CASMO(model1.parameters(), lr=1e-2)
        
        x = torch.randn(20, 10)
        y = torch.randint(0, 2, (20,))
        
        # Train for 10 steps
        for _ in range(10):
            optimizer1.zero_grad()
            loss = nn.CrossEntropyLoss()(model1(x), y)
            loss.backward()
            optimizer1.step()
        
        # Save checkpoint
        checkpoint = {
            'model_state_dict': model1.state_dict(),
            'optimizer_state_dict': optimizer1.state_dict(),
        }
        
        # Second training session (resumed)
        model2 = SimpleModel()
        optimizer2 = CASMO(model2.parameters(), lr=1e-2)
        model2.load_state_dict(checkpoint['model_state_dict'])
        optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Continue training for 10 more steps
        for _ in range(10):
            optimizer2.zero_grad()
            loss = nn.CrossEntropyLoss()(model2(x), y)
            loss.backward()
            optimizer2.step()
        
        # Third training session (continuous)
        torch.manual_seed(42)
        model3 = SimpleModel()
        optimizer3 = CASMO(model3.parameters(), lr=1e-2)
        
        # Train for 20 steps continuously
        for _ in range(20):
            optimizer3.zero_grad()
            loss = nn.CrossEntropyLoss()(model3(x), y)
            loss.backward()
            optimizer3.step()
        
        # Model2 (resumed) should match Model3 (continuous)
        for p2, p3 in zip(model2.parameters(), model3.parameters()):
            assert torch.allclose(p2, p3, atol=1e-6), "Resumed training should match continuous training"
    
    def test_load_with_different_hyperparameters(self):
        """Test loading state dict with different hyperparameters."""
        model = SimpleModel()
        
        # Create optimizer with certain hyperparameters
        optimizer1 = CASMO(model.parameters(), lr=1e-3, weight_decay=0.01)
        
        # Run some steps
        for _ in range(20):
            x = torch.randn(4, 10)
            y = torch.randint(0, 2, (4,))
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            optimizer1.step()
            optimizer1.zero_grad()
        
        state_dict = optimizer1.state_dict()
        
        # Create optimizer with different hyperparameters
        optimizer2 = CASMO(model.parameters(), lr=1e-2, weight_decay=0.001)
        
        # Load state (should work, hyperparameters are separate)
        optimizer2.load_state_dict(state_dict)
        
        # Step counter should be preserved
        assert optimizer2._step_count == optimizer1._step_count
    
    def test_tau_calibration_state_preserved(self):
        """Test that tau calibration state is preserved across save/load."""
        model = SimpleModel()
        optimizer1 = CASMO(model.parameters(), lr=1e-3, tau_init_steps=50)
        
        # Run enough steps to calibrate tau
        for _ in range(60):
            x = torch.randn(4, 10)
            y = torch.randint(0, 2, (4,))
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            optimizer1.step()
            optimizer1.zero_grad()
        
        # Check tau is calibrated
        assert optimizer1._group_states[0]['tau_initialized'] == True
        tau_value = optimizer1._group_states[0]['tau_adapter'].tau
        
        # Save and load
        state_dict = optimizer1.state_dict()
        optimizer2 = CASMO(model.parameters(), lr=1e-3, tau_init_steps=50)
        optimizer2.load_state_dict(state_dict)
        
        # Verify tau calibration is preserved
        assert optimizer2._group_states[0]['tau_initialized'] == True
        assert optimizer2._group_states[0]['tau_adapter'].tau == tau_value


class TestMultipleParameterGroups:
    """Test state dict with multiple parameter groups."""
    
    def test_save_load_multiple_groups(self):
        """Test state dict with multiple parameter groups."""
        model = SimpleModel()
        
        optimizer1 = CASMO([
            {'params': model.fc.weight, 'lr': 1e-3},
            {'params': model.fc.bias, 'lr': 1e-4}
        ])
        
        # Run optimization
        for _ in range(20):
            x = torch.randn(4, 10)
            y = torch.randint(0, 2, (4,))
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            optimizer1.step()
            optimizer1.zero_grad()
        
        # Save state
        state_dict = optimizer1.state_dict()
        
        # Create new optimizer with same structure
        optimizer2 = CASMO([
            {'params': model.fc.weight, 'lr': 1e-3},
            {'params': model.fc.bias, 'lr': 1e-4}
        ])
        
        # Load state
        optimizer2.load_state_dict(state_dict)
        
        # Verify both group states are preserved
        assert len(optimizer2._group_states) == 2
        assert optimizer2._step_count == optimizer1._step_count


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
