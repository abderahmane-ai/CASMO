"""
Unit tests for AGAR (Adaptive Gradient Alignment Ratio) computation.

Tests verify the correctness of AGAR metric calculation under various conditions.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pytest
from casmo import CASMO


class TestAGARComputation:
    """Test suite for AGAR computation logic."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.eps = 1e-8
        self.device = torch.device('cpu')
        
    def test_agar_perfect_signal(self):
        """Test AGAR = 1.0 for zero-variance gradients (perfect signal)."""
        # Create simple model with one parameter
        param = torch.nn.Parameter(torch.randn(10, 10))
        optimizer = CASMO([param], lr=1e-3)
        
        # Simulate gradients with zero variance (same direction)
        consistent_grad = torch.ones(10, 10) * 0.5
        
        for _ in range(100):  # Run enough steps to stabilize moments
            param.grad = consistent_grad.clone()
            optimizer.step()
            optimizer.zero_grad()
        
        # Check AGAR is close to 1.0 (pure signal)
        group_state = optimizer._group_states[0]
        current_agar = group_state.get('current_agar')
        
        assert current_agar is not None, "AGAR should be computed"
        assert current_agar > 0.95, f"AGAR should be ~1.0 for zero-variance gradients, got {current_agar}"
    
    def test_agar_pure_noise(self):
        """Test AGAR ≈ 0 for high-variance random gradients (pure noise)."""
        param = torch.nn.Parameter(torch.randn(10, 10))
        optimizer = CASMO([param], lr=1e-3)
        
        # Simulate random gradients (high variance)
        torch.manual_seed(42)
        for _ in range(100):
            random_grad = torch.randn(10, 10)  # Different each time
            param.grad = random_grad
            optimizer.step()
            optimizer.zero_grad()
        
        # Check AGAR is low (mostly noise)
        group_state = optimizer._group_states[0]
        current_agar = group_state.get('current_agar')
        
        assert current_agar is not None, "AGAR should be computed"
        assert current_agar < 0.3, f"AGAR should be low for random gradients, got {current_agar}"
    
    def test_agar_bounded_range(self):
        """Test AGAR is always in [0, 1] range."""
        param = torch.nn.Parameter(torch.randn(20, 20))
        optimizer = CASMO([param], lr=1e-3)
        
        torch.manual_seed(123)
        agar_values = []
        
        for _ in range(200):
            # Mix of consistent and random gradients
            if torch.rand(1).item() > 0.5:
                grad = torch.ones(20, 20) * 0.3
            else:
                grad = torch.randn(20, 20) * 0.1
            
            param.grad = grad
            optimizer.step()
            optimizer.zero_grad()
            
            current_agar = optimizer._group_states[0].get('current_agar')
            if current_agar is not None:
                agar_values.append(current_agar)
        
        assert len(agar_values) > 0, "Should have computed AGAR values"
        assert all(0.0 <= v <= 1.0 for v in agar_values), "All AGAR values must be in [0, 1]"
    
    def test_agar_with_outlier_clamping(self):
        """Test AGAR computation with outlier clamping enabled."""
        param = torch.nn.Parameter(torch.randn(10, 10))
        optimizer = CASMO([param], lr=1e-3, agar_clamp_factor=10.0)
        
        # Send normal gradients then one outlier
        for i in range(50):
            if i == 25:
                # Inject outlier
                grad = torch.ones(10, 10) * 100.0
            else:
                grad = torch.ones(10, 10) * 0.5
            
            param.grad = grad
            optimizer.step()
            optimizer.zero_grad()
        
        # Should still produce valid AGAR
        current_agar = optimizer._group_states[0].get('current_agar')
        assert current_agar is not None
        assert 0.0 <= current_agar <= 1.0
    
    def test_agar_without_clamping(self):
        """Test AGAR computation with outlier clamping disabled."""
        param = torch.nn.Parameter(torch.randn(10, 10))
        optimizer = CASMO([param], lr=1e-3, agar_clamp_factor=None)
        
        for _ in range(50):
            grad = torch.randn(10, 10)
            param.grad = grad
            optimizer.step()
            optimizer.zero_grad()
        
        current_agar = optimizer._group_states[0].get('current_agar')
        assert current_agar is not None
        assert 0.0 <= current_agar <= 1.0
    
    def test_agar_per_parameter_granularity(self):
        """Test AGAR computation with per-parameter granularity."""
        params = [
            torch.nn.Parameter(torch.randn(5, 5)),
            torch.nn.Parameter(torch.randn(3, 3))
        ]
        optimizer = CASMO(params, lr=1e-3, granularity='parameter')
        
        for _ in range(50):
            for p in params:
                p.grad = torch.randn_like(p) * 0.1
            optimizer.step()
            optimizer.zero_grad()
        
        # Should compute AGAR successfully
        current_agar = optimizer._group_states[0].get('current_agar')
        assert current_agar is not None
    
    def test_agar_per_group_granularity(self):
        """Test AGAR computation with per-group granularity."""
        params = [
            torch.nn.Parameter(torch.randn(5, 5)),
            torch.nn.Parameter(torch.randn(3, 3))
        ]
        optimizer = CASMO(params, lr=1e-3, granularity='group')
        
        for _ in range(50):
            for p in params:
                p.grad = torch.randn_like(p) * 0.1
            optimizer.step()
            optimizer.zero_grad()
        
        current_agar = optimizer._group_states[0].get('current_agar')
        assert current_agar is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
