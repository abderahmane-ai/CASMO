"""
Unit tests for edge cases and error handling in CASMO.

Tests verify robustness to unusual inputs and error conditions.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import pytest
from casmo import CASMO


class TestGradientEdgeCases:
    """Test edge cases related to gradients."""
    
    def test_nan_gradient_detection(self):
        """Test that NaN gradients are detected and raise error."""
        param = torch.nn.Parameter(torch.randn(5, 5))
        optimizer = CASMO([param], lr=1e-3)
        
        # Set gradient to NaN
        param.grad = torch.tensor([[float('nan')]])
        
        with pytest.raises(RuntimeError, match="NaN gradient detected"):
            optimizer.step()
    
    def test_inf_gradient_detection(self):
        """Test that Inf gradients are detected and raise error."""
        param = torch.nn.Parameter(torch.randn(5, 5))
        optimizer = CASMO([param], lr=1e-3)
        
        # Set gradient to Inf
        param.grad = torch.tensor([[float('inf')]])
        
        with pytest.raises(RuntimeError, match="Inf gradient detected"):
            optimizer.step()
    
    def test_zero_gradients(self):
        """Test that zero gradients are handled correctly."""
        param = torch.nn.Parameter(torch.randn(5, 5))
        optimizer = CASMO([param], lr=1e-3)
        
        # Set all gradients to zero
        param.grad = torch.zeros(5, 5)
        
        # Should not raise error
        optimizer.step()
        optimizer.zero_grad()
    
    def test_very_small_gradients(self):
        """Test handling of very small gradients."""
        param = torch.nn.Parameter(torch.randn(5, 5))
        optimizer = CASMO([param], lr=1e-3)
        
        for _ in range(20):
            param.grad = torch.ones(5, 5) * 1e-10
            optimizer.step()
            optimizer.zero_grad()
        
        # Should complete without error
        assert True
    
    def test_very_large_gradients(self):
        """Test handling of very large gradients with clamping."""
        param = torch.nn.Parameter(torch.randn(5, 5))
        optimizer = CASMO([param], lr=1e-3, agar_clamp_factor=10.0)
        
        for _ in range(20):
            param.grad = torch.ones(5, 5) * 1e6
            optimizer.step()
            optimizer.zero_grad()
        
        # Should handle large gradients with clamping
        assert True
    
    def test_mixed_zero_nonzero_gradients(self):
        """Test parameters with some zero and some non-zero gradients."""
        params = [
            torch.nn.Parameter(torch.randn(3, 3)),
            torch.nn.Parameter(torch.randn(2, 2))
        ]
        optimizer = CASMO(params, lr=1e-3)
        
        # First param gets gradient, second doesn't
        params[0].grad = torch.randn(3, 3)
        params[1].grad = None
        
        # Should handle gracefully
        optimizer.step()
        optimizer.zero_grad()
    
    def test_sparse_gradient_error(self):
        """Test that sparse gradients raise NotImplementedError."""
        param = torch.nn.Parameter(torch.randn(10, 10))
        optimizer = CASMO([param], lr=1e-3)
        
        # Create sparse gradient
        indices = torch.LongTensor([[0, 1], [2, 3]])
        values = torch.FloatTensor([1.0, 2.0])
        param.grad = torch.sparse.FloatTensor(indices, values, torch.Size([10, 10]))
        
        with pytest.raises(NotImplementedError, match="sparse gradients"):
            optimizer.step()


class TestParameterEdgeCases:
    """Test edge cases related to parameters."""
    
    def test_empty_parameter_list(self):
        """Test optimizer with empty parameter list."""
        optimizer = CASMO([], lr=1e-3)
        
        # Step should not crash
        optimizer.step()
    
    def test_single_parameter(self):
        """Test optimizer with single parameter."""
        param = torch.nn.Parameter(torch.randn(10))
        optimizer = CASMO([param], lr=1e-3)
        
        for _ in range(20):
            param.grad = torch.randn(10)
            optimizer.step()
            optimizer.zero_grad()
        
        assert True
    
    def test_very_small_parameter(self):
        """Test optimizer with very small parameter tensor."""
        param = torch.nn.Parameter(torch.randn(1))
        optimizer = CASMO([param], lr=1e-3)
        
        for _ in range(20):
            param.grad = torch.randn(1)
            optimizer.step()
            optimizer.zero_grad()
        
        # Should compute AGAR even for single element
        assert optimizer._group_states[0]['current_agar'] is not None
    
    def test_large_parameter_count(self):
        """Test optimizer with many parameters."""
        params = [torch.nn.Parameter(torch.randn(10, 10)) for _ in range(100)]
        optimizer = CASMO(params, lr=1e-3, granularity='group')
        
        for _ in range(5):
            for p in params:
                p.grad = torch.randn_like(p)
            optimizer.step()
            optimizer.zero_grad()
        
        assert True


class TestTauCalibration:
    """Test edge cases in tau calibration."""
    
    def test_calibration_with_constant_gradients(self):
        """Test tau calibration when gradients are constant."""
        param = torch.nn.Parameter(torch.randn(10, 10))
        optimizer = CASMO([param], lr=1e-3, tau_init_steps=50)
        
        # Send constant gradients
        constant_grad = torch.ones(10, 10) * 0.5
        for _ in range(60):
            param.grad = constant_grad.clone()
            optimizer.step()
            optimizer.zero_grad()
        
        # Should calibrate successfully
        assert optimizer._group_states[0]['tau_initialized'] == True
    
    def test_calibration_with_bimodal_distribution(self):
        """Test tau calibration with bimodal AGAR distribution."""
        param = torch.nn.Parameter(torch.randn(10, 10))
        optimizer = CASMO([param], lr=1e-3, tau_init_steps=50)
        
        # Alternate between consistent and random gradients
        for i in range(60):
            if i % 2 == 0:
                param.grad = torch.ones(10, 10) * 0.5
            else:
                param.grad = torch.randn(10, 10)
            optimizer.step()
            optimizer.zero_grad()
        
        # Should calibrate and detect high variance
        assert optimizer._group_states[0]['tau_initialized'] == True
    
    def test_minimum_calibration_steps(self):
        """Test that calibration requires minimum steps."""
        param = torch.nn.Parameter(torch.randn(10, 10))
        optimizer = CASMO([param], lr=1e-3, tau_init_steps=100)
        
        # Run fewer steps than required
        for _ in range(50):
            param.grad = torch.randn(10, 10)
            optimizer.step()
            optimizer.zero_grad()
        
        # Should not be calibrated yet
        assert optimizer._group_states[0]['tau_initialized'] == False


class TestConfidenceEdgeCases:
    """Test edge cases in confidence computation."""
    
    def test_confidence_before_calibration(self):
        """Test that confidence is computed before calibration."""
        param = torch.nn.Parameter(torch.randn(10, 10))
        optimizer = CASMO([param], lr=1e-3, tau_init_steps=100, c_min=0.1)
        
        # Run a few steps before calibration
        for _ in range(10):
            param.grad = torch.randn(10, 10)
            optimizer.step()
            optimizer.zero_grad()
        
        # Confidence should still be computed (using simple passthrough)
        current_conf = optimizer._group_states[0].get('current_confidence')
        assert current_conf is not None
        assert 0.1 <= current_conf <= 1.0
    
    def test_confidence_after_calibration(self):
        """Test confidence computation after calibration."""
        param = torch.nn.Parameter(torch.randn(10, 10))
        optimizer = CASMO([param], lr=1e-3, tau_init_steps=50)
        
        # Calibrate
        for _ in range(60):
            param.grad = torch.randn(10, 10) * 0.1
            optimizer.step()
            optimizer.zero_grad()
        
        assert optimizer._group_states[0]['tau_initialized'] == True
        
        # Run more steps
        for _ in range(20):
            param.grad = torch.randn(10, 10) * 0.1
            optimizer.step()
            optimizer.zero_grad()
        
        # Confidence should use sigmoid mapping
        current_conf = optimizer._group_states[0].get('current_confidence')
        assert current_conf is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
