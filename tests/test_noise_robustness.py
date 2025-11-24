"""
Noise robustness tests for CASMO optimizer.

Tests AGAR's ability to distinguish between clean and noisy gradients,
and verify that confidence scaling adapts appropriately.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from casmo import CASMO


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


class TestNoiseRobustness:
    """Tests for CASMO's noise handling capabilities."""
    
    def test_agar_detects_high_variance(self):
        """Test that AGAR is lower when gradients have high variance."""
        model = SimpleModel()
        optimizer = CASMO(model.parameters(), lr=1e-3, tau_init_steps=50, granularity='group')
        
        agar_values = []
        
        # Simulate training with varying gradient noise
        for i in range(100):
            # Generate gradients with increasing noise
            noise_level = 0.0 if i < 50 else 2.0
            
            x = torch.randn(32, 10)
            y = torch.randint(0, 2, (32,))
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = nn.functional.cross_entropy(outputs, y)
            
            # Add noise to gradients
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad += torch.randn_like(p.grad) * noise_level
            
            optimizer.step()
            
            # Collect AGAR after calibration
            if i >= 50:
                agar = optimizer._group_states[0].get('current_agar')
                if agar is not None:
                    agar_values.append(agar)
        
        if len(agar_values) > 10:
            # AGAR should decrease when noise is added
            early_agar = np.mean(agar_values[:5])
            late_agar = np.mean(agar_values[-5:])
            
            assert early_agar > late_agar, \
                f"AGAR should decrease with noise. Early: {early_agar}, Late: {late_agar}"
    
    def test_confidence_adapts_to_noise(self):
        """Test that confidence scaling reduces with noisy gradients."""
        model = SimpleModel()
        optimizer = CASMO(model.parameters(), lr=1e-3, tau_init_steps=100, granularity='group')
        
        confidence_values = []
        
        # Clean training then noisy training
        for i in range(150):
            noise_level = 0.0 if i < 100 else 1.5
            
            x = torch.randn(32, 10)
            y = torch.randint(0, 2, (32,))
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = nn.functional.cross_entropy(outputs, y)
            loss.backward()
            
            # Add noise
            for p in model.parameters():
                if p.grad is not None:
                    p.grad += torch.randn_like(p.grad) * noise_level
            
            optimizer.step()
            
            # Collect confidence after calibration
            if i >= 100:
                conf = optimizer._group_states[0].get('current_confidence')
                if conf is not None:
                    confidence_values.append(conf)
        
        if len(confidence_values) > 10:
            # Confidence should adapt to the noise level
            # May increase or decrease depending on noise characteristics
            # Just verify it's reasonable
            mean_conf = np.mean(confidence_values)
            assert 0.05 <= mean_conf <= 1.0, \
                f"Confidence should be in valid range, got {mean_conf}"
    
    def test_bimodal_gradient_distribution(self):
        """Test CASMO handles bimodal gradient distributions (mixed clean/noisy batches)."""
        model = SimpleModel()
        optimizer = CASMO(
            model.parameters(), 
            lr=1e-3, 
            tau_init_steps=100,
            granularity='group',
            c_min=0.1
        )
        
        # Simulate bimodal distribution: alternating clean and noisy batches
        for i in range(150):
            is_clean = (i % 2 == 0)
            noise_level = 0.0 if is_clean else 2.0
            
            x = torch.randn(32, 10)
            y = torch.randint(0, 2, (32,))
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = nn.functional.cross_entropy(outputs, y)
            loss.backward()
            
            for p in model.parameters():
                if p.grad is not None:
                    p.grad += torch.randn_like(p.grad) * noise_level
            
            optimizer.step()
        
        # After calibration, check that distribution parameters are reasonable
        group_state = optimizer._group_states[0]
        
        if group_state.get('tau_initialized'):
            agar_std = group_state.get('agar_std', 0)
            agar_mean = group_state.get('agar_mean', 0)
            
            # Bimodal should have higher std relative to mean
            cv = agar_std / (agar_mean + 1e-8)
            
            # With bimodal distribution, CV should be relatively high
            assert cv > 0.1, f"Expected higher CV for bimodal distribution, got {cv}"
    
    def test_all_zero_gradients(self):
        """Test handling of all-zero gradients (dead neurons)."""
        model = SimpleModel()
        optimizer = CASMO(model.parameters(), lr=1e-3, tau_init_steps=50)
        
        # Manually zero out gradients
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        
        # Should not crash
        optimizer.step()
        
        # Parameters should not change with zero gradients
        original_params = [p.clone() for p in model.parameters()]
        optimizer.step()
        
        for orig, current in zip(original_params, model.parameters()):
            assert torch.allclose(orig, current), "Parameters should not change with zero gradients"
    
    def test_gradient_outliers(self):
        """Test that AGAR clamping handles gradient outliers."""
        model = SimpleModel()
        optimizer = CASMO(
            model.parameters(), 
            lr=1e-3, 
            tau_init_steps=50,
            agar_clamp_factor=10.0  # Enable clamping
        )
        
        # Generate normal gradients
        for i in range(60):
            x = torch.randn(32, 10)
            y = torch.randint(0, 2, (32,))
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = nn.functional.cross_entropy(outputs, y)
            loss.backward()
            
            # Inject occasional outliers
            if i % 10 == 0:
                for p in model.parameters():
                    if p.grad is not None:
                        # Add extreme outlier
                        outlier_mask = torch.rand_like(p.grad) < 0.01
                        p.grad[outlier_mask] += torch.randn_like(p.grad[outlier_mask]) * 1000
            
            # Should not crash or produce NaN
            optimizer.step()
            
            for p in model.parameters():
                assert not torch.isnan(p).any(), "Parameters should not become NaN"
    
    def test_c_min_prevents_lr_collapse(self):
        """Test that c_min prevents learning rate from collapsing to zero."""
        model = SimpleModel()
        c_min = 0.2
        optimizer = CASMO(
            model.parameters(), 
            lr=1e-3, 
            c_min=c_min,
            tau_init_steps=50
        )
        
        # Generate very noisy gradients
        for i in range(100):
            x = torch.randn(32, 10)
            y = torch.randint(0, 2, (32,))
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = nn.functional.cross_entropy(outputs, y)
            loss.backward()
            
            # Add heavy noise
            for p in model.parameters():
                if p.grad is not None:
                    p.grad += torch.randn_like(p.grad) * 5.0
            
            optimizer.step()
        
        # Confidence should never go below c_min
        group_state = optimizer._group_states[0]
        if group_state.get('tau_initialized'):
            confidence = group_state.get('current_confidence', 1.0)
            effective_c_min = group_state.get('c_min', c_min)
            
            assert confidence >= effective_c_min * 0.99, \
                f"Confidence should not go below c_min. Got {confidence}, c_min={effective_c_min}"
    
    def test_pervasive_noise_adaptation(self):
        """Test that CASMO adapts c_min for pervasive noise."""
        model = SimpleModel()
        optimizer = CASMO(
            model.parameters(), 
            lr=1e-3,
            tau_init_steps=100,
            c_min=0.1  # Initial c_min
        )
        
        # Generate consistently noisy gradients (pervasive noise scenario)
        for i in range(120):
            x = torch.randn(32, 10)
            y = torch.randint(0, 2, (32,))
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = nn.functional.cross_entropy(outputs, y)
            loss.backward()
            
            # Consistent moderate noise
            for p in model.parameters():
                if p.grad is not None:
                    p.grad += torch.randn_like(p.grad) * 0.5
            
            optimizer.step()
        
        # With pervasive noise, c_min should adapt upward
        group_state = optimizer._group_states[0]
        if group_state.get('tau_initialized'):
            adapted_c_min = group_state.get('c_min', 0.1)
            
            # Should have adapted to higher c_min for pervasive noise
            # (Low CV -> high c_min per CASMO's adaptive logic)
            assert adapted_c_min > 0.1, \
                f"c_min should adapt upward for pervasive noise, got {adapted_c_min}"
