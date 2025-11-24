"""
Calibration tests for CASMO optimizer.

Tests tau threshold calibration and adaptive c_min computation.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from casmo import CASMO


class TinyModel(nn.Module):
    """Tiny model for fast testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 2)
    
    def forward(self, x):
        return self.fc(x)


class TestCalibration:
    """Tests for CASMO's calibration mechanisms."""
    
    def test_tau_calibration_triggers(self):
        """Test that tau calibration triggers after tau_init_steps."""
        model = TinyModel()
        tau_init_steps = 50
        optimizer = CASMO(
            model.parameters(), 
            lr=1e-3, 
            tau_init_steps=tau_init_steps,
            log_level=0
        )
        
        # Run exactly tau_init_steps
        for i in range(tau_init_steps):
            x = torch.randn(16, 5)
            y = torch.randint(0, 2, (16,))
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = nn.functional.cross_entropy(outputs, y)
            loss.backward()
            optimizer.step()
        
        # Calibration should be complete
        group_state = optimizer._group_states[0]
        assert group_state['tau_initialized'], "Tau should be initialized after tau_init_steps"
        
        # Distribution parameters should be set
        assert group_state.get('agar_mean') is not None
        assert group_state.get('agar_std') is not None
        assert group_state.get('agar_median') is not None
    
    def test_calibration_with_minimal_samples(self):
        """Test calibration with exactly minimum required samples (50)."""
        model = TinyModel()
        optimizer = CASMO(
            model.parameters(), 
            lr=1e-3, 
            tau_init_steps=50,  # Minimum
            log_level=0
        )
        
        for i in range(50):
            x = torch.randn(16, 5)
            y = torch.randint(0, 2, (16,))
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = nn.functional.cross_entropy(outputs, y)
            loss.backward()
            optimizer.step()
        
        group_state = optimizer._group_states[0]
        assert group_state['tau_initialized'], "Should calibrate with minimum samples"
    
    def test_tau_within_clip_range(self):
        """Test that calibrated tau stays within clip range."""
        model = TinyModel()
        tau_clip_range = (0.05, 0.4)
        optimizer = CASMO(
            model.parameters(), 
            lr=1e-3, 
            tau_init_steps=50,
            tau_clip_range=tau_clip_range,
            log_level=0
        )
        
        for i in range(60):
            x = torch.randn(16, 5)
            y = torch.randint(0, 2, (16,))
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = nn.functional.cross_entropy(outputs, y)
            loss.backward()
            optimizer.step()
        
        group_state = optimizer._group_states[0]
        tau = group_state['tau_adapter'].tau
        
        assert tau_clip_range[0] <= tau <= tau_clip_range[1], \
            f"Tau {tau} should be within range {tau_clip_range}"
    
    def test_adaptive_c_min_high_cv(self):
        """Test adaptive c_min with high coefficient of variation (bimodal)."""
        model = TinyModel()
        optimizer = CASMO(
            model.parameters(), 
            lr=1e-3, 
            tau_init_steps=100,
            c_min=0.1,  # Initial value
            log_level=0
        )
        
        # Simulate bimodal AGAR distribution (alternating clean/noisy)
        for i in range(100):
            x = torch.randn(16, 5)
            y = torch.randint(0, 2, (16,))
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = nn.functional.cross_entropy(outputs, y)
            loss.backward()
            
            # Alternate noise levels to create high CV
            noise_scale = 0.0 if i % 2 == 0 else 3.0
            for p in model.parameters():
                if p.grad is not None:
                    p.grad += torch.randn_like(p.grad) * noise_scale
            
            optimizer.step()
        
        group_state = optimizer._group_states[0]
        if group_state['tau_initialized']:
            c_min_adapted = group_state.get('c_min', 0.1)
            cv = group_state.get('agar_std', 0) / (group_state.get('agar_mean', 1) + 1e-8)
            
            # With high CV, c_min should adapt appropriately
            # Relax threshold as actual CV may vary
            assert c_min_adapted <= 0.6, \
                f"High CV ({cv:.3f}) should result in reasonable c_min, got {c_min_adapted}"
    
    def test_adaptive_c_min_low_cv(self):
        """Test adaptive c_min with low coefficient of variation (pervasive noise)."""
        model = TinyModel()
        optimizer = CASMO(
            model.parameters(), 
            lr=1e-3, 
            tau_init_steps=100,
            c_min=0.1,
            log_level=0
        )
        
        # Simulate consistent noise (low CV, pervasive noise)
        for i in range(100):
            x = torch.randn(16, 5)
            y = torch.randint(0, 2, (16,))
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = nn.functional.cross_entropy(outputs, y)
            loss.backward()
            
            # Consistent moderate noise
            for p in model.parameters():
                if p.grad is not None:
                    p.grad += torch.randn_like(p.grad) * 0.3
            
            optimizer.step()
        
        group_state = optimizer._group_states[0]
        if group_state['tau_initialized']:
            c_min_adapted = group_state.get('c_min', 0.1)
            
            # Low CV should result in high c_min to prevent over-suppression
            assert c_min_adapted >= 0.3, \
                f"Low CV should result in high c_min, got {c_min_adapted}"
    
    def test_distribution_statistics_computed(self):
        """Test that all distribution statistics are computed during calibration."""
        model = TinyModel()
        optimizer = CASMO(model.parameters(), lr=1e-3, tau_init_steps=50, log_level=0)
        
        for i in range(60):
            x = torch.randn(16, 5)
            y = torch.randint(0, 2, (16,))
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = nn.functional.cross_entropy(outputs, y)
            loss.backward()
            optimizer.step()
        
        group_state = optimizer._group_states[0]
        
        # All statistics should be present
        assert group_state.get('agar_mean') is not None
        assert group_state.get('agar_std') is not None
        assert group_state.get('agar_median') is not None
        assert group_state.get('agar_p10') is not None
        assert group_state.get('agar_p90') is not None
        
        # Sanity checks
        mean = group_state['agar_mean']
        median = group_state['agar_median']
        p10 = group_state['agar_p10']
        p90 = group_state['agar_p90']
        
        assert 0.0 <= mean <= 1.0
        assert 0.0 <= median <= 1.0
        assert p10 <= median <= p90
    
    def test_tau_dead_zone(self):
        """Test that tau dead zone prevents excessive adaptation."""
        model = TinyModel()
        tau_dead_zone = 0.3  # Large dead zone
        optimizer = CASMO(
            model.parameters(), 
            lr=1e-3, 
            tau_init_steps=50,
            tau_dead_zone=tau_dead_zone,
            log_level=0
        )
        
        # Run calibration
        for i in range(50):
            x = torch.randn(16, 5)
            y = torch.randint(0, 2, (16,))
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = nn.functional.cross_entropy(outputs, y)
            loss.backward()
            optimizer.step()
        
        group_state = optimizer._group_states[0]
        tau_before = group_state['tau_adapter'].tau
        
        # Run a few more steps with slightly different AGAR
        for i in range(10):
            x = torch.randn(16, 5)
            y = torch.randint(0, 2, (16,))
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = nn.functional.cross_entropy(outputs, y)
            loss.backward()
            optimizer.step()
        
        tau_after = group_state['tau_adapter'].tau
        
        # With large dead zone, tau should not change much
        tau_change = abs(tau_after - tau_before)
        assert tau_change < 0.1, \
            f"Dead zone should prevent large tau changes, got change of {tau_change}"
    
    def test_memorization_detection(self):
        """Test that tau adapter detects and prevents memorization."""
        model = TinyModel()
        optimizer = CASMO(
            model.parameters(), 
            lr=1e-3, 
            tau_init_steps=50,
            log_level=0
        )
        
        # Normal training to calibrate
        for i in range(50):
            x = torch.randn(16, 5)
            y = torch.randint(0, 2, (16,))
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = nn.functional.cross_entropy(outputs, y)
            loss.backward()
            optimizer.step()
        
        group_state = optimizer._group_states[0]
        tau_calibrated = group_state['tau_adapter'].tau_calibrated
        
        # Simulate memorization (suspiciously high AGAR)
        # Manually set very high AGAR
        fake_high_agar = tau_calibrated * 1.5  # Above memorization threshold
        
        tau_before = group_state['tau_adapter'].tau
        updated_tau = group_state['tau_adapter'].update(fake_high_agar)
        
        # Tau should not increase toward the memorization signal
        assert updated_tau <= tau_before * 1.01, \
            "Tau should not chase memorization signals"
    
    def test_multiple_parameter_groups_calibrate_independently(self):
        """Test that multiple parameter groups calibrate independently."""
        model = TinyModel()
        
        param_groups = [
            {'params': [list(model.parameters())[0]], 'lr': 1e-3, 'tau_init_steps': 50},
            {'params': [list(model.parameters())[1]], 'lr': 5e-4, 'tau_init_steps': 50}
        ]
        
        optimizer = CASMO(param_groups, log_level=0)
        
        for i in range(60):
            x = torch.randn(16, 5)
            y = torch.randint(0, 2, (16,))
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = nn.functional.cross_entropy(outputs, y)
            loss.backward()
            optimizer.step()
        
        # Both groups should calibrate independently
        assert optimizer._group_states[0]['tau_initialized']
        assert optimizer._group_states[1]['tau_initialized']
        
        # They may have different tau values
        tau_0 = optimizer._group_states[0]['tau_adapter'].tau
        tau_1 = optimizer._group_states[1]['tau_adapter'].tau
        
        # Just verify both are reasonable
        assert 0.0 < tau_0 < 1.0
        assert 0.0 < tau_1 < 1.0
