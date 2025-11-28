
import torch
from casmo import CASMO, DDEAdapter
import numpy as np

def test_ddea_integration():
    print("Testing DDEA Integration...")
    
    # Setup dummy model and optimizer
    param = torch.tensor([1.0], requires_grad=True)
    optimizer = CASMO([param], lr=0.1, tau_init_steps=50)
    
    # Access internal state for the group
    group_state = optimizer._group_states[0]
    
    # 1. Manually initialize state as if calibration just finished
    group_state['tau_initialized'] = True
    group_state['agar_mean'] = 0.5  # Calibrated mean
    group_state['agar_std'] = 0.1   # Calibrated std
    group_state['c_min'] = 0.1
    
    # Initialize adapter with a specific tau
    adapter = group_state['tau_adapter']
    adapter.tau = 0.5
    adapter.tau_calibrated = 0.5
    
    # 2. Force adapter to drift significantly
    # We'll set the adapter's tau to 0.8 manually to simulate drift
    # If integration is working, this NEW tau (0.8) should be used as the center (mu)
    # If integration is BROKEN, the old calibrated mean (0.5) will be used
    adapter.tau = 0.8
    print(f"DEBUG: Adapter tau set to {adapter.tau}")
    print(f"DEBUG: Calibrated mean is {group_state['agar_mean']}")
    
    # 3. Run a step with AGAR = 0.65
    # Scenario A (BROKEN): Center = 0.5. AGAR 0.65 is > 0.5. z-score = (0.65 - 0.5)/0.1 = +1.5. 
    #   Sigmoid(1.5) ≈ 0.82. Confidence should be HIGH (> 0.5).
    # Scenario B (FIXED): Center = 0.8. AGAR 0.65 is < 0.8. z-score = (0.65 - 0.8)/0.1 = -1.5.
    #   Sigmoid(-1.5) ≈ 0.18. Confidence should be LOW (< 0.5).
    
    param.grad = torch.tensor([1.0]) # Dummy gradient
    
    # We need to mock the AGAR computation to return exactly 0.65
    # Since that's hard to force through gradients without math, we'll patch _compute_agar temporarily
    original_compute_agar = optimizer._compute_agar
    optimizer._compute_agar = lambda *args: torch.tensor(0.65)
    
    optimizer.step()
    
    # Restore
    optimizer._compute_agar = original_compute_agar
    
    # Check confidence used
    confidence = group_state['current_confidence']
    print(f"Resulting Confidence: {confidence:.4f}")
    
    if confidence > 0.5:
        print("❌ FAIL: Confidence is HIGH. System used static mean (0.5) instead of adapter tau (0.8).")
        print("   The adapter's drift was IGNORED.")
    else:
        print("✅ PASS: Confidence is LOW. System correctly used adapter tau (0.8).")

if __name__ == "__main__":
    test_ddea_integration()
