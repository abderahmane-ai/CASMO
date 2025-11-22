#!/usr/bin/env python3
"""
Quick verification script to test CASMO installation and basic functionality.
"""

import sys
import torch
import torch.nn as nn

# Test import
try:
    from casmo import CASMO, DDEAdapter
    print("✅ CASMO imported successfully")
except ImportError as e:
    print(f"❌ Failed to import CASMO: {e}")
    sys.exit(1)

# Test basic initialization
try:
    model = nn.Linear(10, 2)
    optimizer = CASMO(model.parameters(), lr=1e-3)
    print("✅ CASMO optimizer initialized")
except Exception as e:
    print(f"❌ Failed to initialize optimizer: {e}")
    sys.exit(1)

# Test basic training step
try:
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))
    criterion = nn.CrossEntropyLoss()
    
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print("✅ Training step completed successfully")
except Exception as e:
    print(f"❌ Training step failed: {e}")
    sys.exit(1)

# Test state dict save/load
try:
    state = optimizer.state_dict()
    new_optimizer = CASMO(model.parameters(), lr=1e-3)
    new_optimizer.load_state_dict(state)
    print("✅ State dict save/load works")
except Exception as e:
    print(f"❌ State dict failed: {e}")
    sys.exit(1)

# Test DDEAdapter
try:
    adapter = DDEAdapter(0.5, (0.1, 0.9), 0.2)
    tau = adapter.update(0.6)
    print(f"✅ DDEAdapter works (tau={tau:.4f})")
except Exception as e:
    print(f"❌ DDEAdapter failed: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("All basic functionality tests passed! ✨")
print("="*50)
print("\nTo run full test suite: pytest tests/ -v")
print("To install package: pip install -e .")
