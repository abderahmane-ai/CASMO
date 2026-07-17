#!/usr/bin/env python3
"""Quick verification that CASMO is installed and working."""

import sys

import torch
import torch.nn as nn


def main() -> int:
    try:
        from casmo import CASMO
    except ImportError as exc:
        print(f"FAIL: could not import CASMO: {exc}")
        return 1
    print("ok: CASMO imported")

    try:
        model = nn.Linear(10, 2)
        optimizer = CASMO(model.parameters(), lr=1e-3)
    except Exception as exc:
        print(f"FAIL: could not initialize optimizer: {exc}")
        return 1
    print("ok: optimizer initialized")

    try:
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        optimizer.zero_grad()
        nn.CrossEntropyLoss()(model(x), y).backward()
        optimizer.step()
    except Exception as exc:
        print(f"FAIL: training step failed: {exc}")
        return 1
    print("ok: training step completed")

    try:
        metrics = optimizer.group_metrics(0)
        assert metrics["agar"] is not None and metrics["confidence"] is not None
    except Exception as exc:
        print(f"FAIL: metrics unavailable: {exc}")
        return 1
    print(
        f"ok: metrics reported (agar={metrics['agar']:.4f}, "
        f"confidence={metrics['confidence']:.4f})"
    )

    try:
        state = optimizer.state_dict()
        restored = CASMO(model.parameters(), lr=1e-3)
        restored.load_state_dict(state)
    except Exception as exc:
        print(f"FAIL: state dict save/load failed: {exc}")
        return 1
    print("ok: state dict save/load works")

    print("\nAll basic functionality checks passed.")
    print("Run the full suite with: pytest tests/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
