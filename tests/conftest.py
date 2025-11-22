"""
Pytest configuration and shared fixtures for CASMO tests.
"""

import pytest
import torch
import random
import numpy as np


@pytest.fixture(scope="function")
def seed():
    """Set random seeds for reproducibility."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    yield
    # Cleanup after test
    torch.manual_seed(torch.initial_seed())


@pytest.fixture(scope="function")
def device():
    """Provide device for testing (CPU by default, GPU if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="function")
def simple_params():
    """Provide simple parameter list for testing."""
    return [
        torch.nn.Parameter(torch.randn(10, 10)),
        torch.nn.Parameter(torch.randn(5)),
    ]
