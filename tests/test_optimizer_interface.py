"""Tests for PyTorch Optimizer interface compliance and argument validation."""

import warnings

import pytest
import torch
import torch.nn as nn

from casmo import CASMO


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestOptimizerInterface:
    def test_basic_initialization(self):
        model = SimpleModel()
        optimizer = CASMO(model.parameters(), lr=1e-3)
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]["lr"] == 1e-3

    def test_parameter_groups(self):
        model = SimpleModel()
        optimizer = CASMO(
            [
                {"params": model.fc1.parameters(), "lr": 1e-3},
                {"params": model.fc2.parameters(), "lr": 1e-4},
            ]
        )
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.param_groups[1]["lr"] == 1e-4

    def test_per_group_hyperparameters(self):
        """robustness must be settable per parameter group."""
        model = SimpleModel()
        optimizer = CASMO(
            [
                {"params": model.fc1.parameters(), "robustness": 0.0},
                {"params": model.fc2.parameters(), "robustness": 1.0},
            ],
            lr=1e-3,
        )
        assert optimizer.param_groups[0]["robustness"] == 0.0
        assert optimizer.param_groups[1]["robustness"] == 1.0

    def test_step_and_zero_grad(self):
        model = SimpleModel()
        optimizer = CASMO(model.parameters(), lr=1e-3)
        loss = nn.CrossEntropyLoss()(model(torch.randn(4, 10)), torch.tensor([0, 1, 0, 1]))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.all(p.grad == 0)

    def test_training_loop_reduces_loss(self):
        torch.manual_seed(42)
        model = SimpleModel()
        optimizer = CASMO(model.parameters(), lr=1e-2)
        x, y = torch.randn(20, 10), torch.randint(0, 2, (20,))
        criterion = nn.CrossEntropyLoss()

        initial = criterion(model(x), y).item()
        for _ in range(30):
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
        final = criterion(model(x), y).item()

        assert final < initial, f"loss should decrease: {initial} -> {final}"

    def test_learning_rate_scheduler(self):
        model = SimpleModel()
        optimizer = CASMO(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        initial_lr = optimizer.param_groups[0]["lr"]

        for _ in range(10):
            loss = nn.CrossEntropyLoss()(model(torch.randn(4, 10)), torch.randint(0, 2, (4,)))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        assert optimizer.param_groups[0]["lr"] < initial_lr

    def test_weight_decay_shrinks_unused_weights(self):
        """Decoupled weight decay must shrink a parameter that gets zero gradient."""
        param = torch.nn.Parameter(torch.ones(5))
        optimizer = CASMO([param], lr=1e-1, weight_decay=0.1)
        for _ in range(10):
            param.grad = torch.zeros(5)
            optimizer.step()
        assert torch.all(param < 1.0), "weight decay should shrink weights"

    def test_closure_support(self):
        model = SimpleModel()
        optimizer = CASMO(model.parameters(), lr=1e-3)
        x, y = torch.randn(4, 10), torch.randint(0, 2, (4,))

        def closure():
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        assert loss is not None
        assert isinstance(loss.item(), float)


class TestParameterValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"lr": -0.1},
            {"eps": -1e-8},
            {"betas": (1.5, 0.999)},
            {"betas": (0.9, -0.1)},
            {"weight_decay": -0.01},
            {"c_min": 1.5},
            {"c_min": -0.1},
            {"robustness": -0.5},
            {"rel_floor": 1.5},
            {"rel_floor": -0.1},
        ],
    )
    def test_invalid_hyperparameters_raise(self, kwargs):
        model = SimpleModel()
        with pytest.raises(ValueError):
            CASMO(model.parameters(), **kwargs)

    def test_unknown_kwarg_raises_type_error(self):
        model = SimpleModel()
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            CASMO(model.parameters(), not_a_real_option=1)

    @pytest.mark.parametrize(
        "kwarg",
        ["tau_init_steps", "tau_clip_range", "granularity", "agar_clamp_factor", "total_steps"],
    )
    def test_deprecated_kwargs_warn_but_are_accepted(self, kwarg):
        """Pre-0.4 calibration kwargs are ignored, not fatal."""
        model = SimpleModel()
        with pytest.warns(DeprecationWarning, match=kwarg):
            optimizer = CASMO(model.parameters(), lr=1e-3, **{kwarg: 500})
        assert kwarg not in optimizer.param_groups[0]

    def test_small_total_steps_does_not_crash(self):
        """Regression: total_steps used to raise 'tau_init_steps too small'."""
        model = SimpleModel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            CASMO(model.parameters(), lr=1e-3, total_steps=200)
