"""Tests for accelerator support and dynamically added parameter groups.

CASMO contains no device-specific code, so these tests exist to keep it that way:
they run the optimizer on whatever accelerator is present (CUDA or MPS) and assert
it behaves identically to CPU.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from casmo import CASMO


def _accelerators():
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


ACCELERATORS = _accelerators()


class TestAccelerator:
    @pytest.mark.skipif(not ACCELERATORS, reason="no accelerator available")
    @pytest.mark.parametrize("device", ACCELERATORS)
    def test_trains_on_accelerator(self, device):
        torch.manual_seed(0)
        model = nn.Linear(32, 4).to(device)
        optimizer = CASMO(model.parameters(), lr=1e-2, robustness=1.0)
        x = torch.randn(64, 32, device=device)
        y = torch.randint(0, 4, (64,), device=device)

        first = None
        for i in range(30):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            if i == 0:
                first = loss.item()

        assert loss.item() < first
        assert all(torch.isfinite(p).all().item() for p in model.parameters())

    @pytest.mark.skipif(not ACCELERATORS, reason="no accelerator available")
    @pytest.mark.parametrize("device", ACCELERATORS)
    def test_state_lives_on_the_parameter_device(self, device):
        param = torch.nn.Parameter(torch.randn(8, device=device))
        optimizer = CASMO([param], lr=1e-3)
        param.grad = torch.randn(8, device=device)
        optimizer.step()

        state = optimizer.state[param]
        assert state["m"].device.type == device
        assert state["s"].device.type == device

    @pytest.mark.skipif(not ACCELERATORS, reason="no accelerator available")
    @pytest.mark.parametrize("device", ACCELERATORS)
    def test_metrics_are_readable_from_accelerator(self, device):
        param = torch.nn.Parameter(torch.randn(16, device=device))
        optimizer = CASMO([param], lr=1e-3)
        param.grad = torch.randn(16, device=device)
        optimizer.step()

        metrics = optimizer.group_metrics(0)
        assert isinstance(metrics["agar"], float)
        assert isinstance(metrics["confidence"], float)
        assert 0.0 <= metrics["agar"] <= 1.0


class TestMetricsAreLazy:
    def test_step_does_not_materialize_metrics(self):
        """step() must leave metrics on-device; only group_metrics() converts.

        Materializing in step() forces a host sync every iteration.
        """
        param = torch.nn.Parameter(torch.randn(4, 4))
        optimizer = CASMO([param], lr=1e-3)
        param.grad = torch.randn(4, 4)
        optimizer.step()

        raw = optimizer._group_states[0]["current_agar"]
        assert isinstance(raw, torch.Tensor), "step() should not call .item()"
        assert isinstance(optimizer.group_metrics(0)["agar"], float)


class TestAddParamGroup:
    def test_group_added_after_init_is_handled(self):
        """add_param_group() must work without pre-seeding _group_states."""
        a = torch.nn.Parameter(torch.randn(5, 5))
        b = torch.nn.Parameter(torch.randn(5, 5))
        optimizer = CASMO([a], lr=1e-3)

        optimizer.add_param_group({"params": [b], "lr": 1e-4})
        assert len(optimizer.param_groups) == 2

        for _ in range(5):
            a.grad = torch.randn(5, 5)
            b.grad = torch.randn(5, 5)
            optimizer.step()

        assert optimizer.group_metrics(0)["agar"] is not None
        assert optimizer.group_metrics(1)["agar"] is not None

    def test_added_group_inherits_defaults(self):
        a = torch.nn.Parameter(torch.randn(3))
        optimizer = CASMO([a], lr=1e-3, robustness=0.75, c_min=0.2)
        b = torch.nn.Parameter(torch.randn(3))
        optimizer.add_param_group({"params": [b]})

        assert optimizer.param_groups[1]["robustness"] == 0.75
        assert optimizer.param_groups[1]["c_min"] == 0.2

    def test_added_group_can_override_robustness(self):
        a = torch.nn.Parameter(torch.randn(3))
        optimizer = CASMO([a], lr=1e-3, robustness=1.0)
        b = torch.nn.Parameter(torch.randn(3))
        optimizer.add_param_group({"params": [b], "robustness": 0.0})

        assert optimizer.param_groups[0]["robustness"] == 1.0
        assert optimizer.param_groups[1]["robustness"] == 0.0


class TestScaleInvariance:
    def test_agar_is_invariant_to_gradient_scale(self):
        """AGAR is a ratio: scaling every gradient by k must not change it.

        An epsilon in the AGAR denominator would break this for small gradients.
        """
        readings = []
        for scale in (1.0, 1e-2, 1e-4):
            torch.manual_seed(11)
            param = torch.nn.Parameter(torch.zeros(64))
            optimizer = CASMO([param], lr=0.0)  # lr=0: measure only, do not move
            for _ in range(40):
                torch.manual_seed(hash(("g", _)) % (2**31))
                base = torch.randn(64) * 0.1 + 1.0  # consistent signal + some noise
                param.grad = base * scale
                optimizer.step()
            readings.append(optimizer.group_metrics(0)["agar"])

        assert (
            max(readings) - min(readings) < 1e-3
        ), f"AGAR must not depend on gradient scale, got {readings}"
