"""Tests for the per-coordinate AGAR signal-fraction estimate."""

import torch

from casmo import CASMO


class TestAGAR:
    """AGAR = signal / (signal + noise), averaged for reporting."""

    def test_perfect_signal_gives_high_agar(self):
        """Perfectly consistent gradients are pure signal -> AGAR near 1."""
        param = torch.nn.Parameter(torch.randn(10, 10))
        optimizer = CASMO([param], lr=1e-3)
        grad = torch.ones(10, 10) * 0.5

        for _ in range(100):
            param.grad = grad.clone()
            optimizer.step()

        agar = optimizer.group_metrics(0)["agar"]
        assert agar > 0.95, f"expected AGAR ~1 for zero-variance gradients, got {agar}"

    def test_pure_noise_gives_low_agar(self):
        """Directionless random gradients are pure noise -> AGAR near 0."""
        torch.manual_seed(42)
        param = torch.nn.Parameter(torch.randn(10, 10))
        optimizer = CASMO([param], lr=1e-3)

        for _ in range(100):
            param.grad = torch.randn(10, 10)
            optimizer.step()

        agar = optimizer.group_metrics(0)["agar"]
        assert agar < 0.5, f"expected low AGAR for random gradients, got {agar}"

    def test_agar_is_bounded(self):
        """AGAR must stay in [0, 1] across mixed signal/noise regimes."""
        torch.manual_seed(123)
        param = torch.nn.Parameter(torch.randn(20, 20))
        optimizer = CASMO([param], lr=1e-3)
        values = []

        for i in range(200):
            param.grad = torch.ones(20, 20) * 0.3 if i % 2 else torch.randn(20, 20) * 0.1
            optimizer.step()
            values.append(optimizer.group_metrics(0)["agar"])

        assert values, "AGAR should be reported every step"
        assert all(0.0 <= v <= 1.0 for v in values), "AGAR must remain in [0, 1]"

    def test_signal_scores_higher_than_noise(self):
        """AGAR must rank a consistent gradient above a random one."""
        torch.manual_seed(7)
        signal_param = torch.nn.Parameter(torch.randn(16, 16))
        noise_param = torch.nn.Parameter(torch.randn(16, 16))
        signal_opt = CASMO([signal_param], lr=1e-4)
        noise_opt = CASMO([noise_param], lr=1e-4)

        for _ in range(80):
            signal_param.grad = torch.ones(16, 16) * 0.2
            noise_param.grad = torch.randn(16, 16) * 0.2
            signal_opt.step()
            noise_opt.step()

        assert signal_opt.group_metrics(0)["agar"] > noise_opt.group_metrics(0)["agar"]

    def test_metrics_update_every_step(self):
        """Reported metrics must track training, not freeze at the first value."""
        torch.manual_seed(0)
        params = [torch.nn.Parameter(torch.randn(4, 4)), torch.nn.Parameter(torch.randn(3))]
        optimizer = CASMO(params, lr=1e-3)
        seen = []

        for _ in range(5):
            for p in params:
                p.grad = torch.randn_like(p)
            optimizer.step()
            seen.append(optimizer.group_metrics(0)["agar"])

        assert len(set(seen)) > 1, f"AGAR should vary across steps, got {seen}"

    def test_metrics_none_before_first_step(self):
        param = torch.nn.Parameter(torch.randn(3))
        optimizer = CASMO([param], lr=1e-3)
        assert optimizer.group_metrics(0)["agar"] is None
        assert optimizer.group_metrics(0)["confidence"] is None

    def test_metrics_reported_per_group(self):
        a = torch.nn.Parameter(torch.randn(5, 5))
        b = torch.nn.Parameter(torch.randn(5, 5))
        optimizer = CASMO([{"params": [a]}, {"params": [b]}], lr=1e-3)

        for _ in range(10):
            a.grad = torch.ones(5, 5) * 0.5
            b.grad = torch.randn(5, 5)
            optimizer.step()

        assert optimizer.group_metrics(0)["agar"] is not None
        assert optimizer.group_metrics(1)["agar"] is not None
        assert optimizer.group_metrics(0)["agar"] > optimizer.group_metrics(1)["agar"]
