"""Tests for the confidence map: trust (absolute) x focus (relative)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from casmo import CASMO


def _train_noisy(optimizer, model, steps=60, noise=2.0, seed=0):
    """Drive a model with heavily noise-corrupted gradients."""
    torch.manual_seed(seed)
    for _ in range(steps):
        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        optimizer.zero_grad()
        F.cross_entropy(model(x), y).backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad.add_(torch.randn_like(p.grad) * noise)
        optimizer.step()


class TestConfidence:
    def test_confidence_is_bounded(self):
        """Confidence = trust^robustness * focus must stay within [0, 1]."""
        torch.manual_seed(0)
        param = torch.nn.Parameter(torch.randn(10, 10))
        optimizer = CASMO([param], lr=1e-3)

        for _ in range(50):
            param.grad = torch.randn(10, 10)
            optimizer.step()
            conf = optimizer.group_metrics(0)["confidence"]
            assert 0.0 <= conf <= 1.0, f"confidence out of range: {conf}"

    def test_robustness_dial_is_monotonic_under_noise(self):
        """Higher robustness must suppress noisy updates more aggressively."""
        confidences = []
        for robustness in (0.0, 0.5, 1.0):
            torch.manual_seed(3)
            model = nn.Linear(10, 2)
            optimizer = CASMO(model.parameters(), lr=1e-3, robustness=robustness)
            _train_noisy(optimizer, model, seed=3)
            confidences.append(optimizer.group_metrics(0)["confidence"])

        assert (
            confidences[0] > confidences[1] > confidences[2]
        ), f"confidence should fall as robustness rises, got {confidences}"

    def test_robustness_zero_disables_absolute_suppression(self):
        """robustness=0 -> trust^0 = 1, so only the relative focus applies."""
        torch.manual_seed(1)
        model = nn.Linear(10, 2)
        optimizer = CASMO(model.parameters(), lr=1e-3, robustness=0.0)
        _train_noisy(optimizer, model, seed=1)
        # focus is mean-normalised and capped at 1, so mean confidence stays high
        assert optimizer.group_metrics(0)["confidence"] > 0.5

    def test_c_min_floors_the_trust_factor(self):
        """Even under pure noise, confidence must not collapse below the floor."""
        torch.manual_seed(2)
        model = nn.Linear(10, 2)
        c_min, rel_floor = 0.3, 0.5
        optimizer = CASMO(
            model.parameters(), lr=1e-3, c_min=c_min, rel_floor=rel_floor, robustness=1.0
        )
        _train_noisy(optimizer, model, noise=8.0, seed=2)

        conf = optimizer.group_metrics(0)["confidence"]
        assert conf >= c_min * rel_floor, f"confidence {conf} fell below floor"

    def test_confidence_higher_for_clean_than_noisy_gradients(self):
        torch.manual_seed(5)
        clean_model, noisy_model = nn.Linear(10, 2), nn.Linear(10, 2)
        noisy_model.load_state_dict(clean_model.state_dict())
        clean_opt = CASMO(clean_model.parameters(), lr=1e-3)
        noisy_opt = CASMO(noisy_model.parameters(), lr=1e-3)

        _train_noisy(clean_opt, clean_model, noise=0.0, seed=5)
        _train_noisy(noisy_opt, noisy_model, noise=3.0, seed=5)

        assert clean_opt.group_metrics(0)["confidence"] > noisy_opt.group_metrics(0)["confidence"]

    def test_rel_floor_bounds_per_coordinate_downweighting(self):
        """rel_floor=1.0 disables relative focus entirely (focus == 1)."""
        torch.manual_seed(4)
        model = nn.Linear(10, 2)
        optimizer = CASMO(model.parameters(), lr=1e-3, rel_floor=1.0, robustness=0.0)
        _train_noisy(optimizer, model, seed=4)
        conf = optimizer.group_metrics(0)["confidence"]
        assert abs(conf - 1.0) < 1e-5, f"expected confidence == 1, got {conf}"


class TestVarianceEstimator:
    """The noise EMA must estimate Var[g], not a beta1-scaled version of it."""

    def test_noise_ema_tracks_true_variance(self):
        """s must converge to Var[g] for a known distribution.

        Measuring the deviation against the *updated* m would give
        grad - m_t == beta1 * (grad - m_{t-1}), i.e. beta1^2 * Var[g] -- 19% low at
        beta1=0.9 and far worse as beta1 falls.
        """
        torch.manual_seed(0)
        true_mean, true_std = 1.0, 2.0
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = CASMO([param], lr=0.0)  # measure only

        for _ in range(20000):
            param.grad = torch.randn(1) * true_std + true_mean
            optimizer.step()

        state = optimizer.state[param]
        s_hat = state["s"] / (1 - 0.999 ** state["step"])
        ratio = s_hat.item() / true_std**2
        # Upper bound is loose: measuring against m_{t-1} adds Var[m_{t-1}], a few
        # percent. The lower bound is what matters -- the old ordering scored 0.88.
        assert 0.9 < ratio < 1.2, f"noise EMA should track Var[g]; ratio to true = {ratio:.3f}"

    def test_robustness_does_not_depend_on_betas(self):
        """`robustness` must mean the same thing regardless of beta1.

        A beta1-dependent variance bias would silently weaken noise suppression when
        a user tunes momentum, which has nothing to do with robustness.
        """
        readings = []
        for beta1 in (0.9, 0.7, 0.5):
            torch.manual_seed(4)
            model = nn.Linear(10, 2)
            optimizer = CASMO(model.parameters(), lr=1e-3, betas=(beta1, 0.999), robustness=1.0)
            _train_noisy(optimizer, model, noise=3.0, seed=4)
            readings.append(optimizer.group_metrics(0)["agar"])

        spread = max(readings) - min(readings)
        assert (
            spread < 0.15
        ), f"AGAR should not swing with beta1 (bias would make it), got {readings}"
