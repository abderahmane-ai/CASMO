"""Tests for CASMO's headline property: resistance to noisy training signal.

These are behavioural tests. They assert the property CASMO exists to provide —
that confidence gating suppresses memorisation of label noise relative to AdamW.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from casmo import CASMO


class MLP(nn.Module):
    def __init__(self, d_in=20, hidden=64, d_out=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def make_noisy_split(seed, n=1200, d=20, k=3, noise_frac=0.3):
    """Half train (label-corrupted) / half test (clean labels)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    w = torch.randn(d, k, generator=g)
    y = (x @ w).argmax(1)
    clean = y.clone()
    corrupt = int(n * noise_frac)
    idx = torch.randperm(n, generator=g)[:corrupt]
    y[idx] = torch.randint(0, k, (corrupt,), generator=g)
    half = n // 2
    return (x[:half], y[:half]), (x[half:], clean[half:])


def train_and_score(optimizer_factory, seed, noise_frac=0.3, epochs=40, bs=64):
    torch.manual_seed(seed)
    (x_tr, y_tr), (x_te, y_te) = make_noisy_split(seed, noise_frac=noise_frac)
    model = MLP()
    optimizer = optimizer_factory(model.parameters())
    gen = torch.Generator().manual_seed(seed + 777)

    for _ in range(epochs):
        perm = torch.randperm(x_tr.shape[0], generator=gen)
        for i in range(0, x_tr.shape[0], bs):
            idx = perm[i : i + bs]
            optimizer.zero_grad()
            F.cross_entropy(model(x_tr[idx]), y_tr[idx]).backward()
            optimizer.step()

    with torch.no_grad():
        train_acc = (model(x_tr).argmax(1) == y_tr).float().mean().item()
        test_acc = (model(x_te).argmax(1) == y_te).float().mean().item()
    return train_acc, test_acc


class TestLabelNoiseRobustness:
    def test_beats_adamw_generalization_under_label_noise(self):
        """CASMO(robustness=1) must generalise better than AdamW at 30% label noise."""
        seeds = [0, 1, 2]
        casmo_scores, adamw_scores = [], []
        for seed in seeds:
            _, casmo_te = train_and_score(lambda p: CASMO(p, lr=3e-3, robustness=1.0), seed)
            _, adamw_te = train_and_score(
                lambda p: torch.optim.AdamW(p, lr=3e-3, weight_decay=0.0), seed
            )
            casmo_scores.append(casmo_te)
            adamw_scores.append(adamw_te)

        casmo_mean = sum(casmo_scores) / len(casmo_scores)
        adamw_mean = sum(adamw_scores) / len(adamw_scores)
        assert casmo_mean > adamw_mean + 0.05, (
            f"CASMO should clearly beat AdamW under label noise: "
            f"CASMO {casmo_mean:.3f} vs AdamW {adamw_mean:.3f}"
        )

    def test_resists_memorising_noisy_labels(self):
        """AdamW drives train acc to ~1.0 (memorising noise); CASMO should not."""
        casmo_tr, _ = train_and_score(lambda p: CASMO(p, lr=3e-3, robustness=1.0), seed=0)
        adamw_tr, _ = train_and_score(lambda p: torch.optim.AdamW(p, lr=3e-3), seed=0)

        assert adamw_tr > 0.99, f"baseline should memorise, got {adamw_tr}"
        assert (
            casmo_tr < adamw_tr - 0.05
        ), f"CASMO should resist memorisation: CASMO {casmo_tr:.3f} vs AdamW {adamw_tr:.3f}"

    def test_robustness_dial_trades_memorisation_for_generalisation(self):
        """Raising robustness must monotonically reduce noise memorisation."""
        train_accs = []
        for robustness in (0.0, 0.5, 1.0):
            tr, _ = train_and_score(lambda p, r=robustness: CASMO(p, lr=3e-3, robustness=r), seed=0)
            train_accs.append(tr)

        assert (
            train_accs[0] >= train_accs[1] >= train_accs[2]
        ), f"higher robustness should memorise less, got {train_accs}"

    def test_matches_adamw_on_clean_data(self):
        """Robustness must not cost generalisation when there is no noise."""
        _, casmo_te = train_and_score(lambda p: CASMO(p, lr=3e-3), seed=0, noise_frac=0.0)
        _, adamw_te = train_and_score(
            lambda p: torch.optim.AdamW(p, lr=3e-3), seed=0, noise_frac=0.0
        )
        assert (
            casmo_te >= adamw_te - 0.02
        ), f"CASMO should match AdamW on clean data: {casmo_te:.3f} vs {adamw_te:.3f}"


class TestGradientNoise:
    def test_confidence_drops_when_gradient_noise_is_injected(self):
        torch.manual_seed(0)
        model = nn.Linear(10, 2)
        optimizer = CASMO(model.parameters(), lr=1e-3)

        def run(noise):
            for _ in range(40):
                optimizer.zero_grad()
                F.cross_entropy(model(torch.randn(32, 10)), torch.randint(0, 2, (32,))).backward()
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.add_(torch.randn_like(p.grad) * noise)
                optimizer.step()
            return optimizer.group_metrics(0)["confidence"]

        clean_conf = run(0.0)
        noisy_conf = run(5.0)
        assert (
            noisy_conf < clean_conf
        ), f"confidence should fall under injected noise: {clean_conf:.3f} -> {noisy_conf:.3f}"

    def test_stable_under_heavy_gradient_noise(self):
        torch.manual_seed(0)
        model = nn.Linear(10, 2)
        optimizer = CASMO(model.parameters(), lr=1e-3, robustness=1.0)
        for _ in range(80):
            optimizer.zero_grad()
            F.cross_entropy(model(torch.randn(32, 10)), torch.randint(0, 2, (32,))).backward()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.add_(torch.randn_like(p.grad) * 10.0)
            optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
