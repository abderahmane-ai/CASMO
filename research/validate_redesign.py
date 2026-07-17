"""Reproducible validation of the CASMO v0.4 redesign.

This script is the evidence behind the design decisions documented in
``research/REDESIGN.md``. It compares the shipped :class:`casmo.CASMO` against
Adam/AdamW across four regimes that isolate the mechanisms CASMO claims:

    A. Clean convergence  -- confidence gating must not break normal training.
    B. Label noise        -- the headline claim: resist memorising bad labels.
    C. Gradient noise     -- injected isotropic noise (DP-SGD-like).
    D. High learning rate -- stability when the step size is aggressive.

Every configuration runs over multiple seeds and reports mean +/- population
stdev. Run with::

    python research/validate_redesign.py
"""

import os
import statistics
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from casmo import CASMO  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
LOSS_THRESHOLD = 0.05


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


def make_split(seed, n=1200, d=20, k=3, noise_frac=0.0):
    """Half train (optionally label-corrupted) / half test (always clean)."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=gen)
    w = torch.randn(d, k, generator=gen)
    y = (x @ w).argmax(1)
    clean = y.clone()
    if noise_frac > 0:
        corrupt = int(n * noise_frac)
        idx = torch.randperm(n, generator=gen)[:corrupt]
        y[idx] = torch.randint(0, k, (corrupt,), generator=gen)
    half = n // 2
    return (x[:half], y[:half]), (x[half:], clean[half:])


def build_optimizer(name, params, lr):
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "AdamW":
        return torch.optim.AdamW(params, lr=lr)
    if name.startswith("CASMO"):
        robustness = float(name.split("=")[1]) if "=" in name else 0.5
        return CASMO(params, lr=lr, robustness=robustness)
    raise ValueError(f"unknown optimizer: {name}")


def run_once(name, seed, lr=3e-3, epochs=60, batch_size=64, grad_noise=0.0, noise_frac=0.0):
    torch.manual_seed(seed)
    (x_tr, y_tr), (x_te, y_te) = make_split(seed, noise_frac=noise_frac)
    model = MLP()
    optimizer = build_optimizer(name, model.parameters(), lr)
    gen = torch.Generator().manual_seed(seed + 777)
    n = x_tr.shape[0]
    steps_to_threshold = None
    step = 0

    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            F.cross_entropy(model(x_tr[idx]), y_tr[idx]).backward()
            if grad_noise > 0:
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.add_(torch.randn_like(p.grad) * grad_noise)
            optimizer.step()
            step += 1
        with torch.no_grad():
            train_loss = F.cross_entropy(model(x_tr), y_tr).item()
        if steps_to_threshold is None and train_loss < LOSS_THRESHOLD:
            steps_to_threshold = step

    with torch.no_grad():
        train_acc = (model(x_tr).argmax(1) == y_tr).float().mean().item()
        test_acc = (model(x_te).argmax(1) == y_te).float().mean().item()
        final_loss = F.cross_entropy(model(x_tr), y_tr).item()

    total_steps = epochs * ((n + batch_size - 1) // batch_size)
    return {
        "final_loss": final_loss,
        "steps": steps_to_threshold or total_steps,
        "train_acc": train_acc,
        "test_acc": test_acc,
    }


def aggregate(name, **kwargs):
    runs = [run_once(name, seed, **kwargs) for seed in SEEDS]
    return {
        key: (
            statistics.mean([r[key] for r in runs]),
            statistics.pstdev([r[key] for r in runs]),
        )
        for key in runs[0]
    }


def fmt(stat):
    mean, stdev = stat
    return f"{mean:7.3f} +/-{stdev:5.3f}"


def report(title, optimizers, **kwargs):
    print(f"\n### {title}")
    header = (
        f"{'optimizer':16s} {'final_loss':16s} {'steps':16s} {'train_acc':16s} {'test_acc':16s}"
    )
    print(header)
    print("-" * len(header))
    for name in optimizers:
        r = aggregate(name, **kwargs)
        print(
            f"{name:16s} {fmt(r['final_loss'])} {fmt(r['steps'])} "
            f"{fmt(r['train_acc'])} {fmt(r['test_acc'])}"
        )


def main():
    optimizers = ["Adam", "AdamW", "CASMO=0.0", "CASMO=0.5", "CASMO=1.0"]
    print(f"CASMO redesign validation -- {len(SEEDS)} seeds per configuration")

    report("A. CLEAN (lr=3e-3): fewer steps = faster", optimizers, lr=3e-3)
    report("B. LABEL NOISE 30%: higher test_acc = better", optimizers, lr=3e-3, noise_frac=0.30)
    report("B2. LABEL NOISE 15%", optimizers, lr=3e-3, noise_frac=0.15)
    report("C. GRADIENT NOISE sigma=0.5", optimizers, lr=3e-3, grad_noise=0.5)
    report("D. HIGH LR (lr=3e-2): stability", optimizers, lr=3e-2)


if __name__ == "__main__":
    main()
