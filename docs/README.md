# CASMO Documentation

**CASMO** (Confidence-Adjusted Signal-to-noise Momentum Optimizer) is a drop-in
replacement for `torch.optim.AdamW` that scales each coordinate's update by the
signal-to-noise ratio of its gradient.

```python
from casmo import CASMO

optimizer = CASMO(model.parameters(), lr=1e-3, weight_decay=0.01)
```

## Contents

| Document | What it covers |
|---|---|
| [Getting Started](getting-started.md) | Install, first training loop, choosing `robustness`, monitoring |
| [API Reference](api-reference.md) | Every parameter, method, and exception |
| [Migration Guide](migration-guide.md) | Coming from AdamW, or upgrading from CASMO v0.3 |
| [FAQ](faq.md) | Concepts, usage, performance, troubleshooting |
| [Theory](../CASMO_THEORY.md) | The derivation of AGAR and the confidence map |
| [Redesign evidence](../research/REDESIGN.md) | The experiments that produced the v0.4 design |

## The one-paragraph version

Adam adapts to gradient *magnitude* and cannot tell a strong consistent signal from large
random noise. CASMO tracks the gradient's first moment `m` (signal) and its centered
variance `s` (noise), and forms a per-coordinate signal fraction
`AGAR = m̂²/(m̂² + ŝ) ∈ [0,1]`. That fraction drives two multipliers: **trust**, an
absolute per-tensor pace that shrinks when gradients are noise-dominated, and **focus**, a
relative per-coordinate weight that shifts the step toward reliable directions. The
`robustness` dial sets how strongly the absolute axis applies — `0` behaves like AdamW,
`1` is maximally noise robust.

## The one knob

```python
CASMO(params, lr=1e-3, robustness=1.0)   # noisy labels, long-tail, continual learning
CASMO(params, lr=1e-3)                    # default 0.5 -- general purpose
CASMO(params, lr=1e-3, robustness=0.0)   # clean data, DP-SGD, maximum speed
```

## Measured results

5 seeds, mean ± stdev, clean test labels. Reproduce with
`python research/validate_redesign.py`.

| Regime | Adam | CASMO |
|---|---|---|
| 30% label noise | 0.675 ± 0.020 | **0.810 ± 0.008** (ρ=1.0) |
| 15% label noise | 0.764 ± 0.022 | **0.857 ± 0.016** (ρ=1.0) |
| Clean data | 0.931 ± 0.010 | 0.934 ± 0.007 |
| High LR (steps) | 70 | **38** (ρ=0.5) |
| Gradient noise σ=0.5 | **0.641 ± 0.037** | 0.383 ± 0.035 (ρ=1.0) |

The last row is a real limitation, not an omission: label noise and isotropic gradient
noise want opposite policies. Use low `robustness` for DP-SGD-style workloads.
