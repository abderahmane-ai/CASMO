# CASMO: Confidence-Adjusted Signal-to-noise Momentum Optimizer

[![PyPI version](https://badge.fury.io/py/casmo-optimizer.svg)](https://badge.fury.io/py/casmo-optimizer)
[![Downloads](https://pepy.tech/badge/casmo-optimizer)](https://pepy.tech/project/casmo-optimizer)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: Theory](https://img.shields.io/badge/Paper-Theory-green.svg)](https://github.com/abderahmane-ai/CASMO/blob/main/CASMO_THEORY.md)

> **The optimizer that knows *what* to learn.**

## Abstract

Standard optimizers adapt to gradient **magnitude**. CASMO adapts to gradient
**consistency**. Adam cannot distinguish a large gradient that reflects a real, repeatable
signal from a large gradient that is just noise — CASMO measures the difference directly
and scales each coordinate's step by its **signal fraction**.

CASMO is a drop-in replacement for `torch.optim.AdamW` with the **same optimizer-state
memory as Adam** (two EMAs per parameter), no calibration phase, and one interpretable
knob controlling the speed/robustness trade-off.

---

## Key Innovation: AGAR

The **Adaptive Gradient Alignment Ratio** is a per-coordinate signal fraction:

$$\text{AGAR}_i = \frac{S_i}{S_i + N_i}, \qquad S_i = \hat{m}_i^2 \approx (\mathbb{E}[g_i])^2, \quad N_i = \hat{s}_i \approx \text{Var}[g_i]$$

- **High AGAR (≈ 1)** — consistent, reliable signal. Full update.
- **Low AGAR (≈ 0)** — high variance or conflict (label noise, gradient noise, task
  interference). Damped update.

AGAR is bounded in `[0, 1]` by construction, on an **absolute scale** — so it needs no
threshold, no calibration, and no tuning. CASMO combines it along two axes:

| Axis | Formula | Buys you |
|---|---|---|
| **trust** (absolute) | `c_min + (1-c_min)·mean(AGAR)` | Robustness to label noise |
| **focus** (relative) | `clip(AGAR_i / mean(AGAR), rel_floor, 1)` | Speed, stability, expressivity |

$$C_i = \text{trust}^{\text{robustness}} \cdot \text{focus}_i$$

See [CASMO_THEORY.md](https://github.com/abderahmane-ai/CASMO/blob/main/CASMO_THEORY.md) for the derivation and
[research/REDESIGN.md](https://github.com/abderahmane-ai/CASMO/blob/main/research/REDESIGN.md) for the experiments behind every choice.

---

## Quick Start

### Installation

```bash
pip install casmo-optimizer
```

From source:

```bash
git clone https://github.com/abderahmane-ai/CASMO.git
cd CASMO
pip install -e .
```

### Usage

```python
from casmo import CASMO

optimizer = CASMO(model.parameters(), lr=1e-3, weight_decay=0.01)

for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

Tune one knob for your noise regime:

```python
CASMO(model.parameters(), lr=1e-3, robustness=1.0)  # noisy labels: maximum robustness
CASMO(model.parameters(), lr=1e-3, robustness=0.0)  # clean data / DP-SGD: AdamW-like pace
```

Inspect what the optimizer thinks of your gradients:

```python
metrics = optimizer.group_metrics(0)
print(metrics["agar"], metrics["confidence"])
```

---

## Measured Results

From [`research/validate_redesign.py`](https://github.com/abderahmane-ai/CASMO/blob/main/research/validate_redesign.py) — 5 seeds,
mean ± stdev, clean test labels. Reproduce with `python research/validate_redesign.py`.

| Regime | Adam | CASMO (`robustness=1.0`) |
|---|---|---|
| **30% label noise** (test acc) | 0.675 ± 0.020 | **0.810 ± 0.008** |
| **15% label noise** (test acc) | 0.764 ± 0.022 | **0.857 ± 0.016** |
| **Clean data** (test acc) | 0.931 ± 0.010 | 0.934 ± 0.007 |
| **High LR** (steps to converge) | 70 | **38** (`robustness=0.5`) |

CASMO reaches **+13.5 points** of test accuracy over Adam at 30% label noise, and refuses
to memorize the corrupted training set (0.835 train accuracy vs. Adam's 1.000).

**Honest limitation:** under *isotropic gradient noise* (DP-SGD-style), high `robustness`
hurts — Adam 0.641 vs. CASMO(ρ=1) 0.383. Label noise and injected gradient noise want
opposite policies. Use low `robustness` for DP-SGD. See
[REDESIGN.md §6](https://github.com/abderahmane-ai/CASMO/blob/main/research/REDESIGN.md).

---

## Benchmarks

Larger-scale benchmarks with full reproduction scripts:

| Benchmark | Domain | Challenge | Link |
| :--- | :--- | :--- | :--- |
| **B1** | Generalization | Grokking with 30% label noise | [View](https://github.com/abderahmane-ai/CASMO/tree/main/benchmarks/b1_grokking) |
| **B2** | Long-tail | CIFAR-100 imbalance (100:1) | [View](https://github.com/abderahmane-ai/CASMO/tree/main/benchmarks/b2_long_tail_cifar100) |
| **B3** | Privacy | DP-SGD | [View](https://github.com/abderahmane-ai/CASMO/tree/main/benchmarks/b3_dp_sgd) |
| **B4** | Continual learning | Sequential LLM fine-tuning | [View](https://github.com/abderahmane-ai/CASMO/tree/main/benchmarks/b4_continual_learning) |
| **B5** | Finance | Portfolio optimization | [View](https://github.com/abderahmane-ai/CASMO/tree/main/benchmarks/b5_noisy_timeseries) |

> **Note:** the benchmark reports under `benchmarks/*/reports/` were produced with the
> v0.3 calibration-based algorithm and have **not** been re-run against v0.4. Treat those
> numbers as historical. The results table above is the current, reproducible evidence.

---

## Configuration

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `lr` | `1e-3` | Base learning rate. |
| `betas` | `(0.9, 0.999)` | EMA coefficients for the first moment and the variance. |
| `eps` | `1e-8` | Numerical stability term. |
| `weight_decay` | `0.0` | Decoupled weight decay (AdamW-style). |
| `robustness` | `0.5` | Noise-suppression strength. `0` = AdamW pace, `1` = maximally robust. |
| `c_min` | `0.1` | Floor on the absolute trust factor. |
| `rel_floor` | `0.1` | Floor on the relative focus factor. |
| `nan_guard` | `False` | Raise on non-finite gradients (costs a host sync per step). |

Setting `robustness=0` and `rel_floor=1` makes CASMO exactly equivalent to AdamW.

---

## Migrating from v0.3

The calibration parameters (`tau_init_steps`, `tau_clip_range`, `granularity`,
`agar_clamp_factor`, `total_steps`) are **removed**. They are still accepted with a
`DeprecationWarning` and ignored, so existing code keeps running. See the
[migration guide](https://github.com/abderahmane-ai/CASMO/blob/main/docs/migration-guide.md).

---

## Citation

```bibtex
@software{casmo2025,
  title={CASMO: Confidence-Adjusted Signal-to-noise Momentum Optimizer},
  author={Ainouche, Abderahmane},
  year={2025},
  url={https://github.com/abderahmane-ai/CASMO}
}
```

---

**Made with care for the ML research community.**
