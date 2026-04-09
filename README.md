# CASMO: Confident Adaptive Selective Momentum Optimizer

[![PyPI version](https://badge.fury.io/py/casmo-optimizer.svg)](https://badge.fury.io/py/casmo-optimizer)
[![Downloads](https://pepy.tech/badge/casmo-optimizer)](https://pepy.tech/project/casmo-optimizer)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: Theory](https://img.shields.io/badge/Paper-Theory-green.svg)](CASMO_THEORY.md)

> **The Optimizer That Knows *What* To Learn.**

## Abstract

**CASMO** is a novel optimization algorithm designed to address the limitations of uniform gradient application in deep learning. Unlike standard optimizers (e.g., AdamW, SGD) that treat all gradients as equally valid, CASMO introduces **Adaptive Gradient-Aware Regularization (AGAR)**. This mechanism dynamically quantifies the "signal-to-noise" ratio of gradients over time, allowing the optimizer to selectively dampen updates that correspond to noise, outliers, or conflicting information while accelerating learning on generalizable patterns.

This approach yields state-of-the-art stability in challenging regimes such as **label noise**, **long-tail distributions**, **differential privacy**, and **continual learning**.

---

## 💡 Key Innovation: The AGAR Mechanism

The core of CASMO is the **Adaptive Gradient Alignment Ratio (AGAR)**, a metric that measures the consistency of gradient direction over a sliding window:

$$ \text{AGAR} = \frac{||\mathbb{E}[g]||^2}{||\mathbb{E}[g]||^2 + \text{Var}[g]} $$

*   **High AGAR ($\approx 1$)**: Indicates consistent, reliable signal (e.g., generalizable features). CASMO applies full updates.
*   **Low AGAR ($\approx 0$)**: Indicates high variance or conflict (e.g., label noise, privacy noise, catastrophic forgetting). CASMO automatically dampens the learning rate.

This allows CASMO to act as an **automatic filter** during training, improving robustness without manual hyperparameter tuning or data filtering.

---

## ⚡ Quick Start

### Installation

**Via PyPI (Recommended):**

```bash
pip install casmo-optimizer
```

**From Source (For Development):**

```bash
git clone https://github.com/abderahmane-ai/CASMO.git
cd CASMO
pip install -e .
```

### Usage

CASMO is a drop-in replacement for `torch.optim.AdamW`.

```python
from casmo import CASMO

# Initialize with standard parameters
optimizer = CASMO(
    model.parameters(), 
    lr=1e-3, 
    weight_decay=0.01,
    granularity='group'  # Recommended for efficiency
)

# Standard PyTorch training loop
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

---

## 🧪 Benchmarks & Reproducibility

We provide a comprehensive suite of benchmarks demonstrating CASMO's superiority in specific failure modes of standard optimizers. Each benchmark contains full reproduction scripts and detailed analysis.

| Benchmark | Domain | Challenge | Key Result | Link |
| :--- | :--- | :--- | :--- | :--- |
| **B1** | **Generalization** | Grokking with 30% Label Noise | 90.4% vs 24.0% val acc | [View](benchmarks/b1_grokking/) |
| **B2** | **Long-Tail** | CIFAR-100 Imbalance (100:1) | Better tail-class accuracy | [View](benchmarks/b2_long_tail_cifar100/) |
| **B3** | **Privacy** | DP-SGD (ε ≈ 0.37) | Maintains accuracy under DP | [View](benchmarks/b3_dp_sgd/) |
| **B4** | **Continual Learning** | Sequential LLM Fine-tuning | 13% less forgetting | [View](benchmarks/b4_continual_learning/) |
| **B5** | **Finance** | Portfolio Optimization | Lower turnover, higher Sharpe | [View](benchmarks/b5_noisy_timeseries/) |

See [benchmarks/README.md](benchmarks/README.md) for detailed results and reproduction instructions.

---

## 🛠️ Configuration

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `lr` | `1e-3` | Base learning rate. |
| `weight_decay` | `0.0` | Decoupled weight decay (same as AdamW). |
| `granularity` | `'group'` | `'group'` (Fast, recommended) or `'parameter'` (Precise). |
| `tau_init_steps` | `500` | Steps for initial AGAR calibration (auto-tuned). |
| `c_min` | `0.1` | Minimum confidence floor (prevents dead neurons). |

---

## Citation

If you use CASMO in your research, please cite:

```bibtex
@software{casmo2025,
  title={CASMO: Confident Adaptive Selective Momentum Optimizer},
  author={Ainouche, Abderahmane},
  year={2025},
  url={https://github.com/abderahmane-ai/CASMO}
}
```

---

**Made with ❤️ for the ML research community.**
