# CASMO: Confident Adaptive Selective Momentum Optimizer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: Theory](https://img.shields.io/badge/Paper-Theory-green.svg)](CASMO_THEORY.md)

> **The Optimizer That Knows *What* To Learn.**

## Abstract

**CASMO** is a novel optimization algorithm designed to address the limitations of uniform gradient application in deep learning. Unlike standard optimizers (e.g., AdamW, SGD) that treat all gradients as equally valid, CASMO introduces **Adaptive Gradient-Aware Regularization (AGAR)**. This mechanism dynamically quantifies the "signal-to-noise" ratio of gradients over time, allowing the optimizer to selectively dampen updates that correspond to noise, outliers, or conflicting information while accelerating learning on generalizable patterns.

This approach yields state-of-the-art stability in challenging regimes such as **label noise**, **long-tail distributions**, **differential privacy**, and **continual learning**.

---

## � Key Innovation: The AGAR Mechanism

The core of CASMO is the **Adaptive Gradient Alignment Ratio (AGAR)**, a metric that measures the consistency of gradient direction over a sliding window:

$$ \text{AGAR} = \frac{||\mathbb{E}[g]||^2}{||\mathbb{E}[g]||^2 + \text{Var}[g]} $$

*   **High AGAR ($\approx 1$)**: Indicates consistent, reliable signal (e.g., generalizable features). CASMO applies full updates.
*   **Low AGAR ($\approx 0$)**: Indicates high variance or conflict (e.g., label noise, privacy noise, catastrophic forgetting). CASMO automatically dampens the learning rate.

This allows CASMO to act as an **automatic filter** during training, improving robustness without manual hyperparameter tuning or data filtering.

---

## ⚡ Quick Start

### Installation

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

| Benchmark | Domain | Challenge | Link |
| :--- | :--- | :--- | :--- |
| **B2** | **Generalization** | Grokking with 30% Label Noise | [View Benchmark](benchmarks/b2_grokking/) |
| **B3** | **Long-Tail** | CIFAR-100 Class Imbalance (100:1) | [View Benchmark](benchmarks/b3_long_tail_cifar100/) |
| **B4** | **Privacy** | DP-SGD with High Noise ($\epsilon \approx 0.37$) | [View Benchmark](benchmarks/b4_dp_sgd/) |
| **B6** | **Fine-Tuning** | Noisy Instruction Tuning (LLMs) | [View Benchmark](benchmarks/b6_noisy_instruct/) |
| **B7** | **Continual Learning** | Catastrophic Forgetting (Seq. Tasks) | [View Benchmark](benchmarks/b7_continual_learning/) |

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
