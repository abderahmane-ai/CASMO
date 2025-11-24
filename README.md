# CASMO: Confident Adaptive Selective Momentum Optimizer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: Theory](https://img.shields.io/badge/Paper-Theory-green.svg)](CASMO_THEORY.md)

> **The Optimizer That Knows *What* To Learn.**

CASMO is a production-ready PyTorch optimizer that extends AdamW with **automatic signal-to-noise detection**. By measuring the consistency of gradients over time (AGAR), CASMO dynamically adjusts learning rates to focus on generalizing patterns while ignoring noise, memorization, and outliers.

---

## 🚀 Why CASMO?

Standard optimizers like AdamW treat all gradients as equal. CASMO asks: *"Is this update reliable?"*

*   **🛡️ Noise Robustness**: Automatically filters out label noise (e.g., corrupt datasets).
*   **🔒 Differential Privacy**: Distinguishes signal from DP noise, achieving higher accuracy at the same $\epsilon$.
*   **📉 Long-Tail Learning**: Prevents overfitting to noisy minority classes.
*   **🧠 Continual Learning**: Reduces catastrophic forgetting by detecting gradient conflicts.

## 📊 Key Results

| Benchmark | Challenge | CASMO | AdamW | Improvement |
| :--- | :--- | :---: | :---: | :---: |
| **B2: Grokking** | 30% Label Noise | **90.3%** | 24.0% | **+66.3%** |
| **B3: Long-Tail** | CIFAR-100 (100:1) | **28.9%** | 23.9% | **+5.0%** |
| **B4: Privacy** | DP-SGD ($\epsilon=0.37$) | **27.3%** | 26.3% | **+1.0%** |

> [!TIP]
> **See the Math**: Check out [CASMO_THEORY.md](CASMO_THEORY.md) for the full mathematical derivation of AGAR and confidence mapping.

---

## ⚡ Quick Start

### Installation

```bash
# From source
git clone https://github.com/abderahmane-ai/CASMO.git
cd CASMO
pip install -e .
```

### Usage

CASMO is a drop-in replacement for `torch.optim.AdamW`.

```python
from casmo import CASMO

# Initialize
optimizer = CASMO(
    model.parameters(), 
    lr=1e-3, 
    weight_decay=0.01,
    granularity='group'  # 'group' is faster, 'parameter' is more precise
)

# Train loop (Standard PyTorch)
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

---

## 🔬 How It Works: The AGAR Metric

CASMO computes the **Adaptive Gradient Alignment Ratio (AGAR)** to measure gradient consistency:

$$ \text{AGAR} = \frac{||\mathbb{E}[g]||^2}{||\mathbb{E}[g]||^2 + \text{Var}[g]} $$

*   **High AGAR ($\approx 1$)**: Consistent signal (Generalization) $\rightarrow$ **High Confidence**
*   **Low AGAR ($\approx 0$)**: Random noise (Memorization/DP Noise) $\rightarrow$ **Low Confidence**

This confidence score dynamically scales the learning rate for each parameter group.

---

## 🧪 Benchmarks

We provide reproducible benchmarks covering diverse challenges:

### [B2: Grokking & Generalization](benchmarks/b2_grokking/)
**Challenge**: Modular arithmetic with 30% label noise.
**Result**: AdamW memorizes noise (100% train, 24% val). CASMO groks the rule (90% val).

### [B3: Long-Tail Recognition](benchmarks/b3_long_tail_cifar100/)
**Challenge**: CIFAR-100 with 100:1 class imbalance.
**Result**: CASMO improves few-shot accuracy by **44% relative** to AdamW.

### [B4: Differential Privacy](benchmarks/b4_dp_sgd/)
**Challenge**: DP-SGD training with $\epsilon \approx 0.37$ (High Noise).
**Result**: CASMO extracts more utility (+1.0% acc) from the same privacy budget.

### [B6: Noisy Instruction Tuning](benchmarks/b6_noisy_instruct/)
**Challenge**: Fine-tuning Gemma-2B (4-bit) on noisy instructions.
**Result**: CASMO filters out bad data, improving instruction following.

---

## 🛠️ Configuration

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `lr` | `1e-3` | Base learning rate |
| `weight_decay` | `0.0` | Decoupled weight decay |
| `granularity` | `'group'` | `'group'` (Fast, recommended) or `'parameter'` (Precise) |
| `tau_init_steps` | `500` | Steps for initial calibration (auto-tuned) |
| `c_min` | `0.1` | Minimum confidence floor (auto-adjusted) |

---

## Citation

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
