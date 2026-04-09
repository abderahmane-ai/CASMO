# CASMO Benchmarks

This directory contains comprehensive benchmarks demonstrating CASMO's effectiveness across diverse challenging scenarios where standard optimizers struggle.

## Overview

| Benchmark | Domain | Challenge | Key Result | Status |
|-----------|--------|-----------|------------|--------|
| **B1** | Generalization | Grokking with 30% label noise | 90.4% vs 24.0% val acc | ✅ Complete |
| **B2** | Long-Tail | CIFAR-100 imbalance (100:1) | Better tail-class accuracy | ✅ Complete |
| **B3** | Privacy | DP-SGD (ε ≈ 0.37) | Maintains accuracy under DP | ✅ Complete |
| **B4** | Continual Learning | Sequential LLM fine-tuning | 13% less forgetting | ✅ Complete |
| **B5** | Finance | Portfolio optimization | Lower turnover, higher Sharpe | ✅ Complete |

## Quick Start

### Prerequisites

```bash
# Install CASMO
pip install -e ..

# Install benchmark dependencies
pip install torchvision matplotlib transformers datasets opacus yfinance
```

### Running Benchmarks

Each benchmark can be run independently:

```bash
# B1: Grokking
cd b1_grokking
python train.py --epochs 2000

# B2: Long-Tail CIFAR-100
cd b2_long_tail_cifar100
python train.py --epochs 200

# B3: DP-SGD
cd b3_dp_sgd
python train.py --epochs 20

# B4: Continual Learning (requires GPU)
cd b4_continual_learning
python train.py --epochs_per_task 2

# B5: Portfolio Optimization
cd b5_noisy_timeseries
python train.py --epochs 50
```

## Benchmark Details

### B1: Grokking with Label Noise

**Task**: Modular arithmetic (a + b) mod 97 with 30% label corruption

**Challenge**: Models must learn the underlying algorithmic rule while ignoring random noise

**Setup**:
- Model: 1-layer Transformer
- Dataset: 50% train, 50% test split
- Noise: 30% of training labels randomly corrupted

**Results**:
| Metric | CASMO | AdamW |
|--------|-------|-------|
| Train Accuracy | 70.7% | 100% |
| Val Accuracy | 90.4% | 24.0% |

**Key Insight**: AdamW memorizes noise (100% train, 24% val), while CASMO learns the rule (70% train, 90% val)

**Paper**: [View Report](b1_grokking/reports/REPORT.md)

---

### B2: Long-Tail CIFAR-100

**Task**: Image classification with extreme class imbalance

**Challenge**: Standard optimizers overfit to majority classes, ignoring tail classes

**Setup**:
- Model: ResNet-32
- Dataset: CIFAR-100 with exponential imbalance (100:1 ratio)
- Classes: Many-shot (>100 samples), Medium-shot (20-100), Few-shot (<20)

**Results**:
| Metric | CASMO | AdamW |
|--------|-------|-------|
| Overall Accuracy | ~45% | ~43% |
| Few-shot Accuracy | Better | Worse |
| Forgetting | Lower | Higher |

**Key Insight**: CASMO balances learning across all classes, improving tail-class performance

**Paper**: [View Report](b2_long_tail_cifar100/reports/REPORT.md)

---

### B3: Differential Privacy (DP-SGD)

**Task**: CIFAR-10 classification with differential privacy guarantees

**Challenge**: DP noise makes gradients extremely noisy, degrading performance

**Setup**:
- Model: ResNet-18 (adapted for CIFAR-10)
- Privacy: Opacus PrivacyEngine (ε ≈ 0.37, δ = 1e-5)
- Baselines: DP-SGD, DP-AdamW, CASMO

**Results**:
| Optimizer | Final Accuracy | Privacy Budget (ε) |
|-----------|----------------|-------------------|
| CASMO | ~65% | 0.37 |
| DP-AdamW | ~62% | 0.37 |
| DP-SGD | ~58% | 0.37 |

**Key Insight**: CASMO's AGAR mechanism filters DP noise, maintaining accuracy under strict privacy

**Paper**: [View Report](b3_dp_sgd/reports/REPORT.md)

---

### B4: Continual Learning

**Task**: Sequential fine-tuning of LLMs without catastrophic forgetting

**Challenge**: Models forget previous tasks when learning new ones

**Setup**:
- Model: Gemma-2-2B (4-bit quantized) + LoRA
- Tasks: Math → Code → QA → Writing (sequential)
- Evaluation: Test on all previous tasks after each training phase

**Results**:
| Metric | CASMO | AdamW | Improvement |
|--------|-------|-------|-------------|
| Average Accuracy | 94.39% | 95.03% | -0.64% |
| Backward Transfer | -0.80 | -1.14 | +42% |
| Forgetting (Max Drop) | 1.29 | 1.46 | -13% |

**Key Insight**: CASMO detects conflicting gradients and protects previous knowledge, reducing forgetting by 13%

**Paper**: [View Report](b4_continual_learning/reports/REPORT.md)

---

### B5: Portfolio Optimization

**Task**: Multi-asset portfolio allocation with market noise

**Challenge**: High-frequency noise causes excessive trading (high turnover)

**Setup**:
- Model: LSTM allocator
- Assets: SPY, TLT, GLD, VNQ, BTC-USD
- Objective: Maximize Sharpe ratio net of transaction costs
- Costs: 10bps per turnover

**Results**:
| Metric | CASMO | AdamW | 1/N | Buy & Hold |
|--------|-------|-------|-----|------------|
| Sharpe Ratio | Higher | Lower | Baseline | Baseline |
| Turnover | Lower | Higher | 0% | 0% |
| Net Return | Better | Worse | Moderate | Moderate |

**Key Insight**: CASMO maintains stable allocations (low turnover) while AdamW churns the portfolio chasing noise

**Paper**: [View Report](b5_noisy_timeseries/reports/REPORT.md)

---

## Reproducing Results

### System Requirements

- **CPU**: Any modern CPU (multi-core recommended)
- **GPU**: Required for B4 (Continual Learning), optional for others
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: ~10GB for datasets

### Installation

```bash
# Clone repository
git clone https://github.com/abderahmane-ai/CASMO.git
cd CASMO

# Install CASMO
pip install -e .

# Install benchmark dependencies
pip install torchvision matplotlib transformers datasets opacus yfinance
```

### Running All Benchmarks

```bash
# B1: Grokking (~30 minutes on CPU)
cd benchmarks/b1_grokking
python train.py --epochs 2000

# B2: Long-Tail (~2 hours on GPU)
cd ../b2_long_tail_cifar100
python train.py --epochs 200

# B3: DP-SGD (~1 hour on GPU)
cd ../b3_dp_sgd
python train.py --epochs 20

# B4: Continual Learning (~4 hours on GPU, requires 16GB VRAM)
cd ../b4_continual_learning
python train.py --epochs_per_task 2

# B5: Portfolio (~30 minutes on CPU)
cd ../b5_noisy_timeseries
python train.py --epochs 50
```

## Understanding the Results

### What Makes a Good Benchmark?

Each benchmark demonstrates a specific failure mode of standard optimizers:

1. **B1 (Grokking)**: Memorization vs generalization
2. **B2 (Long-Tail)**: Majority class bias
3. **B3 (DP-SGD)**: Noise robustness
4. **B4 (Continual)**: Catastrophic forgetting
5. **B5 (Finance)**: Overfitting to noise

### Interpreting AGAR Values

During training, you can monitor AGAR:
- **AGAR ≈ 1.0**: High-quality gradients (consistent direction)
- **AGAR ≈ 0.5**: Mixed quality
- **AGAR ≈ 0.0**: Noisy gradients (random direction)

CASMO automatically reduces learning rate when AGAR is low.

### Common Patterns

Across all benchmarks, CASMO shows:
1. **Lower training accuracy** (doesn't overfit to noise)
2. **Higher validation accuracy** (better generalization)
3. **More stable training** (less variance)
4. **Better worst-case performance** (tail classes, forgetting)

## Benchmark Design Principles

### 1. Real-World Relevance

Each benchmark addresses a practical problem:
- Noisy labels (crowdsourced data)
- Imbalanced data (medical diagnosis, fraud detection)
- Privacy (healthcare, finance)
- Continual learning (production ML systems)
- Financial markets (portfolio management)

### 2. Fair Comparison

All benchmarks:
- Use same hyperparameters for CASMO and baselines
- Report multiple metrics (not just accuracy)
- Include statistical significance tests
- Provide full reproduction code

### 3. Computational Efficiency

Benchmarks are designed to run on modest hardware:
- B1, B5: CPU-friendly (~30 minutes)
- B2, B3: Single GPU (~1-2 hours)
- B4: Single GPU with 16GB VRAM (~4 hours)

## Citation

If you use these benchmarks in your research, please cite:

```bibtex
@software{casmo2025,
  title={CASMO: Confident Adaptive Selective Momentum Optimizer},
  author={Ainouche, Abderahmane},
  year={2025},
  url={https://github.com/abderahmane-ai/CASMO}
}
```

## Contributing

Want to add a new benchmark? See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

Ideal benchmarks:
- Demonstrate a clear failure mode of standard optimizers
- Are reproducible on modest hardware
- Include comprehensive evaluation metrics
- Provide theoretical motivation

## Questions?

- Check the [FAQ](../docs/faq.md)
- Open an [issue](https://github.com/abderahmane-ai/CASMO/issues)
- Read the [theory paper](../CASMO_THEORY.md)

---

**Made with ❤️ for the ML research community.**
