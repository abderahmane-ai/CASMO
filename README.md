# CASMO: Confident Adaptive Selective Momentum Optimizer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready PyTorch optimizer that extends Adam with **automatic noise robustness** using AGAR (Adaptive Gradient Alignment Ratio).

## TL;DR

CASMO automatically detects and adapts to gradient noise during training, learning from clean gradients while ignoring noisy ones. It's a drop-in replacement for Adam/AdamW that's faster on large models and more robust on noisy datasets.

```python
from casmo import CASMO

# Use exactly like Adam/AdamW
optimizer = CASMO(model.parameters(), lr=1e-3, weight_decay=0.01)

# Train normally
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## Key Results

| Benchmark | Metric | CASMO | AdamW | Improvement |
|-----------|--------|-------|-------|-------------|
| **Noisy Grokking** (30% label noise) | Val Accuracy | **90.3%** | 24.0% | **+66.3%** |
| **Long-tail CIFAR-100** (100:1) | Val Accuracy | **28.9%** | 23.9% | **+5.0%** |
| **Long-tail CIFAR-100** | Few-shot Acc | **5.9%** | 4.1% | **+44% (rel)** |
| **Performance** | Overhead vs AdamW | **-2%** | baseline | Faster! |

> [!NOTE]
> On the noisy grokking task, AdamW memorized corrupt labels (100% train, 24% val), while CASMO learned the true rule (70% train, 90% val).

## Features

- ✅ **Automatic noise robustness**: No manual hyperparameter tuning needed
- ✅ **Faster than AdamW**: ~2% faster on large models with group granularity
- ✅ **Drop-in replacement**: Compatible with existing PyTorch training code
- ✅ **Production-ready**: Comprehensive testing, checkpointing support, and logging
- ✅ **Well-documented**: Clear API docs and usage examples

## Installation

### From source (development)

```bash
git clone https://github.com/abderahmane-ai/CASMO.git
cd CASMO
pip install -e .
```

### With pip (once published)

```bash
pip install casmo-optimizer
```

## How It Works

### AGAR: Adaptive Gradient Alignment Ratio

CASMO uses a novel metric called AGAR to measure gradient signal-to-noise ratio:

```
AGAR = ||E[g]||² / (||E[g]||² + Var[g])
```

- **High AGAR** (→ 1.0): Consistent gradient direction (clean signal) → Normal learning rate
- **Low AGAR** (→ 0.0): Random gradient fluctuations (noise) → Reduced learning rate

AGAR naturally ranges from 0 to 1, making it interpretable and robust across different tasks.

### Automatic Confidence Scaling

After an initial calibration phase (default: 500 steps), CASMO automatically computes a confidence score for each gradient update:

```
confidence = c_min + (1 - c_min) × sigmoid((AGAR - μ) / σ)
effective_lr = base_lr × confidence
```

This sigmoid mapping adapts to your dataset's noise characteristics without manual tuning.

## Usage Examples

### Basic Usage

```python
from casmo import CASMO
import torch.nn as nn

model = YourModel()
optimizer = CASMO(model.parameters(), lr=1e-3, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch), labels)
        loss.backward()
        optimizer.step()
```

### Advanced Configuration

```python
optimizer = CASMO(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01,
    betas=(0.9, 0.999),           # Adam momentum coefficients
    granularity='group',           # 'group' (fast) or 'parameter' (precise)
    tau_init_steps=500,            # Calibration steps (auto-tuned)
    c_min=0.1,                     # Minimum confidence (auto-adjusted)
    log_level=2,                   # 0=silent, 1=errors, 2=warnings, 3=info
)
```

### With Learning Rate Scheduling

```python
optimizer = CASMO(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

for epoch in range(epochs):
    train_one_epoch(model, optimizer)
    scheduler.step()
```

### Checkpointing

```python
# Save checkpoint
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
# Training continues from saved state
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | `1e-3` | Learning rate |
| `betas` | `(0.9, 0.999)` | Adam momentum coefficients |
| `eps` | `1e-8` | Numerical stability constant |
| `weight_decay` | `0.0` | Decoupled weight decay (AdamW-style) |
| `granularity` | `'group'` | AGAR computation: `'group'` (fast) or `'parameter'` (precise) |
| `tau_init_steps` | `500` | Calibration steps (set to ~5% of total steps) |
| `c_min` | `0.1` | Minimum confidence (auto-adjusted after calibration) |
| `log_level` | `1` | Logging verbosity: 0 (silent), 1 (errors), 2 (warnings), 3 (info) |

## When to Use CASMO

**CASMO excels when:**
- ✅ Training on datasets with label noise or annotation errors
- ✅ Working with imbalanced or long-tail distributions
- ✅ Fine-tuning on small, potentially noisy datasets
- ✅ Dealing with "grokking" phenomena (memorization vs generalization)
- ✅ Training large models where optimizer overhead matters

**Stick with Adam/AdamW when:**
- Standard clean datasets with well-curated labels
- You need the absolute minimum computational overhead
- Your training is already optimized and working perfectly

## Benchmarks

See the [`benchmarks/`](benchmarks/) directory for full reproducible experiments:

- **B1: Noisy CIFAR-10** - Label noise robustness
- **B2: Noisy Alpaca SFT** - Language model fine-tuning
- **B3: Grokking** - Modular arithmetic with 30% label noise
- **B4: Long-tail CIFAR-100** - Class imbalance (100:1 ratio)

Each benchmark includes training scripts, configuration, and results visualization.

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_agar_computation.py -v

# Run with coverage
pytest tests/ --cov=casmo --cov-report=html
```

### Code Quality

```bash
# Format code
black casmo.py tests/

# Type checking
mypy casmo.py

# Linting
flake8 casmo.py
```

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

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built on top of PyTorch's optimizer framework
- Inspired by Adam (Kingma & Ba, 2015) and AdamW (Loshchilov & Hutter, 2019)
- AGAR metric design motivated by signal processing and information theory

## Contact

- GitHub Issues: [Report bugs or request features](https://github.com/abderahmane-ai/CASMO/issues)
- Email: abderahmane.ainouche.ai@gmail.com

---

**Made with ❤️ for the ML research community**
