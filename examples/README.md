# CASMO Examples

This directory contains practical examples demonstrating how to use CASMO in various scenarios.

## Quick Start

### Basic Usage
The simplest way to get started with CASMO:
```bash
python basic_usage.py
```

### Image Classification
Train a ResNet on CIFAR-10 with CASMO:
```bash
python image_classification.py
```

### Fine-tuning LLMs
Fine-tune a language model with CASMO:
```bash
pip install transformers
python fine_tuning_llm.py
```

### Comparing Optimizers
See the difference between CASMO and AdamW on noisy data:
```bash
pip install matplotlib
python comparing_optimizers.py
```

## Examples Overview

| Example | Description | Key Features |
|---------|-------------|--------------|
| `basic_usage.py` | Minimal working example | Drop-in replacement for AdamW |
| `image_classification.py` | CIFAR-10 with ResNet | Real CV task, LR scheduling |
| `fine_tuning_llm.py` | Language model fine-tuning | Transformers integration |
| `comparing_optimizers.py` | CASMO vs AdamW comparison | Noisy labels, visualization |

## Key Concepts

### 1. Drop-in Replacement
CASMO can replace AdamW with minimal code changes:

```python
# Before (AdamW)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# After (CASMO)
from casmo import CASMO
optimizer = CASMO(model.parameters(), lr=1e-3, weight_decay=0.01)
```

### 2. Optimal Configuration
For best results, provide `total_steps` for automatic calibration:

```python
total_steps = len(train_loader) * num_epochs
optimizer = CASMO(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01,
    granularity='group',  # Recommended for efficiency
    total_steps=total_steps
)
```

### 3. When to Use CASMO
CASMO excels in challenging scenarios:
- **Noisy labels** (>10% corruption)
- **Imbalanced datasets** (long-tail distributions)
- **Continual learning** (sequential tasks)
- **Differential privacy** (high noise)
- **Unstable training** (GANs, RL)

### 4. When NOT to Use CASMO
Stick with AdamW for:
- Clean, well-curated datasets
- Very short training runs (<100 steps)
- Sparse gradients (not currently supported)

## Additional Resources

- [Full Documentation](../docs/)
- [Benchmarks](../benchmarks/)
- [API Reference](../docs/api-reference.md)
- [Theory Paper](../CASMO_THEORY.md)

## Need Help?

- Check the [FAQ](../docs/faq.md)
- Open an [issue](https://github.com/abderahmane-ai/CASMO/issues)
- Read the [migration guide](../docs/migration-guide.md)
