# Getting Started with CASMO

This guide will help you get up and running with CASMO in minutes.

## Installation

### Via PyPI (Recommended)

```bash
pip install casmo-optimizer
```

### From Source

```bash
git clone https://github.com/abderahmane-ai/CASMO.git
cd CASMO
pip install -e .
```

## Quick Start

### Basic Usage

CASMO is a drop-in replacement for Adam/AdamW:

```python
import torch
import torch.nn as nn
from casmo import CASMO

# Create your model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

# Replace AdamW with CASMO
optimizer = CASMO(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01
)

# Standard training loop
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

### Optimal Configuration

For best results, provide `total_steps` to enable automatic calibration:

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

## Key Parameters

### Essential Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | `1e-3` | Learning rate (same as Adam/AdamW) |
| `weight_decay` | `0.0` | L2 regularization coefficient |
| `granularity` | `'group'` | `'group'` (fast) or `'parameter'` (precise) |

### Advanced Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `betas` | `(0.9, 0.999)` | Momentum coefficients for gradient moments |
| `eps` | `1e-8` | Numerical stability term |
| `tau_init_steps` | Auto | Calibration period (auto-calculated if not provided) |
| `c_min` | `0.1` | Minimum confidence floor (prevents dead neurons) |
| `total_steps` | `None` | Total training steps (enables optimal calibration) |

## Understanding AGAR

CASMO's core innovation is **AGAR (Adaptive Gradient Alignment Ratio)**:

```
AGAR = ||E[g]||² / (||E[g]||² + Var[g])
```

- **High AGAR (≈1)**: Consistent gradients → Full learning rate
- **Low AGAR (≈0)**: Noisy gradients → Reduced learning rate

CASMO automatically adapts to gradient quality without manual tuning.

## When to Use CASMO

### ✅ Ideal Use Cases

CASMO excels in challenging scenarios:

1. **Noisy Labels** (>10% corruption)
   ```python
   # Training with corrupted labels
   optimizer = CASMO(model.parameters(), lr=1e-3)
   ```

2. **Imbalanced Datasets** (long-tail distributions)
   ```python
   # CIFAR-100 with 100:1 class imbalance
   optimizer = CASMO(model.parameters(), lr=0.1, weight_decay=5e-4)
   ```

3. **Continual Learning** (sequential tasks)
   ```python
   # Fine-tuning on multiple tasks sequentially
   optimizer = CASMO(model.parameters(), lr=2e-4, weight_decay=0.01)
   ```

4. **Differential Privacy** (high noise)
   ```python
   # DP-SGD with Opacus
   optimizer = CASMO(model.parameters(), lr=1e-3)
   model, optimizer, loader = privacy_engine.make_private(...)
   ```

5. **Unstable Training** (GANs, RL)
   ```python
   # GAN generator/discriminator
   gen_optimizer = CASMO(generator.parameters(), lr=2e-4)
   disc_optimizer = CASMO(discriminator.parameters(), lr=2e-4)
   ```

### ❌ When NOT to Use CASMO

Stick with AdamW for:

- **Clean, well-curated datasets** - AdamW is sufficient
- **Very short training** (<100 steps) - Not enough time for calibration
- **Sparse gradients** - Not currently supported

## Common Patterns

### With Learning Rate Scheduling

```python
optimizer = CASMO(model.parameters(), lr=0.1, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

for epoch in range(num_epochs):
    train_epoch(model, optimizer)
    scheduler.step()
```

### With Gradient Clipping

```python
optimizer = CASMO(model.parameters(), lr=1e-3)

for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
```

### With Mixed Precision (AMP)

```python
from torch.cuda.amp import autocast, GradScaler

optimizer = CASMO(model.parameters(), lr=1e-3)
scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        loss = model(batch)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Monitoring AGAR

You can access AGAR values during training:

```python
optimizer = CASMO(model.parameters(), lr=1e-3)

for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
    
    # Access AGAR for first parameter group
    group_state = optimizer._group_states[0]
    agar = group_state.get('current_agar')
    confidence = group_state.get('current_confidence')
    
    if agar is not None:
        print(f"AGAR: {agar:.4f}, Confidence: {confidence:.4f}")
```

## Troubleshooting

### Issue: Training is slower than AdamW

**Solution**: Use `granularity='group'` (default) instead of `'parameter'`:
```python
optimizer = CASMO(model.parameters(), lr=1e-3, granularity='group')
```

### Issue: Model not learning

**Possible causes**:
1. Learning rate too low - Try increasing `lr`
2. `c_min` too low - Try `c_min=0.3` or `c_min=0.5`
3. Not enough calibration - Provide `total_steps` parameter

### Issue: NaN or Inf gradients

CASMO automatically detects and raises errors for NaN/Inf gradients. Check:
1. Learning rate is not too high
2. Gradients are not exploding (use gradient clipping)
3. Input data is normalized

## Next Steps

- Read the [API Reference](api-reference.md) for detailed parameter descriptions
- Check out [Examples](../examples/) for complete working code
- Review [Benchmarks](../benchmarks/) to see CASMO in action
- Read the [Theory Paper](../CASMO_THEORY.md) for mathematical details

## Need Help?

- Check the [FAQ](faq.md)
- Open an [issue](https://github.com/abderahmane-ai/CASMO/issues)
- Read the [migration guide](migration-guide.md) if coming from Adam/AdamW
