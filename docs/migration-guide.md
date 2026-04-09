# Migration Guide: From Adam/AdamW to CASMO

This guide helps you migrate existing code from Adam or AdamW to CASMO.

## Quick Migration

### From AdamW

```python
# Before
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)

# After
from casmo import CASMO

optimizer = CASMO(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)
```

That's it! CASMO is a drop-in replacement.

## Optimal Migration

For best results, add `total_steps`:

```python
total_steps = len(train_loader) * num_epochs

optimizer = CASMO(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01,
    granularity='group',  # Recommended
    total_steps=total_steps
)
```

## Parameter Mapping

| AdamW Parameter | CASMO Parameter | Notes |
|-----------------|-----------------|-------|
| `lr` | `lr` | Same |
| `betas` | `betas` | Same |
| `eps` | `eps` | Same |
| `weight_decay` | `weight_decay` | Same |
| `amsgrad` | N/A | Not supported (use standard CASMO) |
| N/A | `granularity` | New: `'group'` or `'parameter'` |
| N/A | `c_min` | New: Minimum confidence (default: 0.1) |
| N/A | `total_steps` | New: For optimal calibration |

## Common Scenarios

### Image Classification (ResNet, ViT)

```python
# AdamW
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.1,
    weight_decay=5e-4
)

# CASMO
optimizer = CASMO(
    model.parameters(),
    lr=0.1,
    weight_decay=5e-4,
    total_steps=len(train_loader) * num_epochs
)
```

### Fine-tuning LLMs

```python
# AdamW
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4,
    weight_decay=0.01
)

# CASMO
optimizer = CASMO(
    model.parameters(),
    lr=2e-4,
    weight_decay=0.01,
    total_steps=len(train_loader) * num_epochs
)
```

### Training GANs

```python
# AdamW
gen_opt = torch.optim.AdamW(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
disc_opt = torch.optim.AdamW(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

# CASMO
gen_opt = CASMO(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
disc_opt = CASMO(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
```

## Learning Rate Scheduling

CASMO works with all PyTorch schedulers:

```python
optimizer = CASMO(model.parameters(), lr=0.1)

# Cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Step decay
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Reduce on plateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
```

## Checkpoint Compatibility

### Saving Checkpoints

```python
# Same as AdamW
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch
}
torch.save(checkpoint, 'checkpoint.pt')
```

### Loading Checkpoints

```python
# Load checkpoint
checkpoint = torch.load('checkpoint.pt')

# Create optimizer
optimizer = CASMO(model.parameters(), lr=1e-3)

# Load state
optimizer.load_state_dict(checkpoint['optimizer'])
```

### Converting AdamW → CASMO

If you have an AdamW checkpoint and want to continue with CASMO:

```python
# Load AdamW checkpoint
checkpoint = torch.load('adamw_checkpoint.pt')
model.load_state_dict(checkpoint['model'])

# Create new CASMO optimizer (starts fresh)
optimizer = CASMO(model.parameters(), lr=1e-3)

# Continue training
# Note: CASMO will recalibrate during first tau_init_steps
```

## Performance Expectations

### Speed

- CASMO is ~5% slower than AdamW with `granularity='group'`
- Negligible difference for most applications
- Overhead is constant per step (doesn't scale with model size)

### Memory

- Additional ~5% memory for AGAR statistics
- Minimal impact on large models

### Accuracy

CASMO typically matches or exceeds AdamW on:
- Noisy labels (10%+ improvement)
- Imbalanced data (5-15% improvement on tail classes)
- Continual learning (10-20% less forgetting)
- Clean data (similar performance)

## Troubleshooting

### Issue: Training is slower

**Solution**: Ensure you're using `granularity='group'`:
```python
optimizer = CASMO(model.parameters(), lr=1e-3, granularity='group')
```

### Issue: Different results than AdamW

**Expected**: CASMO adapts to gradient quality, so results may differ
- On clean data: Similar to AdamW
- On noisy data: Better than AdamW
- During calibration (first ~500 steps): May differ slightly

### Issue: Model not learning

**Check**:
1. Learning rate is appropriate
2. `c_min` is not too low (try 0.3 or 0.5)
3. Provide `total_steps` for better calibration

### Issue: NaN/Inf errors

CASMO detects NaN/Inf automatically. If you see errors:
1. Reduce learning rate
2. Add gradient clipping
3. Check input data normalization

## Best Practices

### 1. Always Provide total_steps

```python
# Good
total_steps = len(train_loader) * num_epochs
optimizer = CASMO(model.parameters(), lr=1e-3, total_steps=total_steps)

# Okay (but not optimal)
optimizer = CASMO(model.parameters(), lr=1e-3)
```

### 2. Use Group Granularity

```python
# Recommended
optimizer = CASMO(model.parameters(), lr=1e-3, granularity='group')

# Only if you need per-parameter control
optimizer = CASMO(model.parameters(), lr=1e-3, granularity='parameter')
```

### 3. Adjust c_min for Noise Level

```python
# High noise (30%+ label corruption)
optimizer = CASMO(model.parameters(), lr=1e-3, c_min=0.1)

# Moderate noise (10-30%)
optimizer = CASMO(model.parameters(), lr=1e-3, c_min=0.3)

# Low noise or clean data
optimizer = CASMO(model.parameters(), lr=1e-3, c_min=0.5)
```

### 4. Keep Same Hyperparameters

When migrating, keep your existing hyperparameters:
- Same learning rate
- Same weight decay
- Same betas
- Same batch size

CASMO will adapt automatically.

## When to Migrate

### ✅ Migrate if you have:

- Noisy labels
- Imbalanced datasets
- Continual learning tasks
- Unstable training (GANs, RL)
- Differential privacy requirements

### ⚠️ Consider staying with AdamW if:

- Clean, well-curated data
- Very short training runs (<100 steps)
- Sparse gradients (not supported)
- Extreme memory constraints

## Need Help?

- Check the [FAQ](faq.md)
- Review [examples](../examples/)
- Open an [issue](https://github.com/abderahmane-ai/CASMO/issues)
- Read the [API reference](api-reference.md)
