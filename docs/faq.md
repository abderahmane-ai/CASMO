# Frequently Asked Questions (FAQ)

## General Questions

### What is CASMO?

CASMO (Confident Adaptive Selective Momentum Optimizer) is a PyTorch optimizer that extends Adam with automatic gradient quality detection. It uses AGAR (Adaptive Gradient Alignment Ratio) to measure gradient consistency and adaptively scales learning rates based on signal-to-noise ratio.

### How is CASMO different from Adam/AdamW?

- **Adam/AdamW**: Treats all gradients equally
- **CASMO**: Adapts learning rate based on gradient quality
  - High-quality gradients → Full learning rate
  - Noisy gradients → Reduced learning rate

### Is CASMO a drop-in replacement for AdamW?

Yes! CASMO uses the same interface as PyTorch optimizers:

```python
# Replace this
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# With this
optimizer = CASMO(model.parameters(), lr=1e-3)
```

## Performance Questions

### Is CASMO slower than AdamW?

Slightly (~5% with `granularity='group'`), but the overhead is constant and doesn't scale with model size. For most applications, the difference is negligible.

### Does CASMO use more memory?

Yes, about 5% more memory to store AGAR statistics. This is minimal compared to model parameters and activations.

### When does CASMO outperform AdamW?

CASMO excels in:
- **Noisy labels** (10-30%+ improvement)
- **Imbalanced data** (5-15% better on tail classes)
- **Continual learning** (10-20% less forgetting)
- **Differential privacy** (maintains accuracy under noise)
- **Unstable training** (GANs, RL)

On clean, well-curated data, CASMO performs similarly to AdamW.

## Usage Questions

### What learning rate should I use?

Use the same learning rate you would use with AdamW:
- **Training from scratch**: 1e-3 to 0.1
- **Fine-tuning**: 1e-5 to 5e-4
- **LLMs**: 1e-5 to 2e-4

CASMO adapts automatically, so you don't need to tune the LR differently.

### Should I use 'group' or 'parameter' granularity?

**Use `'group'` (default)** for most cases:
- Faster (~5% overhead vs ~15%)
- Sufficient for most tasks
- Recommended for production

**Use `'parameter'`** only if:
- You need fine-grained control
- You're debugging gradient issues
- Speed is not a concern

### What is tau_init_steps and how should I set it?

`tau_init_steps` is the calibration period where CASMO learns the gradient distribution.

**Recommendation**: Don't set it manually. Instead, provide `total_steps`:

```python
total_steps = len(train_loader) * num_epochs
optimizer = CASMO(model.parameters(), lr=1e-3, total_steps=total_steps)
```

CASMO will automatically calculate optimal `tau_init_steps`.

### What is c_min and how should I set it?

`c_min` is the minimum confidence floor (range: 0-1).

**Guidelines**:
- `c_min=0.1`: High noise (30%+ label corruption) - aggressive suppression
- `c_min=0.3`: Moderate noise (10-30%) - balanced
- `c_min=0.5`: Low noise or clean data - conservative

**Default**: 0.1 (works well for most noisy scenarios)

### Can I use CASMO with learning rate schedulers?

Yes! CASMO works with all PyTorch schedulers:

```python
optimizer = CASMO(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

for epoch in range(num_epochs):
    train(model, optimizer)
    scheduler.step()
```

### Can I use CASMO with gradient clipping?

Yes:

```python
optimizer = CASMO(model.parameters(), lr=1e-3)

for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

### Can I use CASMO with mixed precision (AMP)?

Yes:

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

## Technical Questions

### How does AGAR work?

AGAR measures gradient consistency:

```
AGAR = ||E[g]||² / (||E[g]||² + Var[g])
```

- **Numerator**: Signal power (consistent direction)
- **Denominator**: Total power (signal + noise)
- **Range**: 0 (pure noise) to 1 (pure signal)

### What happens during calibration?

During the first `tau_init_steps` steps, CASMO:
1. Collects AGAR samples
2. Computes distribution statistics (mean, std, median)
3. Sets adaptive thresholds
4. Determines optimal `c_min` based on noise level

After calibration, these parameters are frozen.

### Can I monitor AGAR during training?

Yes:

```python
optimizer = CASMO(model.parameters(), lr=1e-3)

for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
    
    # Access AGAR
    group_state = optimizer._group_states[0]
    agar = group_state.get('current_agar')
    confidence = group_state.get('current_confidence')
    
    if agar is not None:
        print(f"AGAR: {agar:.4f}, Confidence: {confidence:.4f}")
```

### Does CASMO support sparse gradients?

No, sparse gradients are not currently supported. CASMO will raise a `NotImplementedError` if it encounters sparse gradients.

### Can I use CASMO with distributed training?

Yes, CASMO works with PyTorch's distributed training:

```python
model = torch.nn.parallel.DistributedDataParallel(model)
optimizer = CASMO(model.parameters(), lr=1e-3)
```

### Can I use different hyperparameters for different layers?

Yes, use parameter groups:

```python
optimizer = CASMO([
    {'params': model.encoder.parameters(), 'lr': 1e-4},
    {'params': model.decoder.parameters(), 'lr': 1e-3}
], weight_decay=0.01)
```

## Troubleshooting

### My model is not learning

**Check**:
1. Learning rate is appropriate (not too low)
2. `c_min` is not too low (try 0.3 or 0.5)
3. Provide `total_steps` for better calibration
4. Verify gradients are flowing (check with `torch.autograd.grad`)

### I'm getting NaN/Inf errors

CASMO automatically detects NaN/Inf. If you see errors:
1. Reduce learning rate
2. Add gradient clipping
3. Check input normalization
4. Verify model architecture (no division by zero, etc.)

### Training is slower than expected

**Solutions**:
1. Use `granularity='group'` (default)
2. Ensure you're using GPU if available
3. Check if other bottlenecks exist (data loading, etc.)

### Results differ from AdamW

**Expected behavior**:
- On clean data: Similar to AdamW
- On noisy data: Better than AdamW (this is the goal!)
- During calibration: May differ slightly

If results are worse:
1. Try adjusting `c_min` (0.3 or 0.5)
2. Ensure `total_steps` is provided
3. Check if your data is actually noisy

### Checkpoint loading fails

If loading an AdamW checkpoint into CASMO:
```python
# Load model weights
model.load_state_dict(checkpoint['model'])

# Create new CASMO optimizer (don't load optimizer state)
optimizer = CASMO(model.parameters(), lr=1e-3)
```

CASMO will recalibrate automatically.

## Comparison Questions

### CASMO vs Adam

- **Adam**: No weight decay, older algorithm
- **CASMO**: Based on AdamW (decoupled weight decay) + AGAR

Use CASMO instead of Adam.

### CASMO vs AdamW

- **AdamW**: Standard, well-tested, fast
- **CASMO**: Noise-robust, adaptive, slightly slower

Use CASMO when you have noisy/challenging data.

### CASMO vs SGD

- **SGD**: Simple, requires careful tuning, no adaptive LR
- **CASMO**: Adaptive, automatic noise handling, easier to use

CASMO is generally better for most tasks.

### CASMO vs other noise-robust methods

- **Data cleaning**: Requires manual effort, may remove useful data
- **Robust loss functions**: Task-specific, requires tuning
- **CASMO**: Automatic, optimizer-level, works with any loss

CASMO is complementary to other methods.

## Best Practices

### For best results:

1. ✅ Provide `total_steps`
2. ✅ Use `granularity='group'`
3. ✅ Keep same hyperparameters as AdamW
4. ✅ Adjust `c_min` based on noise level
5. ✅ Monitor AGAR during training (optional)

### Avoid:

1. ❌ Very short training runs (<100 steps)
2. ❌ Sparse gradients (not supported)
3. ❌ Manually setting `tau_init_steps` (let CASMO auto-calculate)
4. ❌ Using `granularity='parameter'` unless necessary

## Still Have Questions?

- Check the [Getting Started Guide](getting-started.md)
- Review [Examples](../examples/)
- Read the [API Reference](api-reference.md)
- Open an [issue](https://github.com/abderahmane-ai/CASMO/issues)
