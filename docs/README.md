# CASMO Documentation

Welcome to the CASMO documentation! This directory contains comprehensive guides for using CASMO effectively.

## Quick Links

- **[Getting Started](getting-started.md)** - Installation and first steps
- **[API Reference](api-reference.md)** - Complete parameter documentation
- **[FAQ](faq.md)** - Frequently asked questions
- **[Migration Guide](migration-guide.md)** - Moving from Adam/AdamW to CASMO

## Documentation Structure

### For New Users

1. Start with [Getting Started](getting-started.md)
2. Try the [examples](../examples/)
3. Read the [FAQ](faq.md) for common questions

### For Existing Users

1. Check the [API Reference](api-reference.md) for parameter details
2. Use the [Migration Guide](migration-guide.md) to convert existing code
3. Review [benchmarks](../benchmarks/) for advanced usage

### For Researchers

1. Read the [Theory Paper](../CASMO_THEORY.md) for mathematical details
2. Explore [benchmarks](../benchmarks/) for experimental validation
3. Check [CHANGELOG](../CHANGELOG.md) for recent improvements

## Key Concepts

### AGAR (Adaptive Gradient Alignment Ratio)

The core innovation in CASMO:

```
AGAR = ||E[g]||² / (||E[g]||² + Var[g])
```

- Measures gradient signal-to-noise ratio
- Range: 0 (pure noise) to 1 (pure signal)
- Automatically computed during training

### Confidence Scaling

CASMO scales learning rate based on AGAR:

- **High AGAR** → High confidence → Full learning rate
- **Low AGAR** → Low confidence → Reduced learning rate

This happens automatically without manual tuning.

### Calibration

During the first ~500 steps (configurable), CASMO:
1. Collects AGAR statistics
2. Learns gradient distribution
3. Sets adaptive thresholds
4. Determines optimal confidence scaling

After calibration, parameters are frozen.

## Common Use Cases

### Noisy Labels

```python
optimizer = CASMO(
    model.parameters(),
    lr=1e-3,
    c_min=0.1  # Aggressive noise suppression
)
```

See: [B1 Grokking Benchmark](../benchmarks/b1_grokking/)

### Imbalanced Data

```python
optimizer = CASMO(
    model.parameters(),
    lr=0.1,
    weight_decay=5e-4,
    total_steps=len(train_loader) * num_epochs
)
```

See: [B2 Long-Tail Benchmark](../benchmarks/b2_long_tail_cifar100/)

### Continual Learning

```python
optimizer = CASMO(
    model.parameters(),
    lr=2e-4,
    weight_decay=0.01,
    total_steps=total_steps
)
```

See: [B4 Continual Learning Benchmark](../benchmarks/b4_continual_learning/)

### Differential Privacy

```python
optimizer = CASMO(model.parameters(), lr=1e-3)
model, optimizer, loader = privacy_engine.make_private(...)
```

See: [B3 DP-SGD Benchmark](../benchmarks/b3_dp_sgd/)

## Best Practices

### ✅ Do

1. Provide `total_steps` for optimal calibration
2. Use `granularity='group'` for efficiency
3. Keep same hyperparameters as AdamW
4. Monitor AGAR during training (optional)
5. Adjust `c_min` based on noise level

### ❌ Don't

1. Use for very short training (<100 steps)
2. Use with sparse gradients (not supported)
3. Manually set `tau_init_steps` (let CASMO auto-calculate)
4. Use `granularity='parameter'` unless necessary
5. Expect improvements on clean, well-curated data

## Performance

### Speed

- **Group granularity**: ~5% slower than AdamW
- **Parameter granularity**: ~15% slower than AdamW
- Overhead is constant per step

### Memory

- Additional ~5% memory for AGAR statistics
- Minimal impact on large models

### Accuracy

- **Noisy data**: 10-30% improvement
- **Imbalanced data**: 5-15% better on tail classes
- **Continual learning**: 10-20% less forgetting
- **Clean data**: Similar to AdamW

## Troubleshooting

### Model not learning

1. Check learning rate
2. Try higher `c_min` (0.3 or 0.5)
3. Provide `total_steps`

### Training slower than expected

1. Use `granularity='group'`
2. Check for other bottlenecks

### NaN/Inf errors

1. Reduce learning rate
2. Add gradient clipping
3. Check input normalization

See [FAQ](faq.md) for more troubleshooting tips.

## Contributing

Want to improve the documentation?

1. Fork the repository
2. Make your changes
3. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## Additional Resources

- **[Examples](../examples/)** - Working code examples
- **[Benchmarks](../benchmarks/)** - Comprehensive evaluations
- **[Theory Paper](../CASMO_THEORY.md)** - Mathematical foundations
- **[GitHub Issues](https://github.com/abderahmane-ai/CASMO/issues)** - Bug reports and questions

## Need Help?

- Check the [FAQ](faq.md)
- Review [examples](../examples/)
- Open an [issue](https://github.com/abderahmane-ai/CASMO/issues)
- Read the [migration guide](migration-guide.md)

---

**Happy optimizing! 🚀**
