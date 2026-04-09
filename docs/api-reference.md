# API Reference

Complete reference for the CASMO optimizer.

## CASMO Class

```python
class CASMO(torch.optim.Optimizer)
```

Confident Adaptive Selective Momentum Optimizer.

Extends Adam with confidence-based learning rate scaling using AGAR (Adaptive Gradient Alignment Ratio) metrics. Automatically adapts to gradient signal-to-noise ratio for improved convergence in noisy environments.

### Constructor

```python
CASMO(
    params,
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
    tau_init_steps=None,
    tau_clip_range=(0.01, 0.5),
    c_min=0.1,
    granularity='group',
    agar_clamp_factor=10.0,
    total_steps=None
)
```

### Parameters

#### Required Parameters

**`params`** (iterable)
- Iterable of parameters to optimize or dicts defining parameter groups
- Same as PyTorch's standard optimizers

#### Learning Parameters

**`lr`** (float, default: `1e-3`)
- Learning rate
- Typical range: `1e-5` to `0.1`
- Use lower values for fine-tuning, higher for training from scratch

**`betas`** (Tuple[float, float], default: `(0.9, 0.999)`)
- Coefficients for computing running averages of gradient and its square
- `beta1`: First moment decay rate (momentum)
- `beta2`: Second moment decay rate (variance)
- Typical range: `beta1` in `[0.8, 0.95]`, `beta2` in `[0.99, 0.9999]`

**`eps`** (float, default: `1e-8`)
- Term added to denominator for numerical stability
- Rarely needs adjustment

**`weight_decay`** (float, default: `0.0`)
- Decoupled weight decay coefficient (L2 regularization)
- Typical range: `0.0` to `0.1`
- Use `0.01` for most tasks, `0.0` for tasks without regularization

#### AGAR Configuration

**`tau_init_steps`** (int, optional, default: `None`)
- Number of steps for initial AGAR calibration
- If `None`, automatically calculated as: `max(500, int(50 / (1 - beta1)))`
- If `total_steps` is provided, capped at `total_steps // 5`
- Minimum value: `50`
- **Recommendation**: Let CASMO auto-calculate by providing `total_steps`

**`tau_clip_range`** (Tuple[float, float], default: `(0.01, 0.5)`)
- Min/max bounds for tau (AGAR threshold)
- Prevents extreme threshold values
- Rarely needs adjustment

**`c_min`** (float, default: `0.1`)
- Minimum confidence scaling factor
- Range: `[0.0, 1.0]`
- Lower values = more aggressive noise suppression
- Higher values = more conservative (closer to Adam)
- **Recommendations**:
  - `0.1`: High noise scenarios (30%+ label noise)
  - `0.3`: Moderate noise
  - `0.5`: Low noise or when unsure

**`granularity`** (str, default: `'group'`)
- AGAR computation granularity
- Options:
  - `'group'`: Single AGAR per parameter group (faster, recommended)
  - `'parameter'`: Per-parameter AGAR (more precise, slower)
- **Recommendation**: Use `'group'` unless you need fine-grained control

**`agar_clamp_factor`** (float, optional, default: `10.0`)
- Outlier clamping factor for AGAR computation
- Clamps gradients to `mean ± factor * std`
- Set to `None` to disable clamping
- Helps handle gradient spikes

**`total_steps`** (int, optional, default: `None`)
- Total number of training steps
- Enables optimal automatic calibration
- **Highly recommended** to provide this parameter
- Calculate as: `len(train_loader) * num_epochs`

### Methods

#### `step(closure=None)`

Performs a single optimization step.

**Parameters:**
- `closure` (callable, optional): A closure that reevaluates the model and returns the loss

**Returns:**
- Loss value if `closure` is provided, otherwise `None`

**Example:**
```python
optimizer.zero_grad()
loss = model(batch)
loss.backward()
optimizer.step()
```

#### `zero_grad(set_to_none=False)`

Sets gradients of all optimized parameters to zero.

**Parameters:**
- `set_to_none` (bool, default: `False`): If `True`, sets gradients to `None` instead of zero (more memory efficient)

**Example:**
```python
optimizer.zero_grad()
# or
optimizer.zero_grad(set_to_none=True)
```

#### `state_dict()`

Returns the state of the optimizer as a dict.

**Returns:**
- Dictionary containing optimizer state

**Example:**
```python
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch
}
torch.save(checkpoint, 'checkpoint.pt')
```

#### `load_state_dict(state_dict)`

Loads the optimizer state.

**Parameters:**
- `state_dict` (dict): Optimizer state (returned by `state_dict()`)

**Example:**
```python
checkpoint = torch.load('checkpoint.pt')
optimizer.load_state_dict(checkpoint['optimizer'])
```

### Attributes

#### `param_groups`

List of parameter groups. Each group is a dict containing:
- `params`: List of parameters
- `lr`: Learning rate for this group
- `weight_decay`: Weight decay for this group
- All other hyperparameters

**Example:**
```python
# Different learning rates for different layers
optimizer = CASMO([
    {'params': model.base.parameters(), 'lr': 1e-4},
    {'params': model.head.parameters(), 'lr': 1e-3}
])
```

#### `_group_states`

Internal dictionary containing per-group AGAR statistics:
- `current_agar`: Current AGAR value
- `current_confidence`: Current confidence score
- `tau`: Calibrated threshold
- `agar_mean`, `agar_std`: Distribution statistics

**Example:**
```python
# Access AGAR for monitoring
group_state = optimizer._group_states[0]
agar = group_state.get('current_agar')
confidence = group_state.get('current_confidence')
```

## Usage Examples

### Basic Usage

```python
from casmo import CASMO

optimizer = CASMO(model.parameters(), lr=1e-3, weight_decay=0.01)

for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

### With Total Steps (Recommended)

```python
total_steps = len(train_loader) * num_epochs

optimizer = CASMO(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01,
    total_steps=total_steps
)
```

### Multiple Parameter Groups

```python
optimizer = CASMO([
    {'params': model.encoder.parameters(), 'lr': 1e-4},
    {'params': model.decoder.parameters(), 'lr': 1e-3},
    {'params': model.head.parameters(), 'lr': 5e-3}
], weight_decay=0.01)
```

### High Noise Scenario

```python
optimizer = CASMO(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01,
    c_min=0.1,  # Aggressive noise suppression
    granularity='group'
)
```

### Fine-tuning LLMs

```python
optimizer = CASMO(
    model.parameters(),
    lr=2e-4,  # Lower LR for fine-tuning
    weight_decay=0.01,
    betas=(0.9, 0.999),
    total_steps=len(train_loader) * num_epochs
)
```

## Exceptions

### `ValueError`

Raised when invalid parameters are provided:

- `lr < 0`: Invalid learning rate
- `eps < 0`: Invalid epsilon
- `betas[0]` or `betas[1]` not in `[0, 1)`: Invalid beta values
- `weight_decay < 0`: Invalid weight decay
- `c_min` not in `[0, 1]`: Invalid c_min
- `granularity` not in `['parameter', 'group']`: Invalid granularity
- `tau_init_steps < 50`: Too few calibration steps

### `RuntimeError`

Raised during optimization:

- NaN or Inf gradients detected
- Sparse gradients encountered (not supported)

## Performance Considerations

### Speed

- **Group granularity**: ~5% slower than AdamW
- **Parameter granularity**: ~15% slower than AdamW

### Memory

- Additional memory: ~5% over AdamW
- Stores AGAR statistics per group/parameter

### Recommendations

1. Use `granularity='group'` for best speed/accuracy tradeoff
2. Provide `total_steps` for optimal calibration
3. Use `c_min=0.1` for high noise, `0.3` for moderate noise
4. Monitor AGAR values during training for insights

## See Also

- [Getting Started Guide](getting-started.md)
- [Migration Guide](migration-guide.md)
- [Theory Paper](../CASMO_THEORY.md)
- [Examples](../examples/)
