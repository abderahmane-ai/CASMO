# API Reference

## `casmo.CASMO`

```python
CASMO(
    params,
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
    c_min=0.1,
    robustness=0.5,
    rel_floor=0.1,
    nan_guard=False,
)
```

A drop-in replacement for `torch.optim.AdamW` that scales every update by a
per-coordinate confidence derived from the gradient signal-to-noise ratio.

Subclasses `torch.optim.Optimizer`, so it works with LR schedulers, parameter groups,
`state_dict()` / `load_state_dict()`, and closures exactly like any built-in optimizer.

---

### Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `params` | iterable | — | Parameters or parameter-group dicts. |
| `lr` | float | `1e-3` | Learning rate. Must be `>= 0`. |
| `betas` | (float, float) | `(0.9, 0.999)` | EMA coefficients for the first moment `m` and the belief variance `s`. Each must be in `[0, 1)`. |
| `eps` | float | `1e-8` | Numerical stability term. Must be `>= 0`. |
| `weight_decay` | float | `0.0` | Decoupled (AdamW-style) weight decay. Must be `>= 0`. |
| `c_min` | float | `0.1` | Floor of the absolute **trust** factor. Must be in `[0, 1]`. Prevents the effective LR collapsing under sustained noise. |
| `robustness` | float | `0.5` | Exponent on the trust factor. Must be `>= 0`. `0` disables absolute suppression (AdamW-like pace); `1` is maximally noise robust. |
| `rel_floor` | float | `0.1` | Floor of the relative **focus** factor. Must be in `[0, 1]`. Bounds how far a single coordinate can be down-weighted. `1.0` disables focus. |
| `nan_guard` | bool | `False` | Raise `RuntimeError` on non-finite gradients. Off by default because the check forces a host-device sync every step. |

All parameters except `params` can be set **per parameter group**.

### Raises

| Exception | When |
|---|---|
| `ValueError` | A hyper-parameter is outside its valid range, or `params` is empty. |
| `TypeError` | An unrecognised keyword argument is passed. |
| `NotImplementedError` | A sparse gradient is encountered. Use `torch.optim.SparseAdam`. |
| `RuntimeError` | `nan_guard=True` and a NaN/Inf gradient is seen. |
| `DeprecationWarning` | A removed v0.3 calibration kwarg is passed (see below). |

---

### Methods

#### `step(closure=None)`

Performs a single optimization step. Returns the closure's loss if one is given.

#### `group_metrics(group_idx=0)`

Returns the most recent optimizer view of gradient quality for a parameter group:

```python
{"agar": float | None, "confidence": float | None}
```

- `agar` — mean per-coordinate signal fraction across the group, in `[0, 1]`.
- `confidence` — mean applied confidence multiplier `C_i`, in `[0, 1]`.

Both are `None` before the first `step()`. Values are recomputed **every** step.

```python
for batch in loader:
    optimizer.zero_grad()
    loss_fn(model(batch)).backward()
    optimizer.step()
    m = optimizer.group_metrics(0)
    logger.log({"agar": m["agar"], "confidence": m["confidence"]})
```

#### `state_dict()` / `load_state_dict(state_dict)`

Standard PyTorch semantics. Per-parameter state is `{"step", "m", "s"}` — two EMAs, the
same footprint as Adam. Resuming from a checkpoint reproduces uninterrupted training
exactly.

---

### The `robustness` dial

`robustness` (ρ) is the exponent on the absolute trust factor:

```
C_i = trust**robustness * focus_i
```

| Value | Behaviour | Use when |
|---|---|---|
| `0.0` | Pure relative reweighting; AdamW-like pace | Clean data, DP-SGD, maximum speed |
| `0.5` *(default)* | Balanced | General purpose; best at aggressive LR |
| `1.0` | Full absolute suppression | Label noise, long-tail, task conflict |

Measured trade-off (5 seeds, see [`research/REDESIGN.md`](../research/REDESIGN.md)):

| ρ | clean steps | label-noise 30% test | grad-noise σ=0.5 test |
|---|---|---|---|
| Adam | 82 | 0.675 | 0.641 |
| 0.0 | 90 | 0.675 | 0.623 |
| 0.5 | 160 | 0.701 | 0.488 |
| 1.0 | 322 | **0.797** | 0.396 |

Higher ρ trades raw training-loss speed for resistance to noise. It does **not** cost
clean-data generalization (test accuracy is flat at 0.931–0.934 across ρ).

---

### Reducing exactly to AdamW

```python
CASMO(model.parameters(), lr=1e-3, robustness=0.0, rel_floor=1.0)
```

With `robustness=0`, `trust**0 == 1`; with `rel_floor=1.0`, `focus == 1`. So `C_i == 1`
and the update rule is AdamW's.

---

### Deprecated parameters (removed in v0.4)

These were part of the v0.3 calibration design. They are accepted with a
`DeprecationWarning` and **ignored**:

| Removed | Why |
|---|---|
| `tau_init_steps` | No calibration phase exists; AGAR is on an absolute scale. |
| `tau_clip_range` | No threshold to clip. |
| `granularity` | The gate is always per-coordinate. |
| `agar_clamp_factor` | AGAR is bounded in `[0, 1]` by construction. |
| `total_steps` | Only ever used to size the calibration phase. |

Passing an unrecognised kwarg raises `TypeError`.
