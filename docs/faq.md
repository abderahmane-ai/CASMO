# FAQ

## Concepts

### What does CASMO actually do differently from Adam?

Adam adapts to gradient **magnitude** — it divides by `sqrt(E[g²])`. But a large `E[g²]`
can come from a strong consistent signal *or* from large random fluctuations, and Adam
cannot tell which. CASMO estimates the **variance** of the gradient separately and forms
an explicit signal fraction, `AGAR = E[g]² / (E[g]² + Var[g])`, per coordinate. It then
scales each coordinate's step by a confidence derived from that fraction.

### What is AGAR?

The Adaptive Gradient Alignment Ratio: the fraction of a coordinate's gradient power that
is signal rather than noise. It is bounded in `[0, 1]` by construction — 1 means perfectly
consistent gradients, 0 means directionless noise.

### What are `trust` and `focus`?

The two axes of the confidence map:

- **trust** (absolute) = `c_min + (1-c_min)·mean(AGAR)` — one number per tensor. When the
  gradients are mostly noise, this shrinks and the whole tensor slows down. This is what
  resists memorizing label noise.
- **focus** (relative) = `clip(AGAR_i / mean(AGAR), rel_floor, 1)` — per coordinate,
  mean-normalized. It shifts the step toward reliable coordinates without changing the
  overall pace. This is what preserves speed and stability.

They are combined as `C_i = trust**robustness * focus_i`.

### Why is `robustness` a knob instead of automatic?

Because measurement showed that different noise types want **opposite** policies. Label
noise rewards slowing down globally; isotropic gradient noise (DP-SGD) punishes it under a
fixed step budget. No single value is right for both, so CASMO exposes the choice rather
than guessing. See [REDESIGN.md §6](../research/REDESIGN.md).

---

## Usage

### What learning rate should I use?

The one you already use for AdamW. CASMO tolerates aggressive LRs better than Adam
(measured 36 vs. 70 steps to converge at `lr=3e-2`), so if instability currently caps your
LR, try raising it.

### My training loss decreases more slowly than with Adam. Is it broken?

Almost certainly not — that is the mechanism working. At `robustness=1.0`, CASMO
deliberately refuses to drive the training loss to zero when the data contains noise. On
30% label noise it reaches 0.850 train accuracy where AdamW reaches 1.000, and its
**test** accuracy is 12 points higher as a result.

Judge CASMO on validation metrics, not training loss. If validation is also worse on
genuinely clean data, lower `robustness` toward `0`.

### How do I make CASMO behave exactly like AdamW?

```python
CASMO(model.parameters(), lr=1e-3, robustness=0.0, rel_floor=1.0)
```

`trust**0 == 1` and `focus == 1`, so `C_i == 1`.

### Does it work with LR schedulers / AMP / gradient accumulation / DDP?

Yes to schedulers, gradient accumulation, and parameter groups — CASMO is an ordinary
`torch.optim.Optimizer` subclass. AMP works through `GradScaler` normally (the scaler
checks for inf/nan and skips the step before CASMO sees it). DDP works; CASMO's statistics
are computed per-rank from the already-reduced gradients.

### Can I use it with sparse gradients?

No — `NotImplementedError`. Use `torch.optim.SparseAdam`, or densify with
`grad.to_dense()`.

### How do I see what the optimizer thinks of my gradients?

```python
m = optimizer.group_metrics(0)
print(m["agar"], m["confidence"])
```

---

## Performance

### What does it cost?

- **Memory:** the same as Adam — verified identical. Two EMAs per parameter (`m`, `s`). The
  Adam denominator `sqrt(E[g²])` is reconstructed as `sqrt(m̂² + ŝ)` rather than tracked
  separately.
- **Time:** ~1.9× Adam per step (measured ratio 1.88× on a 512→1024→1024→128 MLP; absolute
  timings vary by machine, the ratio is stable). That is the intrinsic cost of
  per-coordinate variance tracking plus the elementwise gate. It is ~14% faster than
  CASMO v0.3, which additionally paid for per-parameter host syncs and numpy calibration.

Note that wall-clock per step is rarely the binding constraint: the optimizer step is
usually small next to forward/backward. Measure end-to-end before worrying about it.

### Why not check for NaN gradients by default?

The check (`isnan().any()`) forces a host-device synchronization every step, for every
parameter. v0.3 did this and paid for it. Set `nan_guard=True` if you want it.

---

## Troubleshooting

### `TypeError: CASMO got an unexpected keyword argument`

You passed a name CASMO does not recognise. Check spelling against the
[API reference](api-reference.md).

### `DeprecationWarning: 'granularity' is deprecated and ignored since v0.4.0`

You are passing a v0.3 calibration parameter. It is safely ignored. Remove it, and set
`robustness` instead — see the [migration guide](migration-guide.md).

### `ValueError: optimizer got an empty parameter list`

Your model has no trainable parameters, or you passed an exhausted generator. Pass
`model.parameters()` directly, not a generator you already consumed.

### Loading a v0.3 checkpoint fails

v0.3 and v0.4 store different state (`exp_avg`/`exp_avg_sq` + `_group_states` vs. `m`/`s`).
Optimizer state is not portable across the redesign. Reload the model weights and start a
fresh optimizer.

### Training diverges

Lower the LR, or raise `robustness` — the absolute axis contracts the step when the signal
degrades and generally improves stability. If gradients are genuinely exploding, use
`torch.nn.utils.clip_grad_norm_` and consider `nan_guard=True` to fail loudly.
