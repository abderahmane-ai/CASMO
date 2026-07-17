# Migration Guide

## From AdamW to CASMO

CASMO is a drop-in replacement. Keep your learning rate and weight decay:

```python
# Before
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# After
from casmo import CASMO
optimizer = CASMO(model.parameters(), lr=1e-3, weight_decay=0.01)
```

Then pick a `robustness` for your regime:

| Your situation | Setting |
|---|---|
| Noisy / crowd-sourced labels, long-tail classes, continual learning | `robustness=1.0` |
| General purpose, or you can raise the LR | `robustness=0.5` (default) |
| Clean data, DP-SGD, or you need maximum raw speed | `robustness=0.0` |

Everything else — LR schedulers, parameter groups, `state_dict`, closures, gradient
accumulation — works unchanged.

### What to expect

- **Clean data:** the same test accuracy as AdamW (measured 0.934 vs. 0.931). Training
  loss falls more slowly at higher `robustness`; that is the anti-memorization mechanism
  working, not a failure to converge.
- **Noisy labels:** materially better generalization (measured +13.5 points at 30% label
  noise with `robustness=1.0`).
- **Aggressive LR:** CASMO is often *faster* than Adam (38 vs. 70 steps at `lr=3e-2`)
  because confidence gating absorbs the instability a large step would otherwise cause.
- **Cost:** ~2× Adam's per-step wall-clock (elementwise variance tracking), at Adam's
  memory footprint.

---

## From CASMO v0.3 to v0.4

v0.4 replaces the calibration-based algorithm with a per-coordinate confidence gate on an
absolute scale. The rationale and measurements are in
[`research/REDESIGN.md`](../research/REDESIGN.md).

### Your code keeps running

The removed parameters are still accepted; they emit a `DeprecationWarning` and are
ignored:

```python
# v0.3 code -- still runs, warns, ignores the dead kwargs
optimizer = CASMO(
    model.parameters(),
    lr=1e-3,
    granularity='group',      # DeprecationWarning, ignored
    total_steps=10_000,       # DeprecationWarning, ignored
    tau_init_steps=500,       # DeprecationWarning, ignored
)
```

### Recommended update

```python
# v0.4
optimizer = CASMO(model.parameters(), lr=1e-3, robustness=1.0)
```

### Parameter mapping

| v0.3 | v0.4 | Notes |
|---|---|---|
| `tau_init_steps` | *(removed)* | No calibration phase. AGAR is absolute. |
| `tau_clip_range` | *(removed)* | No threshold to clip. |
| `granularity` | *(removed)* | The gate is always per-coordinate — strictly more expressive than either old mode. |
| `agar_clamp_factor` | *(removed)* | AGAR is bounded in `[0, 1]` by construction. |
| `total_steps` | *(removed)* | Only sized the calibration phase. |
| `c_min` | `c_min` | Unchanged meaning: floor on the trust factor. No longer auto-adapted. |
| — | `robustness` | **New.** The speed/robustness dial. |
| — | `rel_floor` | **New.** Floor on the relative focus factor. |
| — | `nan_guard` | **New.** Opt-in non-finite gradient check. |

### Behavioural changes to expect

1. **Noise robustness actually works now.** v0.3 measured 0.680 test accuracy at 30% label
   noise versus AdamW's 0.679 — essentially no benefit, because it calibrated its
   threshold against an already-noisy baseline. v0.4 at `robustness=1.0` measures 0.810.
2. **No 500-step calibration ramp.** v0.3 spent its first `tau_init_steps` at a reduced
   effective LR (mean confidence ≈ 0.70). v0.4 has no warm-up phase.
3. **NaN/Inf gradients no longer raise by default.** v0.3 scanned every gradient every
   step and raised `RuntimeError`. That cost two host syncs per parameter per step. Set
   `nan_guard=True` to restore the old behaviour.
4. **Checkpoints are not compatible.** v0.3 stored `exp_avg`/`exp_avg_sq` plus a custom
   `_group_states` blob; v0.4 stores `m`/`s`. Start `robustness` tuning from a fresh run,
   or reload only the model weights.
5. **Monitoring is fixed and now live.** Use `optimizer.group_metrics(idx)`. In v0.3,
   `current_agar` froze at the first value in `'parameter'` mode. (The
   `_group_states[idx]['current_agar']` access pattern still works for existing logging
   code.)

### Removed `DDEAdapter`

The `DDEAdapter` class was already gone from the v0.3 implementation but still referenced
by `verify_installation.py`, parts of the test suite, and the changelog. It does not
exist. Dynamic threshold adaptation is unnecessary in v0.4: AGAR is absolute, so there is
no threshold to adapt.
