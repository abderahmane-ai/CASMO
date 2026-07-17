# CASMO — agent instructions

CASMO is a single-file PyTorch optimizer (`casmo.py`) that scales each gradient
coordinate's update by its signal-to-noise ratio. It is a drop-in `torch.optim.AdamW`
replacement with Adam's memory footprint.

## Pre-commit checklist (always do these)

```sh
python3 -m ruff check casmo.py tests/ examples/ benchmarks/ verify_installation.py research/
python3 -m black --check casmo.py tests/ examples/ --line-length 100   # CI hard-gate
python3 -m pytest tests/ -q                                            # must be 71/71
python3 verify_installation.py                                         # must exit 0
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null
find . -name ".ruff_cache" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
```

`black --check` on `casmo.py tests/ examples/` at line-length 100 is a **build-failing**
step in `.github/workflows/lint.yml`. Run black before pushing or CI goes red.

## Project layout

- `casmo.py` — the entire optimizer. ~190 lines. Keep it that way.
- `tests/` — 71 tests, all fast (<3s total). Run from repo root.
- `research/validate_redesign.py` — reproducible multi-seed benchmark; the evidence for
  the algorithm. `research/REDESIGN.md` records what was measured and what was rejected.
- `benchmarks/b1..b5/` — large-scale benchmarks (need GPU / extra deps).
- `docs/` — user documentation. `CASMO_THEORY.md` — the derivation.

## The algorithm (v0.4)

```
m ← β₁m + (1-β₁)g                       # signal (first moment)
s ← β₂s + (1-β₂)(g-m)²                   # noise (centered "belief" variance)
sig = m̂², noise = ŝ, total = sig + noise
agar_i  = sig_i / max(total_i, tiny)                     # signal fraction ∈ [0,1]
trust   = c_min + (1-c_min)·mean(agar)                   # absolute pace  (scalar/tensor)
focus_i = clip(agar_i / mean(agar), rel_floor, 1.0)      # relative focus (downweight-only)
C_i     = trust**robustness · focus_i
θ ← θ(1-ηλ) - η·C_i·m̂_i/(√total_i + eps)
```

Invariants that must not be broken:

1. **Two EMAs only** (`m`, `s`). The Adam denominator is *reconstructed* as `√(m̂²+ŝ)`
   because `E[g²] = E[g]² + Var[g]`. Do not add a third EMA — it was measured as
   behaviourally identical and costs 50% more optimizer memory.
2. **AGAR is absolute, never calibrated.** Do not reintroduce a threshold calibrated
   against the run's own early distribution. That was v0.3's central bug: under label
   noise the baseline is itself noisy, so the optimizer concluded its noisy gradients were
   normal and suppressed nothing.
3. **`focus` is capped at 1.0 — downweight only.** Allowing amplification (>1) measured
   better on gradient noise but became unstable at high LR. Note this means
   `mean(focus) <= 1`, so focus does cost a little pace — say that, don't claim it is
   pace-neutral.
4. **No host sync in the hot path.** No `.item()` anywhere in `step()` — the reported
   metrics stay device tensors and `group_metrics()` materializes them on demand.
   (Removing a per-step `.item()` measured **1.19× faster** on a 40-layer model on MPS.)
   Also no `isnan().any()` by default (that is `nan_guard`, opt-in), no numpy, no `deque`.
   There is a test that asserts `step()` leaves metrics as tensors — do not "simplify" it
   back into a float.
5. **No epsilon in the AGAR denominator.** `agar = sig / max(total, tiny)`. Since
   `sig <= total`, a zero denominator implies a zero numerator, so no eps is needed. Adding
   one makes AGAR depend on the *absolute gradient scale* instead of the ratio it is defined
   as: with `+eps`, a fixed SNR of 100 reads 0.897 at gradient scale 1 but 0.470 at scale
   1e-4. `test_agar_is_invariant_to_gradient_scale` guards this.
6. **State stays standard.** Per-param state is `{step, m, s}`, so the parent
   `torch.optim.Optimizer` handles `state_dict`/`load_state_dict`. Do not add custom
   serialization — v0.3 did and it was a bug source.
7. `robustness=0, rel_floor=1` must reduce **exactly** to AdamW.

## Validate before you claim

This project's rule: **no design change lands without measurement.** Run

```sh
python3 research/validate_redesign.py    # ~25s, 5 seeds, 5 regimes
```

Four regimes matter and they disagree with each other — a change that helps one usually
hurts another. Check all four before concluding anything:

| Regime | What it catches |
|---|---|
| Clean | Gating that breaks or slows normal training |
| Label noise 15/30% | The headline claim; the reason CASMO exists |
| Gradient noise (DP-like) | Over-aggressive absolute suppression |
| High LR | Instability from amplification |

Current reference numbers (5 seeds, test accuracy unless noted):

| Regime | Adam | CASMO ρ=0.5 | CASMO ρ=1.0 |
|---|---|---|---|
| clean | 0.931 | 0.934 | 0.934 |
| label noise 30% | 0.675 | 0.701 | **0.797** |
| label noise 15% | 0.764 | 0.793 | **0.851** |
| grad noise σ=0.5 | **0.641** | 0.488 | 0.396 |
| high LR (steps) | 70 | **36** | 50 |

## Known trade-off (do not "fix" this)

Label noise and isotropic gradient noise want **opposite** policies. Label noise rewards
slowing down globally; DP-SGD-style injected noise punishes it under a fixed step budget.
This is why `robustness` is exposed rather than auto-tuned. If someone reports "CASMO is
worse under DP-SGD", the answer is `robustness=0.0–0.5`, not a code change.

Generalisation was checked across depth/width/dimension/noise (7 configs, 3 seeds):
**CASMO ρ=1 beats AdamW in 6/7**, and the margin grows with noise (+15.6 points at 50%
label corruption). The one loss is a config where both optimizers are near chance —
i.e. **under-fitting**. Rule of thumb: high ρ pays when the model *would otherwise
memorise noise*; it costs when the model has not yet fit the signal (short step budget,
LR too low, or isotropic noise). See `research/REDESIGN.md` §5–6.

The ~1.9× per-step wall-clock over Adam (measured ratio 1.88×; absolute ms varies by
machine, quote the ratio not the ms) is intrinsic to per-coordinate variance tracking.
v0.4 is already ~14% faster than v0.3, at identical optimizer memory (verified equal).

## History: what was wrong before v0.4

An earlier refactor removed the `DDEAdapter` / dynamic-`tau` machinery from `casmo.py` but
left its references everywhere. The fallout, all fixed in v0.4:

- `verify_installation.py` imported the deleted `DDEAdapter` → crashed on line 1.
- 4 tests referenced `group_state['tau_adapter']` → `KeyError`.
- B4 benchmark used `train_loader` 36 lines before assignment → `NameError`, so the
  README's advertised B4 result was unreproducible.
- `total_steps <= 249` raised `ValueError: tau_init_steps too small`.
- `current_agar`/`current_confidence` froze at their first value in `'parameter'` mode.
- `CHANGELOG` claimed `verify_installation.py` passed and cited a test file that never existed.
- `CASMO_THEORY.md` contradicted itself on the sigmoid center (median vs. mean).

Lesson encoded here: when removing a concept, grep for it everywhere
(`grep -rn "<name>" --include="*.py" --include="*.md" .`) before calling it done.

## Conventions

- Python ≥3.8, PyTorch ≥1.10. No numpy in `casmo.py`.
- Line length 100 (black + ruff agree).
- Tests are behavioural where it matters: assert CASMO *beats AdamW under label noise*,
  not just that it runs.
- Benchmark reports under `benchmarks/*/reports/` were produced with v0.3 and have **not**
  been re-run. They are historical; do not cite them as current results.
- Removed kwargs stay accepted with a `DeprecationWarning` (see `_DEPRECATED_KWARGS`) so
  downstream code keeps running. Unknown kwargs raise `TypeError`.

## Known headroom (measured, not yet taken)

**`foreach` / multi-tensor.** `step()` loops per parameter, so a model with many small
tensors pays a kernel launch per tensor. Measured headroom on the hardware available here
(60x256 MLP): `torch.optim.AdamW(foreach=True)` beats `foreach=False` by only **1.02x on
CPU / 1.11x on MPS**, and CASMO sits at ~2.2x AdamW either way — i.e. the cost is the
per-coordinate gate, not the launches. On **CUDA** the launch overhead is known to be far
more significant (it is why PyTorch defaults to foreach), so the win is likely larger
there, but **it has not been measured on CUDA and must not be claimed until it is**.

If you implement it: `torch._foreach_*` covers everything except the per-tensor mean, which
`torch._foreach_norm(agar, 1)` gives (L1 == sum, since `agar >= 0`), divided by `numel`.
Validate numerical equivalence against the current path before landing.

**`capturable` / CUDA graphs.** `state["step"]` is a Python int, which blocks CUDA-graph
capture. PyTorch gates the tensor-step variant behind an opt-in `capturable` flag precisely
because a device-tensor step costs performance in the common case — follow that pattern if
it is ever needed; do not make it unconditional.

**MPS/CUDA**: `casmo.py` contains zero device-specific code and works on both as-is;
`tests/test_device_and_groups.py` runs the suite on whatever accelerator is present. Keep it
device-agnostic.
