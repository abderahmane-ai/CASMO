# CASMO — agent instructions

CASMO is a single-file PyTorch optimizer (`casmo.py`, ~260 lines) that scales each gradient
coordinate's update by its signal-to-noise ratio. It is a drop-in `torch.optim.AdamW`
replacement with **Adam's exact memory footprint**.

---

# 1. CURRENT STATE — read this first

**Version:** v0.4.0. A complete algorithm redesign; v0.3's algorithm is gone, not tweaked.

| | |
|---|---|
| Branch | `main`, clean, synced with origin |
| Tests | **81 passing**, ~4s, 8 files |
| CI | **green** (Tests + Lint) across ubuntu/windows/macos × Python 3.8–3.12 |
| Tag | `v0.4.0` |
| PyPI | **0.3.0 is the latest published — 0.4.0 is NOT published** (see §10) |

**The single most important fact:** every performance claim CASMO makes rests on
**synthetic MLP experiments only**. No real dataset has ever been run against v0.4. The
`benchmarks/b1..b5` reports are all v0.3 historical and are banner-marked as such.

**The immediate next task is therefore §9: run B1.** Not more algorithm work.

---

# 2. Pre-commit checklist (always do these)

```sh
python3 -m ruff check casmo.py tests/ examples/ benchmarks/ verify_installation.py research/
python3 -m black --check casmo.py tests/ examples/ --line-length 100   # CI hard-gate
python3 -m pytest tests/ -q                                            # must be 81/81
python3 verify_installation.py                                         # must exit 0
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null
find . -name ".ruff_cache" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
```

`black --check` on `casmo.py tests/ examples/` at line-length 100 is a **build-failing**
step in `.github/workflows/lint.yml`. Run black before pushing or CI goes red.

**Gotcha that bit us:** the local machine has numpy installed globally, CI does not install
it (CASMO no longer depends on it). A test that imports numpy passes locally and fails all
12 CI jobs. To reproduce CI conditions:

```sh
python3 -m venv /tmp/ci && /tmp/ci/bin/pip install -q -e . pytest && /tmp/ci/bin/python -m pytest tests/ -q
```

---

# 3. Project layout

- `casmo.py` — the entire optimizer. Keep it single-file and small.
- `tests/` — 81 tests, all fast. Run from repo root.
- `research/validate_redesign.py` — the reproducible 5-seed benchmark; **the evidence**.
  `research/REDESIGN.md` — the full evidence log: what was measured, what was *refuted*,
  what is still unknown. Read it before changing the algorithm.
- `benchmarks/b1..b5/` — large-scale benchmarks (GPU / extra deps). See §8–9.
- `docs/` — user documentation. `CASMO_THEORY.md` — the derivation.

---

# 4. The algorithm (v0.4)

```
diff = g - m                             # measure BEFORE m absorbs g  (invariant 6)
m  ← β₁m + (1-β₁)g                       # signal (first moment)
s  ← β₂s + (1-β₂)·diff²                  # noise (centered variance)
sig = m̂², noise = ŝ, total = sig + noise
agar_i  = sig_i / max(total_i, tiny)                     # signal fraction ∈ [0,1]
trust   = c_min + (1-c_min)·mean(agar)                   # absolute pace (per tensor)
focus_i = clip(agar_i / mean(agar), rel_floor, 1.0)      # relative focus (downweight-only)
C_i     = trust**robustness · focus_i
θ ← θ(1-ηλ) - η·C_i·m̂_i/(√total_i + eps)
```

Public API: `CASMO(params, lr, betas, eps, weight_decay, c_min, robustness, rel_floor,
nan_guard)` plus `group_metrics(group_idx) -> {"agar": float, "confidence": float}`.

Mental model: **`trust` is a brake** (whole tensor is noise → slow down; this is what
resists memorising bad labels). **`focus` is steering** (shift the step toward coordinates
above their tensor's average signal fraction).

---

# 5. Invariants that must not be broken

Each of these was established by measurement. Breaking one silently degrades the optimizer.

1. **Two EMAs only** (`m`, `s`). The Adam denominator is *reconstructed* as `√(m̂²+ŝ)`
   because `E[g²] = E[g]² + Var[g]`. A separate `v` EMA measured behaviourally identical
   and costs 50% more optimizer memory.
2. **AGAR is absolute, never calibrated.** Do not reintroduce a threshold calibrated
   against the run's own early distribution. That was v0.3's central bug: under label noise
   the baseline is *itself* noisy, so the optimizer concluded its noisy gradients were
   normal and suppressed nothing (v0.3 scored 0.680 vs AdamW's 0.679 at 30% label noise —
   i.e. delivered none of its reason for existing).
3. **`focus` is capped at 1.0 — downweight only.** Amplification (>1) measured *better* on
   gradient noise but destabilised at high LR. Note `mean(focus) <= 1`, so focus does cost
   a little pace — say that, don't claim it is pace-neutral.
4. **No host sync in `step()`.** No `.item()` anywhere in the hot path — metrics stay device
   tensors; `group_metrics()` materializes on demand. (Removing a per-step `.item()`
   measured **1.19× faster** on a 40-layer model on MPS.) Also no `isnan().any()` by default
   (that is `nan_guard`, opt-in), no numpy, no `deque`.
   Guarded by `test_step_does_not_materialize_metrics`.
5. **No epsilon in the AGAR denominator.** `agar = sig / max(total, tiny)`. Since
   `sig <= total`, a zero denominator implies a zero numerator. Adding an eps makes AGAR
   depend on the *absolute gradient scale* instead of the ratio it is defined as: a fixed
   SNR of 100 reads 0.897 at gradient scale 1 but 0.470 at 1e-4.
   Guarded by `test_agar_is_invariant_to_gradient_scale`.
6. **Center the deviation on `m_{t-1}`, not `m_t`.** Compute `diff = grad - m` BEFORE
   updating `m`. Reordering these two lines looks harmless and is not: `g - m_t ==
   β₁·(g - m_{t-1})`, so centering on the updated mean estimates `β₁²·Var[g]` — 19% low at
   β₁=0.9, 75% low at β₁=0.5 — which **silently couples `robustness` to `betas`** (measured:
   robustness=1 swung 0.792→0.707 as β₁ went 0.9→0.5; corrected it holds 0.808→0.794).
   Guarded by `test_noise_ema_tracks_true_variance` + `test_robustness_does_not_depend_on_betas`.
7. **State stays standard.** Per-param state is `{step, m, s}`, so the parent
   `torch.optim.Optimizer` handles `state_dict`/`load_state_dict`. Do not add custom
   serialization — v0.3 did and it was a bug source.
8. **`robustness=0, rel_floor=1` must reduce exactly to AdamW.**
9. **Stay device-agnostic.** `casmo.py` contains zero device-specific code; it works on
   CUDA and MPS as-is. `tests/test_device_and_groups.py` exercises whatever accelerator is
   present.

---

# 6. Validate before you claim

**This project's rule: no design change lands without measurement.**

```sh
python3 research/validate_redesign.py    # ~25s, 5 seeds, 5 regimes, CPU
```

Four regimes matter and **they disagree with each other** — a change that helps one usually
hurts another. Check all four before concluding anything:

| Regime | What it catches |
|---|---|
| Clean | Gating that breaks or slows normal training |
| Label noise 15/30% | The headline claim; the reason CASMO exists |
| Gradient noise (DP-like) | Over-aggressive absolute suppression |
| High LR | Instability from amplification |

**Current reference numbers** (5 seeds, test accuracy unless noted). If your change moves
these, you must explain why:

| Regime | Adam | CASMO ρ=0.5 (default) | CASMO ρ=1.0 |
|---|---|---|---|
| clean | 0.931 | 0.935 | 0.933 |
| clean (steps to 0.05 loss) | **82** | 180 | 370 |
| label noise 30% | 0.675 | 0.713 | **0.810** |
| label noise 15% | 0.764 | 0.803 | **0.857** |
| grad noise σ=0.5 | **0.641** | 0.472 | 0.383 |
| high LR (steps) | 70 | **38** | 52 |

Wall-clock: **~1.9× Adam** (ratio 1.88× is stable; absolute ms drifts with machine load —
**quote the ratio, not the ms**). Optimizer memory: **identical to Adam** (verified equal).

---

# 7. Known trade-off — do NOT try to "fix" this

Label noise and isotropic gradient noise want **opposite** policies. Label noise rewards
slowing down globally; DP-SGD-style injected noise punishes it under a fixed step budget.
This is why `robustness` is exposed rather than auto-tuned. If someone reports "CASMO is
worse under DP-SGD", the answer is `robustness=0.0–0.5`, **not a code change**.

**Why this is probably irreducible, not merely unexplored:** `robustness` trades "fit the
training set" against "refuse to". Whether fitting the training data is *desirable* is a
property of the **task**, not of the gradients — so an optimizer that sees only gradients
cannot infer it. v0.3 tried to infer it from the run's own distribution, and that is exactly
why it failed (invariant 2). Auto-tuning ρ needs a signal from outside the optimizer's
contract (e.g. validation loss).

Generalisation across depth/width/dimension/noise (7 configs, 3 seeds): **CASMO ρ=1 beats
AdamW in 6/7**, margin growing with noise (**+16.4 points at 50% label corruption**). The
one loss is a config where both optimizers sit near chance — i.e. **under-fitting**.

Rule of thumb: **high ρ pays when the model would otherwise memorise noise; it costs when
the model has not yet fit the signal** (short step budget, LR too low, or isotropic noise).

---

# 8. Benchmark status (audited 2026-07-17)

All five were audited for leakage and unfair comparison. **Do not re-audit from scratch —
this is the result.**

**Verified clean (data pipelines):**
- **B1**: noise applied to the train split only; test split disjoint and all-clean; both
  splits derive from the same seeded shuffle.
- **B2**: long-tail subsampling happens strictly inside CIFAR-100's *train* file; the test
  split is a separate balanced file. Oversampling cannot duplicate across splits.
- **B5**: split is temporal (by date); sequences are built within each slice so none spans
  the boundary; no scaler is fit on the full series; backtest loader unshuffled. No lookahead.
- **B3**: `make_private()` wraps every arm identically — DP treatment is uniform.
- **None** select on the test set. All report final/full curves, not best-epoch.

**Confounds found and FIXED** (root cause: `torch.optim.AdamW` silently defaults to
`weight_decay=1e-2`, and the code left it unset):
- B1 ran CASMO on default betas `(0.9, 0.999)` vs AdamW `(0.9, 0.98)` → both now `(0.9, 0.98)`.
- B3 gave CASMO `wd=1e-4`, AdamW `1e-2`, SGD none → now one shared `WEIGHT_DECAY` constant.
- B5 gave CASMO `wd=0.0`, AdamW `1e-2`, while measuring turnover/Sharpe → now explicit for both.

**Bug found and FIXED:** B5 builds train and test with separate constructor calls, each
with its own yfinance download and silent synthetic fallback — a partial failure would
train on real data and backtest on synthetic. Source is now recorded per split; mismatch
raises; synthetic runs announce themselves.

**Outstanding methodology gap:** **B2–B5 are still single-seed (42).** Their numbers are
anecdotes. **Do not quote them as results.** B1 now takes `--seeds` (default `0 1 2`),
reseeding split + noise + init per seed, running both arms paired, reporting mean ± std.

---

# 9. THE NEXT TASK: run B1

Every claim rests on synthetic MLPs. B1 is the headline, the cheapest, and the only
multi-seed benchmark. **Run this first:**

```sh
cd benchmarks/b1_grokking
python3 train.py --epochs 2000 --seeds 0 1 2 3 4
```

Prints per-seed final val accuracy plus mean ± std and the CASMO−AdamW delta.

Then, in value order (all need extra deps):

```sh
pip install torchvision matplotlib && cd benchmarks/b2_long_tail_cifar100 && python3 train.py
pip install opacus                && cd benchmarks/b3_dp_sgd              && python3 train.py
pip install yfinance pandas       && cd benchmarks/b5_noisy_timeseries    && python3 train.py
```

Skip **B4** (Gemma-2-2B + LoRA + 4-bit) — heaviest, least diagnostic.

**Worth running B3 specifically to try to falsify CASMO.** Synthetic data says CASMO at
ρ=0.5 is *worse* than Adam under isotropic gradient noise (0.472 vs 0.641). If B3 confirms
it, **drop the B3 claim from the README** rather than defend it. If B3 contradicts it, the
synthetic grad-noise task is unrepresentative and REDESIGN.md §6 needs revising.

**Expect B1's numbers to move.** The old "90.4% vs 24.0%" was measured with the β₂
confound, single-seed, on v0.3. The honest number will likely be less dramatic. That is the
point of running it. Update `README.md`, `benchmarks/README.md` and the B1 report with
whatever comes out — including if it is bad.

**Do not overwrite `benchmarks/*/results/*.png` with smoke-test output.** They are committed
v0.3 artifacts. `git checkout -- benchmarks/<b>/results/` restores them.

---

# 10. Release / PyPI

The publish workflow (`.github/workflows/publish.yml`) triggers on **GitHub Release
published** (not on tag push) and needs `secrets.PYPI_API_TOKEN`.

**The repo currently has ZERO Actions secrets, so publishing cannot work.** Do not create a
Release until the token exists — it would build, then fail at `twine upload`, leaving a red
X and no publish. A bare tag push triggers nothing and is safe.

To publish, the user must first add the token (never paste it into chat):

```sh
gh secret set PYPI_API_TOKEN          # token from https://pypi.org/manage/account/token/
gh release create v0.4.0 --title "v0.4.0 — per-coordinate SNR gate" --notes-file <notes>
# OR, manual one-off:  python3 -m build && python3 -m twine upload dist/*
```

Packaging notes: `pyproject.toml` is the **single source of truth**; `setup.py` is a 3-line
shim. `[tool.setuptools] py-modules = ["casmo"]` is required (flat layout). Dependencies are
**torch only** — `casmo.py` imports nothing else but stdlib; numpy lives in the `[dev]`
extra. Verified by installing the wheel into a numpy-free venv and training.

**v0.4.0 is a behavioural change for existing users.** Old kwargs still work with a
`DeprecationWarning`, but optimizer checkpoints are not portable from v0.3 and training
dynamics differ.

---

# 11. What is still unknown (do not overstate the evidence)

1. **No real dataset has been run against v0.4.** Biggest gap by far. See §9.
2. **`β₂=0.999` for the variance is inherited from Adam unexamined.** No reason a *variance*
   estimator wants Adam's second-moment horizon.
3. **The AGAR→confidence map is a guess.** `trust` is linear in mean AGAR; `focus` is a
   clipped ratio. Neither was compared against alternatives.
4. **`ρ=0.5` default is a bet, not an optimum.** It is measurably worse than Adam under
   gradient noise (0.472 vs 0.641) buying +3 points on label noise. Right only if user data
   is more often label-noisy than DP-noisy.
5. **`focus` barely earns its place.** Ablated: it buys **+1.2 to +1.6 points** of
   label-noise accuracy for ~8% clean speed and one extra hyper-parameter. Against AdamW,
   `trust` alone delivers +10.6 and focus adds +1.2 on top — **trust is ~90% of the value**.
   Kept because the gain is real and consistent, but **it is the first thing to cut** if the
   design needs simplifying. (REDESIGN.md §8.1)
6. **`foreach`/CUDA-graph headroom unmeasured on CUDA.** See §12.

---

# 12. Known headroom (measured, not yet taken)

**`foreach` / multi-tensor.** `step()` loops per parameter, so a model with many small
tensors pays a kernel launch per tensor. Measured on the hardware available here (60×256
MLP): `AdamW(foreach=True)` beats `foreach=False` by only **1.02× on CPU / 1.11× on MPS**,
and CASMO sits at ~2.2× AdamW either way — i.e. the cost is the per-coordinate gate, not the
launches. On **CUDA** launch overhead is known to be far larger (it is why PyTorch defaults
to foreach), so the win is plausibly bigger — but **it has not been measured on CUDA and
must not be claimed until it is.**

If you implement it: `torch._foreach_*` covers everything except the per-tensor mean, which
`torch._foreach_norm(agar, 1)` gives (L1 == sum, since `agar >= 0`), divided by `numel`.
Validate numerical equivalence against the current path before landing.

**`capturable` / CUDA graphs.** `state["step"]` is a Python int, which blocks CUDA-graph
capture. PyTorch gates the tensor-step variant behind an opt-in `capturable` flag precisely
because a device-tensor step costs performance in the common case — follow that pattern if
needed; **do not make it unconditional.**

---

# 13. History: what was wrong before v0.4

An earlier refactor removed the `DDEAdapter` / dynamic-`tau` machinery from `casmo.py` but
left its references everywhere:

- `verify_installation.py` imported the deleted `DDEAdapter` → crashed on line 1.
- 4 tests referenced `group_state['tau_adapter']` → `KeyError`.
- B4 used `train_loader` 36 lines before assignment → `NameError`, so the README's
  advertised B4 result was unreproducible.
- `total_steps <= 249` raised `ValueError: tau_init_steps too small`.
- `current_agar`/`current_confidence` froze at their first value in `'parameter'` mode.
- `CHANGELOG` claimed `verify_installation.py` passed and cited a test that never existed.
- `CASMO_THEORY.md` contradicted itself on the sigmoid center (median vs mean).

**Lesson:** when removing a concept, grep for it everywhere before calling it done:
`grep -rn "<name>" --include="*.py" --include="*.md" .`

**Second lesson, learned the hard way in v0.4 itself:** I wrote "no host-device
synchronization in the hot path" in the theory doc while `step()` called `.item()` every
step. Docs that describe intent rather than code are how v0.3 rotted. **Verify the claim
against the code before writing it.**

---

# 14. Conventions

- Python ≥3.8, PyTorch ≥1.10. **No numpy in `casmo.py`** (it is not a dependency).
- Line length 100 (black + ruff agree).
- Tests are behavioural where it matters: assert CASMO *beats AdamW under label noise*, not
  merely that it runs. New invariants get a regression test that **provably fails on the old
  behaviour** — check that it does.
- Benchmark reports under `benchmarks/*/reports/` are v0.3 historical, banner-marked. Do not
  cite as current.
- Removed kwargs stay accepted with a `DeprecationWarning` (`_DEPRECATED_KWARGS`) so
  downstream code keeps running. Unknown kwargs raise `TypeError`.
- Prefer deleting to adding. `casmo.py` is small on purpose.
