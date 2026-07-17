# Changelog

All notable changes to CASMO will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-07-17

Algorithm redesign. Every change below was validated by the reproducible experiment in
[`research/validate_redesign.py`](research/validate_redesign.py); the measurements and the
rejected alternatives are recorded in [`research/REDESIGN.md`](research/REDESIGN.md).

### Changed — algorithm

- **Per-coordinate confidence.** AGAR is now applied per coordinate instead of being
  averaged into a single scalar per parameter group. The gate reshapes the update
  direction rather than only rescaling its length.
- **Correct variance estimate.** Noise is now `s = EMA[(g - m_{t-1})²]` (centered) instead
  of `v - m²`, which mixed the `beta1` (~10 step) and `beta2` (~1000 step) horizons and was
  therefore a biased SNR estimate.
- **The deviation is centered on `m_{t-1}`, not `m_t`.** Since `g - m_t == beta1*(g - m_{t-1})`,
  centering on the updated mean estimates `beta1^2 * Var[g]` — 19% low at `beta1=0.9` and 75%
  low at `beta1=0.5`. Because the shrinkage factor *is* `beta1^2`, that silently coupled
  `robustness` to `betas`: at 30% label noise with `robustness=1` fixed, the biased form swung
  0.792 → 0.707 as `beta1` went 0.9 → 0.5 (spread 0.085); corrected, it holds at 0.808 → 0.794
  (spread 0.016). Verified against a known distribution: the estimator now tracks `Var[g]` to
  within a few percent instead of 0.79×.
- **Absolute scale, no calibration.** AGAR is bounded in `[0, 1]` by construction, so the
  calibration phase, the `tau` threshold, the sigmoid mapping and the freeze are all gone.
  *This fixes the algorithm's central flaw:* calibrating against the early-training AGAR
  distribution meant that when noise was present from step one — the label-noise case
  CASMO exists for — the baseline was itself noisy and no suppression occurred. Measured:
  v0.3 scored 0.680 test accuracy at 30% label noise vs. AdamW's 0.679.
- **Two-axis confidence:** `C_i = trust**robustness * focus_i`, where `trust` is an
  absolute per-tensor pace and `focus` is a mean-normalized, downweight-only per-coordinate
  weight.
- **Adam denominator reconstructed** as `sqrt(m̂² + ŝ)` instead of tracking a third EMA,
  since `E[g²] = E[g]² + Var[g]`. Optimizer state is now two EMAs per parameter — the same
  footprint as Adam.

### Added

- `robustness` (default `0.5`): the speed/robustness dial. `0` is AdamW-like; `1` is
  maximally noise robust.
- `rel_floor` (default `0.1`): floor on the relative focus factor.
- `nan_guard` (default `False`): opt-in non-finite gradient check.
- `group_metrics(group_idx)`: live AGAR/confidence reporting.
- `research/validate_redesign.py` and `research/REDESIGN.md`: reproducible evidence.
- `CLAUDE.md`: codebase context and conventions.

### Removed

- `tau_init_steps`, `tau_clip_range`, `granularity`, `agar_clamp_factor`, `total_steps` —
  all calibration-era parameters. Still accepted with a `DeprecationWarning` and ignored,
  so existing call sites keep running.
- The custom `state_dict`/`load_state_dict` overrides. Checkpointing now uses the standard
  `torch.optim.Optimizer` machinery.

### Fixed

- **`verify_installation.py` crashed on import.** It imported `DDEAdapter`, a class removed
  from `casmo.py` in an earlier incomplete refactor. The first command a new user runs
  failed immediately.
- **4 tests failed with `KeyError: 'tau_adapter'`.** `test_calibration.py` and
  `test_state_dict.py` referenced a `tau_adapter` object that no longer existed.
- **B4 continual-learning benchmark raised `NameError`.** `train_loader` was referenced 36
  lines before it was assigned, so the CASMO branch could never run — meaning the README's
  advertised B4 result was not reproducible.
- **`total_steps` crashed `__init__` for modest values.** `total_steps <= 249` produced
  `ValueError: tau_init_steps too small`, defeating the parameter's own purpose.
- **Monitoring froze in `'parameter'` granularity.** `current_agar` / `current_confidence`
  were guarded by `if ... is None`, so they were written once and never updated, making all
  AGAR logging and plots in that mode meaningless.
- **Calibration triggered early in `'parameter'` mode.** The AGAR buffer collected one
  sample per parameter per step but was compared against a per-step budget, so a 5-parameter
  group calibrated at step 10 instead of step 50 — on un-warmed moments.
- **Per-step host synchronization.** `isnan().any()` and `isinf().any()` ran per parameter
  per step. Now opt-in via `nan_guard`.
- Undefined names in benchmark code (`args` in the B2 notebook), unused imports, bare
  `except:` clauses, and formatting drift that broke the `black --check` CI gate on 12 files.

### Documentation

- Rewrote `CASMO_THEORY.md`, `README.md` and all of `docs/` against the implementation.
  The theory document previously contradicted itself (§2.3 said the sigmoid was centered on
  the median, §2.4's formula used the mean) and the README expanded AGAR as "Adaptive
  Gradient-Aware Regularization" while the code said "Adaptive Gradient Alignment Ratio".
- Documented an honest limitation: under isotropic gradient noise (DP-SGD-like), high
  `robustness` **hurts** (Adam 0.641 vs. CASMO ρ=1 0.383). Label noise and injected
  gradient noise want opposite policies.
- Removed false claims from this changelog's history — the 0.2.0 entry asserted
  "`verify_installation.py` passes with no regressions" (it crashed) and cited a
  reproduction test `tests/reproduce_ddea_fix.py` that does not exist.

### Measured

| Regime (5 seeds) | Adam | CASMO v0.3 | CASMO v0.4 |
|---|---|---|---|
| 30% label noise (test acc) | 0.675 | 0.680 | **0.810** (ρ=1) |
| 15% label noise (test acc) | 0.764 | 0.773 | **0.857** (ρ=1) |
| Clean data (test acc) | 0.931 | 0.933 | 0.935 |
| High LR (steps to converge) | 70 | 50 | **38** (ρ=0.5) |
| Wall-clock (relative to Adam) | **1.00×** | 2.54× | 2.16× |
| Optimizer state | **1.00×** | 1.5× | **1.00×** |

Generalisation of the label-noise result was checked across 7 configurations varying depth,
width, dimension, class count and noise level: **CASMO ρ=1 beats AdamW in 6/7**, with the
margin widening as noise rises (+16.4 points at 50% label corruption). The single loss is a
configuration where both optimizers sit near chance, i.e. the model is under-fitting.

## [0.3.0] - 2025-01-09

### Added
- Documentation suite in `docs/`, examples in `examples/`, GitHub CI/CD workflows and
  issue templates, `CONTRIBUTING.md`, and `benchmarks/README.md`.

### Changed
- Benchmarks renumbered sequentially (b1–b5) and updated to the then-current interface.

### Removed
- Incomplete benchmarks (`b1_noisy_cifar10`, `b5_gan_stability`) that had no reports.

## [0.2.0] - 2025-11-28

### Changed
- **Benchmark B3 (DP-SGD)** — standardized hyperparameters across CASMO, DP-AdamW and
  DP-SGD so all optimizers use the same learning rate.

> **Historical note.** This release's original notes described a "critical DDEA integration
> fix" centred on `group_state['tau_adapter'].tau`. The `DDEAdapter` class was subsequently
> removed without cleaning up its references, leaving a broken installer, four failing
> tests, and a changelog describing machinery that no longer existed. v0.4.0 removes the
> concept entirely: AGAR is on an absolute scale, so there is no threshold to adapt.

## [0.1.0] - 2025-11-22

### Added
- Initial release: the AGAR metric, sigmoid confidence mapping, group- and
  parameter-level granularity modes, test suite, and benchmark suite.
