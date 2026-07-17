# CASMO v0.4 Redesign: Evidence Log

This document records the experiments that produced the v0.4 algorithm. Every design
decision below was made **after** measurement, not before. Several initial hypotheses
were refuted by the data and abandoned.

Reproduce with:

```bash
python research/validate_redesign.py
```

All numbers: 5 seeds, mean ± population stdev, 2-layer MLP (20→64→64→3) on synthetic
data, 600 optimizer steps. `steps` = optimizer steps to reach 0.05 train loss (lower is
faster; capped at 600). Test labels are always clean, so `test_acc` measures true
generalization.

---

## 1. What was wrong with the v0.3 design

The v0.3 optimizer computed a per-element AGAR, **averaged it to a single scalar per
parameter group**, calibrated a threshold τ against the AGAR distribution of the first
~500 steps, froze τ, and mapped AGAR→confidence through a sigmoid centered on τ.

Three problems, all measured:

1. **It didn't deliver its headline claim.** At 30% label noise, v0.3 scored 0.680 test
   accuracy vs. AdamW's 0.679 — no meaningful robustness.
2. **The calibration was self-defeating.** Calibrating against the early-training AGAR
   distribution means that when noise is present from step one (exactly the label-noise
   case), the baseline is itself noisy, and the optimizer decides its noisy gradients are
   normal. *A relative reference cannot detect a pervasive problem.*
3. **A scalar gate discards the signal.** AGAR is computed per element, then collapsed to
   one number for a whole group — potentially millions of parameters share one gate.

## 2. Ablation: which mechanism actually does the work?

Two candidate gate families were implemented and measured (`gate` applied to the AdamW
update; `agar_i` is the per-coordinate signal fraction, `ā` its mean):

| gate | form | clean steps | label-noise 30% test | grad-noise test | high-LR |
|---|---|---|---|---|---|
| Adam (none) | — | 82 | 0.675 | 0.641 | 0.921 |
| v0.3 CASMO | scalar, calibrated | 128 | 0.680 | 0.537 | 0.930 |
| **absolute** | `c_min+(1-c_min)·agar_i` | 206 | **0.734** | 0.377 | 0.936 |
| **relative** | `agar_i/ā` (cap 5) | **52** | 0.682 | **0.712** | **0.453 (unstable)** |
| scalar-absolute | `c_min+(1-c_min)·ā` | 312 | **0.784** | 0.353 | 0.933 |

Findings that shaped the design:

- **H1 (confirmed): a mean-preserving relative gate removes the clean-data slowdown.**
  It was *faster* than Adam (52 vs. 82 steps) because it reallocates the step toward
  reliable coordinates.
- **H2 (refuted): label-noise robustness does NOT survive mean-preservation.** The
  relative gate scored 0.682 — no better than Adam. The robustness comes from *absolute
  global slowdown*, not from coordinate reallocation. This was the key negative result.
- **Amplification is unstable.** Letting the relative gate exceed 1 (cap 5) gave the best
  gradient-noise number (0.712) but broke down at high LR (0.453). The shipped design
  therefore caps focus at **1.0 — downweight only**.

Conclusion: the two mechanisms are complementary and neither alone is sufficient, so the
final design multiplies them: `C_i = trust^ρ · focus_i`.

## 3. Tuning the robustness dial (ρ)

| ρ | clean steps | label-noise 30% | label-noise 15% | grad-noise σ=0.5 | high-LR |
|---|---|---|---|---|---|
| Adam | 82 | 0.675 | 0.764 | 0.641 | 0.921 |
| 0.0 | 90 | 0.675 | 0.771 | 0.623 | 0.925 |
| **0.5** (default) | 180 | 0.701 | 0.793 | 0.472 | **0.932 / 36 steps** |
| 1.0 | 370 | **0.810** | **0.857** | 0.383 | 0.932 / 50 steps |

The dial is cleanly monotonic in both directions, which is why it is exposed as a
parameter rather than hard-coded. **Default ρ = 0.5**: it clearly beats Adam and v0.3 on
label noise, matches Adam's clean-data test accuracy (0.934 vs. 0.931), and is the
fastest and most accurate configuration at high LR.

## 4. Efficiency: eliminating the third EMA

Since `E[g²] = E[g]² + Var[g]`, Adam's denominator `√v̂` can be reconstructed as
`√(m̂² + ŝ)` from quantities already computed for AGAR. Validated as behaviourally
equivalent:

| denominator | EMAs/param | clean steps | label-noise 30% | high-LR |
|---|---|---|---|---|
| separate `v` EMA | 3 | 170 | 0.700 | 36 |
| **reconstructed `√(m̂²+ŝ)`** | **2** | 160 | 0.701 | 36 |

Identical behaviour at **Adam's memory footprint** (verified: 13.65 MB of optimizer state
for both, on a 512→1024→1024→128 MLP).

Wall-clock, all three measured in one run (512→1024→1024→128, 300 steps):

| optimizer | ms/step | vs Adam |
|---|---|---|
| Adam | 4.43 | 1.00× |
| CASMO v0.3 | 11.26 | 2.54× |
| **CASMO v0.4** | **9.59** | **2.16×** |

v0.4 is ~14% faster than v0.3. Absolute timings drift with machine load — a later
5×300-step median gave Adam 5.48 / v0.4 10.28 — but the **v0.4-vs-Adam ratio is stable at
1.9–2.2×**, so quote the ratio rather than the milliseconds. The overhead is the intrinsic
cost of per-coordinate variance tracking plus the elementwise gate.

v0.4 also removes from the hot path: the per-parameter `isnan`/`isinf` scans (two host
syncs per parameter per step, now opt-in via `nan_guard`), the numpy/deque calibration
buffers, and the group-concatenation reuse buffers.

## 5. Does the result generalise, or is it an artifact?

The tuning above used one architecture and one data distribution. To check that the
label-noise result is not overfit to that setup, `robustness=1.0` was re-tested against
AdamW across varied depth, width, dimensionality, class count, noise level and learning
rate (3 seeds each, test accuracy on clean labels):

| Config | AdamW | CASMO (ρ=1) | Δ |
|---|---|---|---|
| deep (4 layers), d=20, k=3, noise=.3 | 0.696 | **0.781** | +0.086 |
| wide (256), d=20, k=3, noise=.3 | 0.707 | **0.743** | +0.037 |
| shallow (1 layer), d=50, k=5, noise=.3 | 0.584 | **0.648** | +0.064 |
| d=100, k=10, noise=.4, lr=1e-3 | **0.364** | 0.348 | −0.015 |
| small data (n=400), noise=.2 | 0.715 | **0.727** | +0.012 |
| **noise=.5 (half the labels corrupted)** | 0.573 | **0.729** | **+0.156** |
| lr=1e-2, noise=.3 | 0.688 | **0.719** | +0.032 |

**CASMO wins 6/7.** The advantage grows with the noise level, which is the expected
signature of the mechanism: at 50% corruption the gap reaches +16.4 points.

The single loss is informative. In `d=100, k=10, noise=.4, lr=1e-3` both optimizers score
~0.35 on a 10-class problem — the model is **under-trained** within the step budget, and
slowing it down costs more than the memorisation it prevents. This is the same underlying
failure mode as §6: absolute suppression is harmful whenever the model has not yet fit the
signal.

## 6. Honest limitation

Isotropic gradient noise (DP-SGD-like) and label noise want **opposite** policies. Label
noise rewards slowing down; isotropic noise punishes it under a fixed step budget. No
single fixed ρ is optimal for both:

| | Adam | ρ=0 | ρ=1 |
|---|---|---|---|
| label noise 30% | 0.675 | 0.675 | **0.810** |
| grad noise σ=0.5 | **0.641** | 0.623 | 0.383 |

This is a real trade-off, not a bug, and is the reason `robustness` is a first-class
parameter. Use low ρ for DP-SGD, high ρ for label noise. The B3 (DP-SGD) benchmark is
configured with ρ=0.5 accordingly.

**Rule of thumb:** high ρ pays off when the model *would otherwise memorise noise*. It
costs you when the model is still under-fitting — whether because the noise is isotropic,
the step budget is short, or the learning rate is too low to converge.

## 7. Post-review refinements

An external review of the shipped v0.4 raised four points. Each was checked by measurement
rather than accepted or dismissed on argument.

### 7.1 Host synchronization in `step()` — valid, fixed

`step()` called `.item()` on the accumulated metrics once per group per step, which blocks
the host on the accelerator. `CASMO_THEORY.md` simultaneously claimed "no host-device
synchronization in the hot path" — the same class of doc/code contradiction this redesign
set out to remove from v0.3.

Metrics are now accumulated as device tensors and materialized only in `group_metrics()`.
Measured on a 40-layer 256-wide MLP on MPS:

| | ms/step |
|---|---|
| with per-step `.item()` | 15.30 |
| deferred (tensors) | **12.89** |

**1.19× faster (15.7%)**, and the docs now match the code.

### 7.2 Epsilon in the AGAR denominator — valid, fixed, but not results-changing

`agar = signal / (total + eps)` makes AGAR depend on the *absolute gradient scale*, not the
ratio it is defined as. Holding the true SNR fixed at 100 and varying only the scale:

| gradient scale | AGAR with `+eps` | AGAR true |
|---|---|---|
| 1e0 | 0.9901 | 0.9901 |
| 1e-3 | 0.9804 | 0.9901 |
| 1e-4 | 0.4975 | 0.9901 |
| 1e-5 | 0.0099 | 0.9901 |

Now computed as `signal / max(total, tiny)`; since `signal <= total`, a zero denominator
implies a zero numerator, so no epsilon is needed.

**Honest scope of the fix.** In real training the coordinates that fall below eps are also
genuinely noise-dominated, so the two effects coincide: on a long clean run the `+eps`
reading was 0.0001 where the true value was 0.0003 — both round to "no signal". Re-running
all five regimes with the fix produced **identical results within seed noise**. It is a
correctness fix that makes the implementation match the documented mathematics (and would
matter in pure bf16, where gradient² routinely sits under 1e-8); it is **not** a bug that
was changing outcomes. `test_agar_is_invariant_to_gradient_scale` fails on the old formula
(spread 0.427) and passes on the new one (spread 6e-8).

### 7.3 `focus` is a dampener, not a reweighting — valid, documentation fixed

The cap at 1.0 means `mean(focus) <= 1`, so focus does slow the overall pace slightly; the
docstring claimed it reweighted "without changing the overall pace". The cap itself stays —
§2 measured that allowing amplification destabilises at high LR — but the description now
says what the code does.

### 7.4 `foreach` / multi-tensor — valid direction, deliberately not taken yet

Measured headroom on a 60×256 MLP, where kernel-launch overhead should dominate:

| | CPU | MPS |
|---|---|---|
| AdamW `foreach=True` | 11.65 ms | 9.29 ms |
| AdamW `foreach=False` | 11.84 ms | 10.35 ms |
| **foreach speedup** | **1.02×** | **1.11×** |
| CASMO vs AdamW(foreach) | 2.28× | 2.25× |

So on available hardware foreach is worth 2–11%, and CASMO's ~2.2× gap is the per-coordinate
gate itself, not launch overhead — foreach would recover roughly a tenth of it. On CUDA the
overhead is known to be materially larger, so the win is plausibly bigger, **but there is no
CUDA here to measure it**. Landing a substantial rewrite whose benefit cannot be verified
would violate this project's own rule. Recorded as measured headroom in `CLAUDE.md`, to be
implemented and validated when a CUDA device is available.

### 7.5 MPS — already supported

`casmo.py` contains no device-specific code. Verified training end-to-end on MPS;
`tests/test_device_and_groups.py` now exercises whatever accelerator is present.

## 8. Is this the best design? Three suspicions, tested

Prompted by the question "is this the best we can do", three specific doubts about the
shipped v0.4 were tested rather than argued. Two were founded; one was not.

### 8.1 Is `focus` earning its place? — barely

`robustness=0` (pure focus, no brake) measured equal to or *worse* than AdamW on every
axis, which suggested focus might be dead weight. Ablating it directly (5 seeds):

| | clean steps | label-noise 30% | grad-noise σ=0.5 |
|---|---|---|---|
| AdamW | 82 | 0.679 | 0.636 |
| ρ=1, focus **off** | 292 | 0.785 | 0.399 |
| ρ=1, focus **on** (shipped) | 318 | **0.797** | 0.396 |
| ρ=0.5, focus **off** | 150 | 0.688 | 0.498 |
| ρ=0.5, focus **on** (shipped) | 160 | **0.704** | 0.488 |

So focus is **not** dead weight — it is worth **+1.2 to +1.6 points** of label-noise
accuracy — but it costs ~7–9% clean-data speed and is slightly *negative* on gradient
noise. Put in proportion: against AdamW's 0.679, `trust` alone delivers +10.6 points and
focus adds +1.2 on top. **`trust` is ~90% of the value; focus is ~10%** for one extra
hyper-parameter (`rel_floor`) and an extra elementwise op.

This is honest over-engineering. A `trust`-only CASMO would be simpler and get ~98% of the
benefit. Focus is kept because the gain is real and consistent at both ρ values, but it is
the first thing to cut if the design needs simplifying.

### 8.2 Is the noise estimator biased? — yes, and it leaked into `robustness`

`s` was computed as `EMA[(g - m_t)²]` with `m_t` already containing `g`. Algebraically
`g - m_t = β₁(g - m_{t-1})`, so `s ≈ β₁²·Var[g]`. Verified against a known distribution
(μ=1, σ=2, 20k samples): measured ratio to true variance **0.791** (algebra predicts 0.81);
centering on `m_{t-1}` instead gives **0.977**.

The consequence was worse than a constant offset. Because the shrink factor *is* β₁², the
strength of `robustness` silently depended on `betas` — at 30% label noise with ρ=1 fixed:

| β₁ | biased (shipped) | corrected |
|---|---|---|
| 0.9 | 0.792 | 0.808 |
| 0.7 | 0.737 | 0.793 |
| 0.5 | 0.707 | 0.794 |
| **spread** | **0.085** | **0.016** |

A user tuning momentum would have quietly lost up to 8.5 points of noise robustness with no
warning. **Fixed** by reordering two lines (free). It also improves the headline at the
default β₁: 30% label noise **0.797 → 0.810**, 15% noise **0.851 → 0.857**.

### 8.3 Should `trust` be global rather than per-tensor? — no meaningful difference

| | clean steps | label-noise 30% | grad-noise |
|---|---|---|---|
| ρ=1 per-tensor (shipped) | 318 | 0.797 | 0.396 |
| ρ=1 global | 342 | 0.805 | 0.395 |
| ρ=0.5 per-tensor | 160 | 0.704 | 0.488 |
| ρ=0.5 global | 170 | 0.703 | 0.487 |

Within noise, and global costs a cross-tensor reduction that would complicate any future
`foreach` port. Per-tensor kept.

## 9. What is still unknown

Stated plainly, so nobody mistakes this evidence log for a proof of optimality:

1. **No real dataset has been run against v0.4.** Every number here is a synthetic MLP
   study (plus a 7-config generalisation sweep). The `benchmarks/b1..b5` results are all
   v0.3 historical. This is the single biggest gap.
2. **`β₂` for the variance is inherited from Adam (0.999) without examination.** There is no
   reason a *variance* estimator wants the same horizon as Adam's second moment.
3. **The map from AGAR to confidence is a guess.** `trust = c_min + (1-c_min)·ā` is linear
   and `focus` is a clipped ratio; neither was compared against alternatives.
4. **The ρ=0.5 default is a bet**, not an optimum: it is measurably *worse than Adam* under
   isotropic gradient noise (0.472 vs 0.641) in exchange for +3 points on label noise. It is
   the right default only if user data is more often label-noisy than DP-noisy.
5. **`foreach`/CUDA-graph headroom is unmeasured on CUDA** (see §7.4).

## 10. What is probably irreducible

One limit looks fundamental rather than unexplored. `robustness` trades "fit the training
set" against "refuse to fit the training set". Label noise wants the latter; DP-SGD under a
fixed step budget wants the former. **Whether fitting the training data is desirable is a
property of the task, not of the gradients** — so an optimizer that sees only gradients
cannot infer it, and v0.3's attempt to infer it from the run's own early distribution is
precisely why it failed (§1). Auto-tuning ρ would require a signal from outside the
optimizer's contract, such as validation loss. Until that contract changes, ρ stays a knob.
