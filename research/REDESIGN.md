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
| **0.5** (default) | 160 | 0.701 | 0.793 | 0.488 | **0.932 / 36 steps** |
| 1.0 | 322 | **0.797** | **0.851** | 0.396 | 0.932 / 50 steps |

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
signature of the mechanism: at 50% corruption the gap reaches +15.6 points.

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
| label noise 30% | 0.675 | 0.675 | **0.797** |
| grad noise σ=0.5 | **0.641** | 0.623 | 0.396 |

This is a real trade-off, not a bug, and is the reason `robustness` is a first-class
parameter. Use low ρ for DP-SGD, high ρ for label noise. The B3 (DP-SGD) benchmark is
configured with ρ=0.5 accordingly.

**Rule of thumb:** high ρ pays off when the model *would otherwise memorise noise*. It
costs you when the model is still under-fitting — whether because the noise is isotropic,
the step budget is short, or the learning rate is too low to converge.
