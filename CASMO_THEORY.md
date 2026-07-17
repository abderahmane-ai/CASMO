# CASMO: Theoretical Foundation

**CASMO** (Confidence-Adjusted Signal-to-noise Momentum Optimizer) adapts to gradient *consistency* rather than gradient *magnitude*. Where Adam asks "how big is this gradient?", CASMO asks "how much of this gradient is signal?" — and scales each coordinate's step accordingly.

Every claim in this document is backed by the reproducible experiment in [`research/validate_redesign.py`](research/validate_redesign.py); the measured results and the design decisions they forced are recorded in [`research/REDESIGN.md`](research/REDESIGN.md).

---

## 1. Core Philosophy: Signal vs. Noise

In stochastic optimization the observed gradient is a noisy estimate of the true one:

$$g_t = \nabla \mathcal{L}(\theta_t) + \epsilon_t$$

Adam normalizes by $\sqrt{\mathbb{E}[g^2]}$. But a large $\mathbb{E}[g^2]$ can mean either a strong, consistent signal *or* large random fluctuations — Adam cannot tell them apart. CASMO separates them explicitly.

---

## 2. Mathematical Formulation

### 2.1 Moment Estimation

CASMO tracks two EMAs per coordinate — the same optimizer-state footprint as Adam:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \;\approx\; \mathbb{E}[g]$$

$$s_t = \beta_2 s_{t-1} + (1 - \beta_2)\,(g_t - m_t)^2 \;\approx\; \mathrm{Var}[g]$$

$s_t$ is a **centered** (belief-style) variance. This matters: the earlier design estimated noise as $v_t - m_t^2$ using Adam's uncentered second moment, which mixes two different EMA horizons ($1/(1-\beta_1) \approx 10$ steps vs. $1/(1-\beta_2) \approx 1000$ steps) and is therefore a biased SNR estimate. Measuring the deviation $(g_t - m_t)^2$ directly avoids that timescale mismatch.

Bias correction is applied at use time: $\hat{m}_t = m_t/(1-\beta_1^t)$, $\hat{s}_t = s_t/(1-\beta_2^t)$.

### 2.2 AGAR: Adaptive Gradient Alignment Ratio

Decompose gradient power into signal and noise, **per coordinate** $i$:

- **Signal power**: $S_i = \hat{m}_i^2 \approx (\mathbb{E}[g_i])^2$
- **Noise power**: $N_i = \hat{s}_i \approx \mathrm{Var}[g_i]$

$$\mathrm{AGAR}_i = \frac{S_i}{S_i + N_i} \in [0, 1]$$

Note there is no $\epsilon$ in this denominator. Since $S_i \le S_i + N_i$ by construction, a
zero denominator implies a zero numerator, so the ratio is simply clamped to 0 there. Adding
an $\epsilon$ would make AGAR depend on the *absolute scale* of the gradient rather than on the
ratio it is defined as: with $\epsilon = 10^{-8}$, a coordinate with a perfect signal-to-noise
ratio of 100 reads $\mathrm{AGAR} \approx 0.99$ at gradient scale $1$ but $\approx 0.01$ at scale
$10^{-5}$, purely because the gradient got small.

| Value | Interpretation |
|---|---|
| $\mathrm{AGAR}_i \approx 1$ | Pure signal — this coordinate's gradient is consistent across steps |
| $\mathrm{AGAR}_i \approx 0$ | Pure noise — this coordinate's gradient is directionless |

AGAR is bounded in $[0,1]$ by construction (both $S_i$ and $N_i$ are non-negative), so it needs **no clamping, no calibration, and no tunable threshold**. It is an *absolute* scale: 0 always means noise and 1 always means signal, regardless of architecture, batch size, or task.

> **Why absolute matters.** The pre-0.4 design calibrated a threshold $\tau$ against the AGAR distribution observed in the first ~500 steps, then froze it. Under label noise the noise is present *from step one*, so the calibration baseline is itself noisy — and the optimizer concludes that its noisy gradients are "normal". Measurement confirmed this failure: the calibrated design scored 0.680 test accuracy at 30% label noise versus AdamW's 0.679, i.e. it delivered essentially none of the robustness it existed to provide. An absolute reference is *required* for the label-noise use case.

### 2.3 The Confidence Map

CASMO turns per-coordinate AGAR into a confidence $C_i$ along two complementary axes. Let $\bar{a} = \mathrm{mean}_i(\mathrm{AGAR}_i)$ over the parameter tensor.

**Absolute axis — trust** (how fast should this tensor move at all?):

$$\text{trust} = c_{\min} + (1 - c_{\min})\cdot \bar{a}$$

When a tensor's gradients are noise-dominated, $\bar{a}$ is small, trust is small, and the whole tensor slows down. This is a self-adjusting learning-rate reduction, and it is the mechanism that resists memorizing label noise.

**Relative axis — focus** (which coordinates deserve the step?):

$$\text{focus}_i = \mathrm{clip}\!\left(\frac{\mathrm{AGAR}_i}{\bar{a} + \epsilon},\; r_{\text{floor}},\; 1\right)$$

Focus is mean-normalized, so when all coordinates are equally reliable it is $\approx 1$ everywhere and costs nothing. When reliability is uneven it down-weights the noise-dominated coordinates *relative to their peers*. It is capped at 1 — never amplifying — which preserves stability at aggressive learning rates.

**Combined:**

$$C_i = \text{trust}^{\,\rho} \cdot \text{focus}_i, \qquad \rho \in [0,1]$$

The exponent $\rho$ (`robustness`) dials the absolute axis:

- $\rho = 0$: $\text{trust}^0 = 1$ — pure relative reweighting, AdamW-like pace.
- $\rho = 1$: full absolute suppression — maximum noise robustness.

### 2.4 Parameter Update Rule

$$\theta_{t+1} = \theta_t(1 - \eta\lambda) \;-\; \eta \cdot C_i \cdot \frac{\hat{m}_i}{\sqrt{S_i + N_i} + \epsilon}$$

Since $\mathbb{E}[g^2] = (\mathbb{E}[g])^2 + \mathrm{Var}[g]$, the quantity $S_i + N_i \approx \hat{v}_i$ — Adam's bias-corrected second moment. **The Adam denominator is reconstructed from the signal and noise terms already computed**, so no third EMA is needed and CASMO's state footprint equals Adam's. Measurement confirmed this reconstruction is behaviourally equivalent to tracking $v$ separately.

With $\rho = 0$ and $r_{\text{floor}} = 1$, $C_i \equiv 1$ and the rule reduces exactly to AdamW.

---

## 3. Workflow Summary

| Step | Operation |
|---|---|
| 1 | Compute gradients $g_t$ |
| 2 | Update $m_t$ (signal) and $s_t$ (centered variance / noise) |
| 3 | $S = \hat{m}^2$, $N = \hat{s}$, $\mathrm{AGAR} = S/(S+N)$ per coordinate |
| 4 | $\text{trust} = c_{\min} + (1-c_{\min})\overline{\mathrm{AGAR}}$ |
| 5 | $\text{focus}_i = \mathrm{clip}(\mathrm{AGAR}_i/\overline{\mathrm{AGAR}},\, r_{\text{floor}},\, 1)$ |
| 6 | $C_i = \text{trust}^{\rho}\cdot\text{focus}_i$ |
| 7 | AdamW step scaled by $C_i$, denominator $\sqrt{S+N}$ |

There is no calibration phase, no frozen threshold, and no warm-up.

The step itself performs no host-device synchronization: the reported AGAR and confidence are
accumulated as device tensors and only materialized when `group_metrics()` is called, so a run
that logs them every 100 steps pays for 1 sync per 100 steps rather than 1 per step. (Removing
an earlier per-step `.item()` from this path measured 1.19× faster on a 40-layer model on MPS.)

---

## 4. Why It Works — and Where It Doesn't

### 4.1 Filtering Memorization (validated)

Memorization gradients depend on individual mislabeled samples, so they are inconsistent across batches: $\mathrm{Var}[g]$ is high and AGAR is low. Generalizing gradients align across many samples, so AGAR is high. Trust therefore shrinks precisely when the model starts fitting noise.

Measured at 30% label noise (5 seeds): CASMO ($\rho=1$) reaches **0.797** test accuracy versus AdamW's **0.679**, while refusing to memorize the training set (0.850 train accuracy vs. AdamW's 1.000). At 15% noise: **0.851** vs. **0.763**.

### 4.2 Stability at Aggressive Learning Rates (validated)

Because focus never amplifies and trust contracts when the signal degrades, CASMO tolerates larger steps. At $\eta = 3\times10^{-2}$ CASMO ($\rho=0.5$) reaches the loss threshold in **36 steps** versus Adam's **70**, with better final accuracy (0.932 vs. 0.921). Where a larger stable learning rate is available, confidence gating is a *speed* win, not a tax.

### 4.3 Long-Tail Learning

Rare classes contribute few samples per batch, so their gradients are high-variance and low-AGAR. Focus down-weights those coordinates relative to well-estimated ones, limiting the damage tail noise does to the shared representation.

### 4.4 Honest Limitation: Isotropic Gradient Noise

When noise is injected uniformly into *every* coordinate (as in DP-SGD), the absolute axis is counter-productive: it slows training exactly when the fixed step budget can least afford it. Measured at $\sigma = 0.5$: Adam 0.641 test accuracy, CASMO $\rho=0$ 0.623, CASMO $\rho=1$ **0.396**.

Label noise and isotropic gradient noise genuinely want opposite policies — the former rewards slowing down, the latter punishes it. This is why `robustness` is exposed rather than hard-coded. **For DP-SGD-style workloads, use a low `robustness`** (0 – 0.5); for label noise, use a high one.

---

## 5. Relationship to Prior Work

- **Adam / AdamW** — CASMO reduces to AdamW at $\rho = 0,\, r_{\text{floor}} = 1$.
- **AdaBelief** — shares the centered variance $s_t = \mathrm{EMA}[(g-m)^2]$, but uses it as the *denominator*. CASMO keeps Adam's denominator (reconstructed as $\sqrt{m^2+s}$) and uses $s$ to form an explicit, bounded confidence signal instead.
- **Trust-ratio methods (LARS/LAMB)** — also multiply the update by a trust factor, but derive it from weight/gradient *norms* per layer. CASMO derives it from *signal-to-noise* per coordinate.
