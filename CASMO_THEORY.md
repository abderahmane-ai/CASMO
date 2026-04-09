# CASMO: Theoretical Foundation and Mathematical Formulation

**CASMO** (Confident Adaptive Selective Momentum Optimizer) is a novel optimization algorithm designed to improve training stability and generalization by dynamically assessing the quality of gradient updates. Unlike standard adaptive optimizers (Adam, RMSprop) that adapt to gradient *magnitude*, CASMO adapts to gradient *consistency* — the signal-to-noise ratio of the update direction.

---

## 1. Core Philosophy: Signal vs. Noise

In stochastic optimization, the gradient $g_t$ at step $t$ is a noisy estimate of the true gradient:

$$g_t = \nabla \mathcal{L}(\theta_t) + \epsilon_t$$

where $\nabla \mathcal{L}(\theta_t)$ is the true gradient (signal) and $\epsilon_t$ is stochastic noise.

Standard optimizers like Adam normalize by the second moment $\sqrt{\mathbb{E}[g^2]}$. However, a large $\mathbb{E}[g^2]$ can come from either a strong signal or high noise. Adam cannot distinguish between:

- **High Signal**: Consistent large gradients pointing in the same direction.
- **High Noise**: Large gradients oscillating randomly across steps.

CASMO introduces the **Adaptive Gradient Alignment Ratio (AGAR)** to explicitly measure this distinction.

---

## 2. Mathematical Formulation

### 2.1 Moment Estimation

CASMO tracks exponential moving averages of the first and second gradient moments, identical to Adam:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \approx \mathbb{E}[g]$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 \approx \mathbb{E}[g^2]$$

Importantly, AGAR is computed from the **raw (uncorrected) moments**. Applying bias correction would distort the variance relationship used in the next step.

---

### 2.2 AGAR: Adaptive Gradient Alignment Ratio

Using the variance identity $\text{Var}[X] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$, we decompose gradient power into signal and noise:

- **Signal Power**: $S_t = m_t^2 \approx (\mathbb{E}[g])^2$
- **Noise Power**: $N_t = v_t - m_t^2 \approx \text{Var}[g]$

AGAR is defined as the ratio of signal power to total power:

$$\text{AGAR}_t = \frac{S_t}{S_t + N_t} = \frac{m_t^2}{m_t^2 + (v_t - m_t^2)} = \frac{m_t^2}{v_t}$$

| Value | Interpretation |
|---|---|
| $\text{AGAR} \approx 1$ | Pure signal — gradients are perfectly consistent across steps |
| $\text{AGAR} \approx 0$ | Pure noise — gradients are random and directionless |

In practice, AGAR is computed per-element and then averaged across all elements in a parameter group (in `'group'` mode) or per individual parameter (in `'parameter'` mode), producing a robust scalar confidence signal.

---

### 2.3 Automatic Calibration

A fixed AGAR threshold would behave inconsistently across architectures (CNNs vs. Transformers), batch sizes, and tasks — all of which have different natural signal-to-noise baselines.

Instead, CASMO **self-calibrates** during the initial $T_{\text{init}}$ steps. It collects a buffer of AGAR values and computes:

- $\mu$: mean of the AGAR distribution
- $\sigma$: standard deviation of the AGAR distribution
- $\tau$: median of the distribution (used as the sigmoid center — robust to outliers)

The median is used over the mean because it is less sensitive to early-training spikes.

#### Principled Selection of $T_{\text{init}}$

AGAR is computed from $m_t$ and $v_t$, which are only reliable estimates once the EMA has warmed up. The effective memory window of the first moment is $1/(1-\beta_1)$ steps (e.g. 10 steps for $\beta_1 = 0.9$). Calibrating before the moments have warmed up means the AGAR buffer reflects initialization artifacts, not real training dynamics.

$T_{\text{init}}$ is therefore derived from $\beta_1$ rather than total run length:

$$T_{\text{init}} = \max\!\left(500,\ \frac{50}{1 - \beta_1}\right)$$

The factor of 50 gives approximately $5\times$ the EMA window as a statistical safety margin — enough samples for the median and percentiles to stabilize. If `total_steps` is provided, $T_{\text{init}}$ is additionally capped at 20% of the total run to avoid spending an unreasonable fraction of training in calibration:

$$T_{\text{init}} = \min\!\left(T_{\text{init}},\ \lfloor\text{total\_steps} / 5\rfloor\right)$$

In all cases $T_{\text{init}} \geq 50$ is enforced as a hard floor.

Additionally, CASMO derives an adaptive $c_{\min}$ from the **coefficient of variation** $\text{CV} = \sigma / \mu$:

| CV | Regime | $c_{\min}$ |
|---|---|---|
| $\text{CV} > 0.5$ | Bimodal (mixed signal/noise) | $0.1$ — strong discrimination |
| $0.3 < \text{CV} \leq 0.5$ | Some separation | $0.3$ — moderate discrimination |
| $\text{CV} \leq 0.3$ | Unimodal (pervasive noise) | $0.5$ — prevent over-suppression |

Once calibrated, $\tau$, $\mu$, $\sigma$, and $c_{\min}$ are **frozen for the rest of training**.

---

### 2.4 Confidence Mapping

CASMO translates the raw AGAR value into a **Confidence Score** $C_t \in [c_{\min},\, 1.0]$, which acts as a multiplicative learning rate scaler.

The mapping uses a sigmoid centered on the calibrated distribution:

$$z_t = \frac{\text{AGAR}_t - \mu}{\sigma}$$

$$C_t = c_{\min} + (1 - c_{\min}) \cdot \sigma(z_t), \quad \sigma(z) = \frac{1}{1 + e^{-z}}$$

**Behavior:**

- $\text{AGAR}_t \gg \mu$ ($z \gg 0$): $C_t \to 1.0$ — full learning rate, high-confidence update.
- $\text{AGAR}_t \approx \mu$ ($z \approx 0$): $C_t \approx c_{\min} + \frac{1 - c_{\min}}{2}$ — neutral update.
- $\text{AGAR}_t \ll \mu$ ($z \ll 0$): $C_t \to c_{\min}$ — suppressed learning rate, noisy update.

This sigmoid parameterization naturally adapts to any noise regime. In a clean data setting ($\mu$ high, $\sigma$ low), most samples receive high confidence. In a pervasive noise setting ($\mu$ low, $\sigma$ low), confidence scales smoothly from $c_{\min}$ without collapsing to zero.

---

### 2.5 Parameter Update Rule

CASMO applies the AdamW update direction scaled by the confidence score:

$$\theta_{t+1} = \theta_t(1 - \eta\lambda) - \eta \cdot C_t \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

where:

- $\eta$: base learning rate
- $\lambda$: weight decay coefficient (decoupled, AdamW-style)
- $\hat{m}_t = m_t / (1 - \beta_1^t)$, $\hat{v}_t = v_t / (1 - \beta_2^t)$: bias-corrected moments
- $C_t \in [c_{\min}, 1.0]$: confidence score from Section 2.4

Bias correction is applied **only at the update step**, not during AGAR computation, preserving the variance relationship $\text{Var}[g] = \mathbb{E}[g^2] - (\mathbb{E}[g])^2$.

---

## 3. Workflow Summary

| Step | Operation |
|---|---|
| 1 | Compute gradients $g_t$ via backpropagation |
| 2 | Update raw moments $m_t$, $v_t$ |
| 3 | Compute AGAR $= m_t^2 / v_t$ (per-element, then averaged) |
| 4 | If $t \leq T_{\text{init}}$: accumulate AGAR buffer |
| 5 | If $t = T_{\text{init}}$: calibrate $\mu$, $\sigma$, $\tau$, $c_{\min}$ — then freeze |
| 6 | Map AGAR to confidence $C_t$ via sigmoid |
| 7 | Apply bias-corrected AdamW update scaled by $C_t$ |

---

## 4. Why It Works

### 4.1 Filtering Memorization

In tasks with label noise, memorization gradients are inconsistent — they depend on specific random samples, so $\text{Var}[g]$ is high and AGAR is low. Generalization gradients align across many samples, so AGAR is high. CASMO naturally suppresses the memorization direction without requiring explicit regularization.

### 4.2 Differential Privacy

In DP-SGD, Gaussian noise is added to gradients. Standard optimizers treat this noise as valid gradient magnitude. CASMO detects that the added noise is directionally random (low AGAR) and automatically reduces the effective learning rate, preventing divergence from noise-chasing.

### 4.3 Long-Tail Learning

For rare classes with few samples per batch, gradients are high-variance due to insufficient averaging. CASMO detects this and scales down the update for the affected parameters or groups, preventing tail-class noise from corrupting the shared representation.