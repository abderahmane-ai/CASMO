# CASMO: Theoretical Foundation and Mathematical Formulation

**CASMO** (Confident Adaptive Selective Momentum Optimizer) is a novel optimization algorithm designed to improve training stability and generalization by dynamically assessing the "quality" of gradient updates. Unlike standard adaptive optimizers (like Adam or RMSprop) that only adapt to gradient *magnitude*, CASMO adapts to gradient *consistency* (signal-to-noise ratio).

## 1. Core Philosophy: Signal vs. Noise

In stochastic optimization, the gradient $g_t$ at step $t$ is a noisy estimate of the true gradient. We can decompose it into:

$$ g_t = \nabla \mathcal{L}(\theta_t) + \epsilon_t $$

Where:
*   $\nabla \mathcal{L}(\theta_t)$ is the true gradient (Signal).
*   $\epsilon_t$ is the stochastic noise (Noise).

Standard optimizers like Adam normalize by the second moment $\sqrt{\mathbb{E}[g^2]}$. However, a large $\mathbb{E}[g^2]$ can come from either a strong signal OR high noise. Adam cannot distinguish between:
1.  **High Signal**: Consistent large gradients (Good direction).
2.  **High Noise**: Large gradients oscillating wildly (Bad direction).

CASMO introduces the **Adaptive Gradient Alignment Ratio (AGAR)** to explicitly distinguish these two cases.

---

## 2. Mathematical Formulation

### 2.1 Moment Estimation
CASMO tracks the exponential moving averages (EMA) of the first and second moments, similar to Adam:

$$ m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad (\text{Estimate of } \mathbb{E}[g]) $$
$$ v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad (\text{Estimate of } \mathbb{E}[g^2]) $$

### 2.2 AGAR: Adaptive Gradient Alignment Ratio
We derive the signal and noise components from the moments. Using the variance identity $\text{Var}[X] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$:

*   **Signal Power**: $S_t = (\mathbb{E}[g])^2 \approx m_t^2$
*   **Noise Power**: $N_t = \text{Var}[g] \approx v_t - m_t^2$

The **AGAR** metric is defined as the ratio of signal power to total power:

$$ \text{AGAR}_t = \frac{\text{Signal}}{\text{Signal} + \text{Noise}} = \frac{m_t^2}{m_t^2 + (v_t - m_t^2)} = \frac{m_t^2}{v_t} $$

*   **AGAR $\approx$ 1**: Pure Signal. Gradients are perfectly consistent.
*   **AGAR $\approx$ 0**: Pure Noise. Gradients are random/orthogonal.

*Note: In implementation, we compute AGAR per-element and then average it (either per-group or per-parameter) to get a robust scalar metric.*

### 2.3 Automatic Calibration
Instead of using a fixed threshold for AGAR, CASMO self-calibrates during the initial training phase (e.g., first 500 steps). It collects a buffer of AGAR values and computes distribution statistics:

*   Mean: $\mu_{\text{agar}}$
*   Standard Deviation: $\sigma_{\text{agar}}$

This allows CASMO to adapt to different architectures (CNNs vs Transformers) and batch sizes, which naturally have different baseline signal-to-noise ratios.

### 2.4 Confidence Mapping
CASMO translates the raw AGAR value into a **Confidence Score** $C_t \in [c_{\min}, 1.0]$. This acts as a dynamic learning rate scaler.

We use a sigmoid mapping based on the Z-score of the current AGAR value relative to the calibrated distribution:

$$ z_t = \frac{\text{AGAR}_t - \mu_{\text{agar}}}{\sigma_{\text{agar}}} $$

$$ C_t = c_{\min} + (1 - c_{\min}) \cdot \sigma(z_t) $$

Where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid function.

*   **High AGAR ($z > 0$)**: $C_t \to 1.0$ (Full learning rate).
*   **Low AGAR ($z < 0$)**: $C_t \to c_{\min}$ (Suppressed learning rate).

### 2.5 DDEA: Drift-Detecting EMA Adapter
While the initial calibration sets a baseline, training dynamics change over time. A fixed threshold might become stale. However, simply using an Exponential Moving Average (EMA) to track AGAR is dangerous: it can be dragged by **memorization** (overfitting to specific samples, causing high AGAR) or **noise spikes**.

CASMO employs a **Drift-Detecting EMA Adapter (DDEA)** to robustly adapt the $\tau$ threshold (used as the center of the sigmoid mapping).

#### Mechanisms:
1.  **Variance Tracking**: DDEA tracks the variance of the AGAR signal itself.
    *   High Variance $\to$ Unstable dynamics $\to$ Increase adaptation rate (Fast reaction).
    *   Low Variance $\to$ Stable dynamics $\to$ Decrease adaptation rate (Stability).
2.  **Dead Zone**: Small deviations in AGAR are ignored (treated as noise). The threshold only updates if the drift exceeds a percentage (e.g., 20%) of the current value. This prevents "chasing the noise."
3.  **Memorization Guard**: If AGAR rises suspiciously high (e.g., > 1.2x calibrated baseline), DDEA freezes adaptation. This prevents the optimizer from validating overfitting as "high confidence signal."

This ensures CASMO remains responsive to genuine distributional shifts (like curriculum learning changes) while resisting the instability of non-stationary noise.

### 2.6 Parameter Update Rule
The final update rule combines the AdamW update direction with the CASMO confidence scalar:

$$ \theta_{t+1} = \theta_t - \eta \cdot \underbrace{C_t}_{\text{Confidence}} \cdot \underbrace{\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}}_{\text{Adam Direction}} - \eta \lambda \theta_t $$

Where:
*   $\eta$: Base learning rate.
*   $\hat{m}_t, \hat{v}_t$: Bias-corrected moments.
*   $\lambda$: Weight decay.

---

## 3. Workflow Summary

1.  **Compute Gradients**: Obtain $g_t$ from backpropagation.
2.  **Update Moments**: Update $m_t$ and $v_t$.
3.  **Compute AGAR**: Calculate signal consistency $m_t^2 / v_t$.
4.  **Calibrate (if needed)**: If in warmup, update $\mu, \sigma$ statistics.
5.  **Compute Confidence**: Map AGAR to $C_t$ using sigmoid.
6.  **Apply Update**: Scale step size by $C_t$.

## 4. Why It Works (Theoretical Intuition)

### 4.1 The "Grokking" Effect
In tasks with label noise or complex patterns, "memorization" gradients tend to be inconsistent (high variance, low AGAR) because they depend on specific random samples. "Generalization" gradients tend to be consistent (low variance, high AGAR) because they align across many samples.

By suppressing updates with low AGAR, CASMO effectively **filters out memorization** while allowing generalization.

### 4.2 Differential Privacy
In DP-SGD, Gaussian noise is added to gradients. Standard optimizers treat this noise as valid magnitude. CASMO detects that the noise is orthogonal/random (low AGAR) and automatically lowers the learning rate, preventing the model from "chasing the noise" and diverging.

### 4.3 Long-Tail Learning
For rare classes, gradients are noisy due to small batch sizes. CASMO detects this high variance and scales down updates for those specific parameters (or groups), preventing the "overfitting to noise" that typically destroys tail performance.
