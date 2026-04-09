# B4: Differential Privacy - DP-SGD on CIFAR-10

## Overview

This benchmark tests CASMO's performance under rigorous differential privacy constraints, where calibrated Gaussian noise is added to gradients to protect individual training samples.

## Task Description

- **Dataset**: CIFAR-10
- **Architecture**: ResNet-18 (adapted for CIFAR-10)
- **Privacy Framework**: Opacus (PyTorch DP library)
- **Privacy Budget**: ε (epsilon) tracking with δ=1e-5
- **Noise Mechanism**: Per-sample gradient clipping + Gaussian noise
- **Baselines**: DP-SGD, DP-AdamW
- **Metric**: Accuracy vs Privacy Budget (ε)

## Why This Benchmark?

**Differential Privacy (DP)** is the gold standard for privacy-preserving machine learning, but it comes with a significant challenge: added noise degrades model performance.

### The Challenge

DP-SGD works by:
1. **Clipping** each sample's gradient to a fixed norm (e.g., C=1.0)
2. **Adding** Gaussian noise proportional to the clip norm
3. **Tracking** the cumulative privacy budget (ε)

This creates a fundamental tension:
- **More noise** → Better privacy (lower ε) → Worse accuracy
- **Less noise** → Worse privacy (higher ε) → Better accuracy

### The Gradient Quality Problem

DP noise is _intentionally added_ to gradients, making them:
- High variance even for clean, informative samples
- Difficult to distinguish from naturally noisy gradients
- Challenging for adaptive optimizers

Standard adaptive methods (like Adam) struggle because:
- They rely on gradient statistics that are now heavily corrupted
- Momentum and variance estimates become unreliable
- Learning rate adaptation may be counterproductive

### How CASMO Addresses This

CASMO's AGAR metric is designed to measure gradient _alignment quality_, not just magnitude:
- **True signal gradients** (even with DP noise): Still align with the model's learned patterns → Reasonable AGAR
- **Pure noise**: Random directions → Low AGAR

This allows CASMO to potentially:
1. Distinguish signal from noise even under DP
2. Adapt learning rates based on alignment rather than noisy statistics
3. Achieve better accuracy for the same privacy budget

## Hypothesis

For the same privacy budget (ε):
- **DP-SGD**: Baseline performance, no adaptation
- **DP-AdamW**: May struggle due to corrupted second-moment estimates
- **CASMO**: Should achieve better accuracy by detecting gradient alignment despite noise

## Technical Details

### Privacy Accounting
Opacus tracks privacy using the Rényi Differential Privacy (RDP) accountant:
- **Target δ**: 1e-5 (probability of privacy breach)
- **Reported ε**: Privacy budget consumed
- **Lower ε** = Better privacy (but typically worse accuracy)

### Noise Calibration
- **Clip norm (C)**: 1.0 (maximum gradient norm per sample)
- **Noise multiplier (σ)**: 1.0 (scales the Gaussian noise)
- **Effective noise**: N(0, (σ·C)²)

### Model Modifications
Opacus requires:
- BatchNorm → GroupNorm (for per-sample gradient computation)
- Specialized DataLoader for privacy accounting
- Modified backward pass for per-sample clipping

## Expected Outcome

CASMO should demonstrate:
1. **Better accuracy** at the same privacy budget (ε)
2. **Faster convergence** under DP constraints
3. **More stable training** despite gradient noise
4. **Efficient privacy-utility tradeoff** compared to baselines

This benchmark validates whether CASMO's gradient quality assessment remains effective even when noise is deliberately added for privacy.
