# B5: GAN Stability - DCGAN on Tiny ImageNet

## Overview

This benchmark tests CASMO's ability to stabilize adversarial training, where high-variance gradients from the generator-discriminator game often lead to training instability.

## Task Description

- **Dataset**: Tiny ImageNet (64×64, 20k subset)
- **Architecture**: DCGAN (Deep Convolutional GAN)
- **Configurations**: All combinations of CASMO/Adam for Generator and Discriminator
  - `adam_adam`: Baseline
  - `casmo_adam`: CASMO on Generator only
  - `adam_casmo`: CASMO on Discriminator only  
  - `casmo_casmo`: CASMO on both networks
- **Metrics**: Generator loss, Discriminator loss, Gradient variance, Sample quality

## Why This Benchmark?

**GANs are notoriously difficult to train** due to the adversarial nature of the optimization:
- The Generator and Discriminator are locked in a minimax game
- Training is inherently unstable (mode collapse, oscillation)
- Gradient quality varies wildly during training
- Requires careful hyperparameter tuning

### The Challenge

GAN training suffers from several pathologies:

1. **Mode Collapse**: Generator produces limited diversity
2. **Vanishing Gradients**: Discriminator becomes too strong
3. **Oscillation**: Networks fail to converge to equilibrium
4. **High Variance**: Adversarial gradients are inherently noisy

These issues arise from the adversarial gradient dynamics:
- **Generator**: Receives gradients through the Discriminator (high variance)
- **Discriminator**: Receives gradients from real vs fake classification (unstable)

### How CASMO Addresses This

CASMO's gradient quality detection can help stabilize GAN training:

**When CASMO is on the Generator:**
- Detects when Discriminator feedback is unreliable
- Reduces learning rate when gradients are misleading
- Prevents overreaction to noisy discriminator signals

**When CASMO is on the Discriminator:**
- Prevents overpowering the Generator
- Maintains balanced adversarial dynamics
- Avoids overfitting to current Generator samples

**When CASMO is on both:**
- Most stable configuration (hypothesis)
- Both networks adapt to gradient quality
- Better equilibrium between G and D

## Hypothesis

- **adam_adam**: Baseline, expected instability
- **casmo_adam**: Improved Generator stability under noisy D feedback
- **adam_casmo**: Better D-G balance, prevents D from dominating
- **casmo_casmo**: Most stable training, best sample quality

## Technical Details

### DCGAN Architecture
Standard DCGAN for 64×64 images:
- **Generator**: 5-layer transposed convolution (z → 64×64 RGB)
- **Discriminator**: 5-layer convolution (64×64 RGB → scalar)
- **Latent dimension**: 100
- **Batch size**: 16 (optimized for 6GB VRAM)

### Training Protocol
- **Loss**: Binary Cross-Entropy (non-saturating GAN)
- **Label smoothing**: 0.9 for real images (prevents D overconfidence)
- **Learning rate**: 0.0002 for both G and D
- **Betas**: (0.5, 0.999) - matches DCGAN paper

### Stability Metrics

1. **Loss Tracking**: Monitor G and D losses over time
2. **Gradient Variance**: Measure consistency of updates
3. **AGAR Evolution**: Track CASMO's confidence adaptation
4. **Sample Quality**: Visual inspection of generated images
5. **D(x) and D(G(z))**: Discriminator scores on real/fake images

### Dataset Choice: Tiny ImageNet

- **Native resolution**: 64×64 (perfect for DCGAN)
- **Download size**: ~237MB (faster than LSUN)
- **Diversity**: 200 classes, 100k training images
- **Subset**: 20k images for faster experimentation

## Expected Outcome

CASMO should demonstrate:

1. **Reduced training instability** (smoother loss curves)
2. **Lower gradient variance** (more consistent updates)
3. **Better Generator-Discriminator balance** (stable D(x) and D(G(z)))
4. **Higher quality samples** (visual inspection)
5. **Faster convergence** to stable equilibrium

The `casmo_casmo` configuration is expected to show the best overall stability and sample quality.
