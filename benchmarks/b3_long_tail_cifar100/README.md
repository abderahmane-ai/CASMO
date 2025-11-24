# B3: Long-Tail Recognition - CIFAR-100

## Overview

This benchmark tests CASMO's ability to handle class imbalance, where minority classes provide sparse and noisy gradients that standard optimizers tend to ignore in favor of majority classes.

## Task Description

- **Dataset**: CIFAR-100 (100 fine-grained classes)
- **Architecture**: ResNet-32
- **Imbalance Ratio**: 100:1 (exponential long-tail distribution)
- **Head Classes**: >100 samples (many-shot)
- **Medium Classes**: 20-100 samples (medium-shot)
- **Tail Classes**: <20 samples (few-shot)
- **Metric**: Per-group accuracy (head/medium/tail)

## Why This Benchmark?

**Long-tail recognition** is ubiquitous in real-world applications:
- Medical diagnosis (rare diseases)
- Wildlife classification (endangered species)
- E-commerce (niche products)
- Content moderation (rare violations)

### The Challenge

With extreme class imbalance:
- **Majority classes** dominate the loss and gradients
- **Minority classes** are effectively ignored during training
- Standard optimizers overfit to head classes
- Tail class accuracy suffers dramatically

### The Gradient Quality Problem

From a gradient perspective:
- **Head class gradients**: Consistent and reliable (many samples) → High quality
- **Tail class gradients**: Sparse and variable (few samples) → Appear "noisy"

Standard optimizers can't distinguish between:
1. **True noise**: Gradients that should be ignored
2. **Rare but valid signals**: Important tail class gradients

### How CASMO Addresses This

CASMO's AGAR-based adaptation is designed to help with imbalanced learning:

- **Head classes**: Consistent gradients → High AGAR → Normal updates
- **Tail classes**: Even with high variance, CASMO's adaptive `c_min` prevents over-suppression

The key insight: CASMO adapts its confidence thresholds based on the _distribution_ of gradients, not just their magnitude. For tail classes with genuinely informative gradients, CASMO can maintain reasonable learning rates despite higher variance.

## Hypothesis

- **AdamW**: Will excel on head classes but fail on tail classes due to gradient domination
- **CASMO**: Will achieve more balanced learning across all class groups, especially improving tail class accuracy

## Technical Details

### Dataset Construction
- **Max samples (head)**: 500 per class
- **Min samples (tail)**: 5 per class
- **Distribution**: Exponential decay n_i = n_max × (100)^(-i/99)
- **Total**: ~12,000 training samples (vs 50,000 in balanced CIFAR-100)

### Class Groups
- **Many-shot** (>100 samples): Expected high accuracy for both optimizers
- **Medium-shot** (20-100 samples): Moderate challenge
- **Few-shot** (<20 samples): Critical differentiator for CASMO

### Evaluation
Test set remains balanced to fairly evaluate generalization across all classes.

## Expected Outcome

CASMO should demonstrate:
1. **Competitive head class accuracy** (similar to AdamW)
2. **Improved medium class accuracy** (better than AdamW)
3. **Significantly better tail class accuracy** (key advantage)
4. **Higher overall accuracy** due to balanced learning
