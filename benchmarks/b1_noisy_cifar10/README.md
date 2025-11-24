# B1: Noisy Label Learning - CIFAR-10

## Overview

This benchmark tests CASMO's ability to train effectively on datasets with corrupted labels by detecting and adapting to gradient quality.

## Task Description

- **Dataset**: CIFAR-10 (image classification)
- **Architecture**: ResNet-32 (standard for noisy label research)
- **Label Corruption**: 40% symmetric noise (random incorrect labels)
- **Training Set**: 50,000 images with noisy labels
- **Test Set**: 10,000 images with clean labels
- **Metric**: Test accuracy on clean labels

## Why This Benchmark?

**Noisy label learning** is a fundamental challenge in real-world machine learning where training data contains incorrect labels due to:
- Crowdsourced annotation errors
- Automatic labeling mistakes
- Data collection noise

### The Challenge

With 40% label corruption:
- Training labels are unreliable
- Standard optimizers tend to memorize the noise
- The model should learn the true patterns while ignoring corrupted samples

### How CASMO Addresses This

CASMO uses **AGAR (Adaptive Gradient Alignment Ratio)** to measure gradient quality:
- **Clean samples**: Gradients align well with the model's learned patterns → High AGAR → Normal learning rate
- **Noisy samples**: Gradients conflict with true patterns → Low AGAR → Reduced confidence → Less aggressive updates

This allows CASMO to naturally down-weight noisy samples without explicit noise detection.

## Hypothesis

- **AdamW**: Will gradually overfit to the noisy labels, achieving high training accuracy but poor test accuracy
- **CASMO**: Will resist memorizing noise by detecting low-quality gradients, achieving better generalization

## Technical Details

### Calibration Phase
CASMO uses an initial calibration period (5% of training steps, minimum 100 steps) to:
1. Collect AGAR statistics from early training
2. Learn the typical gradient distribution
3. Set adaptive confidence thresholds based on task characteristics

### Confidence Mapping
Uses a sigmoid-based mapping that adapts `c_min` (minimum confidence) based on gradient variance:
- **Low variance**: High `c_min` (more conservative, suitable for pervasive noise)
- **High variance**: Low `c_min` (aggressive discrimination for bimodal distributions)

## Expected Outcome

CASMO should achieve higher test accuracy by:
1. Learning faster from clean samples (high confidence)
2. Being more cautious with noisy samples (low confidence)
3. Better generalization to the clean test set
