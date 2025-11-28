# B2: Grokking - Modular Arithmetic with Noise

## Overview

This benchmark tests CASMO's ability to achieve faster generalization ("grokking") on algorithmic tasks with label noise, where understanding requires learning the underlying rule rather than memorizing patterns.

## Task Description

- **Task**: Modular addition (a + b) mod 97
- **Model**: 1-layer Transformer
- **Dataset**: All possible pairs (97 × 97 = 9,409 pairs)
- **Split**: 50% train, 50% test
- **Label Noise**: 30% of training labels are corrupted to random values
- **Metric**: Validation accuracy on clean labels

## Why This Benchmark?

**Grokking** is a fascinating phenomenon where models suddenly transition from memorization to generalization after extended training. This benchmark tests CASMO's ability to:
1. Ignore noisy labels that don't follow the algorithmic rule
2. Accelerate the discovery of the true underlying pattern
3. Achieve generalization faster than standard optimizers

### The Challenge

- Models must learn the modular arithmetic rule: `(a + b) % 97`
- With 30% noise, memorization strategies fail on the test set
- Standard optimizers may get stuck trying to fit the noise
- True understanding requires extracting the algorithmic structure

### How CASMO Addresses This

CASMO's gradient quality detection helps distinguish:
- **Generalizable patterns**: Gradients that align with the algorithmic rule → High AGAR
- **Noise/Memorization**: Gradients from corrupted labels → Low AGAR

By reducing confidence on noisy gradients, CASMO can:
- Focus computational resources on learning the true rule
- Avoid costly memorization of random corruptions
- Achieve faster "grokking" transitions

## Hypothesis

- **AdamW**: Will attempt to memorize both clean and noisy samples, leading to high train accuracy but low validation accuracy
- **CASMO**: Will downweight noisy samples, focusing on the underlying rule, achieving faster and better generalization

## Technical Details

### Dataset Characteristics
- **Input**: Two integers (a, b) where 0 ≤ a, b < 97
- **Output**: (a + b) % 97
- **Vocabulary size**: 98 (0-96 for values, plus prime number token)
- **Sequence format**: [a, b, p] → (a + b) % p

### Training Dynamics
- Initial phase: Both optimizers fit training data
- Mid-training: AdamW overfits to noise, CASMO resists
- Late training: CASMO achieves generalization while AdamW plateaus

### Expected Metrics
CASMO should show:
- Lower training accuracy (not fitting noise)
- Higher validation accuracy (better generalization)
- Faster grokking transition
- More stable AGAR values after calibration

## Observed Results (Nov 2025)

The benchmark confirmed the hypothesis with striking clarity:

| Metric | CASMO | AdamW |
| :--- | :---: | :---: |
| **Train Accuracy** | **~70.7%** | **100%** |
| **Val Accuracy** | **~90.4%** | **24.0%** |

- **AdamW** fell into the "memorization trap," learning the 30% noisy labels perfectly (100% train acc) but failing to learn the rule (24% val acc).
- **CASMO** refused to memorize the noise (capping at ~70% train acc) and successfully "grokked" the rule (90% val acc).

See [Full Report](reports/REPORT.md) for details.
