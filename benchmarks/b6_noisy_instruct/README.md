# B6: Noisy Instruction Tuning - LLM Fine-tuning

## Overview

This benchmark tests CASMO's ability to fine-tune large language models (LLMs) on instruction datasets with corrupted responses, a critical challenge for real-world LLM deployment.

## Task Description

- **Base Model**: Gemma-2-2B (quantized to 4-bit)
- **Method**: LoRA (Low-Rank Adaptation) fine-tuning
- **Dataset**: Instruction-response pairs with 30% corruption
- **Noise Type**: Shuffled or random responses
- **Training**: Gradient accumulation (effective batch size 32)
- **Metric**: Training loss, confidence distribution on clean vs noisy batches

## Why This Benchmark?

**Instruction tuning** is essential for making LLMs useful, but training data quality is a major concern:
- Crowdsourced annotations contain errors
- Automated dataset generation produces inconsistencies
- Model-generated responses may be incorrect
- Human feedback can be contradictory

### The Challenge

With corrupted instruction-response pairs:
- **Noisy responses**: Teach the model incorrect behaviors
- **Inconsistent patterns**: Break the instruction-following capability
- **Scale**: LLMs are expensive to train, wasting compute on bad data is costly

Traditional approaches require:
- Manual data cleaning (expensive)
- Multiple rounds of filtering (time-consuming)
- Confidence-based sample selection (requires separate models)

### How CASMO Addresses This

CASMO can potentially detect data quality issues automatically:

- **Clean instruction-response pairs**: Gradients align with language modeling objectives → High AGAR → Full confidence
- **Corrupted pairs**: Gradients conflict with coherent language patterns → Low AGAR → Reduced confidence

This enables:
1. **Automatic noise robustness** without manual filtering
2. **Efficient compute usage** by downweighting bad samples
3. **Better final model quality** despite noisy training data

## Hypothesis

- **AdamW**: Will attempt to fit both clean and noisy responses, learning inconsistent behaviors
- **CASMO**: Will detect low-quality gradients from corrupted pairs and focus on clean data, achieving better instruction-following

## Technical Details

### Model Configuration

**Base Model**: Gemma-2-2B
- **Quantization**: 4-bit NF4 (reduces memory from 8GB to ~2GB)
- **Compute dtype**: FP16
- **Total parameters**: 2.6B
- **Trainable parameters**: ~16M (via LoRA)

**LoRA Settings**:
- **Rank**: 16
- **Alpha**: 32
- **Target modules**: All attention and MLP projections
- **Dropout**: 0.05

### Training Setup

- **Batch size**: 1 (fits in 6GB VRAM)
- **Gradient accumulation**: 32 steps
- **Effective batch size**: 32
- **Learning rate**: 2e-4
- **Optimizer**: CASMO vs AdamW
- **Weight decay**: 0.01

### Data Corruption

30% of instruction-response pairs are corrupted by:
1. **Shuffling**: Pairing instructions with random responses
2. **Ensuring mismatch**: Corrupted pairs are guaranteed to be wrong

### Evaluation Strategy

Since this is a short benchmark (150 steps):
- **Primary metric**: Training loss trajectory
- **Secondary metric**: CASMO confidence on clean vs noisy batches
- **Hypothesis**: CASMO should show lower confidence on corrupted pairs

## Expected Outcome

CASMO should demonstrate:

1. **Similar or better loss** compared to AdamW (efficient learning from clean data)
2. **Confidence discrimination**: Lower confidence on noisy batches, higher on clean batches
3. **Robustness**: Less overfitting to corrupted instruction-response pairs
4. **Efficiency**: Better compute utilization by focusing on quality gradients

This benchmark validates CASMO's applicability to modern LLM fine-tuning workflows, where data quality is a persistent challenge.

## Hardware Requirements

- **Minimum VRAM**: 6GB (tested on RTX 4050 Laptop)
- **Quantization**: 4-bit required for 6GB VRAM
- **Gradient checkpointing**: Enabled to reduce memory
