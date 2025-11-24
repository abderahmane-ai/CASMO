# B7: Continual Learning - Sequential Multi-Task Fine-tuning

## Overview

This benchmark tests CASMO's ability to prevent **catastrophic forgetting** during sequential task fine-tuning on LLMs - one of the most critical unsolved problems in modern AI deployment.

## Task Description

- **Model**: Gemma-2-2B (4-bit quantized) + LoRA
- **Tasks**: 4 distinct domains trained sequentially
  1. **Mathematics** (GSM8K) - Arithmetic reasoning
  2. **Code Generation** (MBPP) - Python programming
  3. **Commonsense QA** - Factual knowledge
  4. **Creative Writing** - Story completion
- **Training Protocol**: Train on each task sequentially (3 epochs each)
- **Evaluation**: Test on all previous tasks after each training phase
- **Metrics**: Backward Transfer, Forgetting, Average Accuracy

## Why This Benchmark?

### The Catastrophic Forgetting Problem

**Catastrophic forgetting** is THE major barrier to deploying multi-domain LLMs:
- When you fine-tune on Task B, the model forgets Task A
- Industry workaround: Maintain separate model copies (expensive)
- Or use replay buffers (memory intensive, slower training)
- No efficient optimizer-level solution exists

### Real-World Impact

This problem affects:
- **Multi-domain chatbots**: Train on medical data → forget customer service
- **Personalization**: Fine-tune for User B → forget User A's preferences
- **Continual deployment**: Update model weekly → degrades on older tasks
- **Resource-constrained settings**: Can't afford multiple model copies

### Why Standard Optimizers Fail

**AdamW's fundamental problem:**
- **Momentum**: Accumulates in direction of new task gradients
- **Adaptive moments**: Statistics shift to new task distribution
- **No conflict detection**: Can't tell when gradients harm previous knowledge

This leads to **50%+ accuracy drops** on old tasks after learning new ones.

### How CASMO Addresses This

CASMO's AGAR (Adaptive Gradient Alignment Ratio) naturally detects knowledge conflicts:

**When learning Task 2:**
- **Conflicting gradients** (hurt Task 1 knowledge) → Low AGAR → Low confidence → Preserve old knowledge
- **Aligned gradients** (compatible with both tasks) → High AGAR → Normal learning → Learn new patterns
- **General improvements** (help all tasks) → High AGAR → Full confidence → Update aggressively

This is effectively **elastic weight consolidation** without explicit importance weighting!

## Hypothesis

| Optimizer | Behavior | Expected Forgetting |
|-----------|----------|---------------------|
| **AdamW** | Aggressive learning on new tasks destroys old knowledge | **50%+ accuracy drop** |
| **CASMO** | Gradient quality detection preserves old tasks while learning new ones | **<10% accuracy drop** |

**If CASMO achieves this, it's a breakthrough result** - no optimizer has solved catastrophic forgetting at this level before.

## Technical Details

### Tasks Designed for Maximum Interference

The 4 tasks are deliberately chosen to maximally interfere with each other:
- **Math → Code**: Different token distributions (numbers vs syntax)
- **Code → QA**: Different output formats (code blocks vs letters)
- **QA → Writing**: Different modes (factual vs creative)

This creates the **worst-case scenario** for forgetting - exactly what we want to test!

### Evaluation Protocol

```
Phase 1: Train Math     → Test: [Math]
                           Record: Math 75%

Phase 2: Train Code     → Test: [Math, Code]
                           Record: Math 45% ❌ Code 70%
                           Forgetting: -30% on Math!

Phase 3: Train QA       → Test: [Math, Code, QA]
                           Record: Math 30%, Code 40%, QA 80%
                           More forgetting!

Phase 4: Train Writing  → Test: [Math, Code, QA, Writing]
                           Record: All final scores
                           Calculate metrics
```

### Continual Learning Metrics

1. **Average Accuracy**: Mean performance across all 4 tasks at end
2. **Backward Transfer (BWT)**: Average accuracy change on previous tasks
   - BWT = (1/3) × Σ(final_acc[i] - initial_acc[i]) for i < 3
   - Negative = forgetting, Positive = knowledge transfer
3. **Forgetting**: Maximum accuracy drop on any previous task
   - Forgetting = max(max_acc[i] - final_acc[i]) for i < 3
   - Lower is better

### Model Configuration

**Gemma-2-2B Settings:**
- **Quantization**: 4-bit NF4 (~2GB VRAM)
- **LoRA**: r=16, alpha=32
- **Target modules**: All attention and MLP projections
- **Trainable params**: ~16M (0.6% of base model)

**Training:**
- **Batch size**: 1 (memory efficient)
- **Gradient accumulation**: 32 steps
- **Effective batch size**: 32
- **Epochs per task**: 3
- **Total runtime**: ~2 hours on RTX 4050 6GB

### Why This Will Impress Top Labs

1. **Berkeley AI Research (BAIR)**: Active continual learning research (A-GEM, Experience Replay papers)
2. **Edinburgh NLP**: Strong focus on multi-task learning and catastrophic forgetting
3. **DeepMind**: Extensive publications on continual learning for neural networks
4. **OpenAI**: Critical for GPT fine-tuning and personalization

**Novel contribution:**
- First demonstration of optimizer-level forgetting prevention
- No replay buffers or importance weights needed
- Purely gradient quality-based approach

## Expected Results

### AdamW Baseline (Expected)
- Task 1 (Math): 75% → 25% (catastrophic collapse)
- Task 2 (Code): 70% → 35% (severe degradation)
- Task 3 (QA): 80% → 40% (major forgetting)
- Task 4 (Writing): 75% (only current task maintained)
- **Average**: 43.75%
- **Forgetting**: -50%

### CASMO (Hypothesis)
- Task 1 (Math): 75% → 65% (graceful retention)
- Task 2 (Code): 70% → 62% (maintained)
- Task 3 (QA): 80% → 68% (slight drop)
- Task 4 (Writing): 73% (all tasks balanced)
- **Average**: 67%
- **Forgetting**: -10%

**This would be a 54% improvement in average accuracy and 80% reduction in forgetting!**

## Visualization

The benchmark produces 6 comprehensive plots:

1. **CASMO Task Performance Timeline**: Shows how each task evolves
2. **AdamW Task Performance Timeline**: Direct visual comparison
3. **Final Performance Bar Chart**: Side-by-side comparison
4. **Catastrophic Forgetting Curves**: Track degradation over time
5. **Summary Metrics**: Avg accuracy, BWT, forgetting
6. **Per-Task Forgetting**: Breakdown by task

## Potential Impact

If successful, this benchmark demonstrates:
- ✅ **Practical solution** to catastrophic forgetting without architectural changes
- ✅ **Memory efficient**: No replay buffers needed
- ✅ **Computation efficient**: Single pass training
- ✅ **Drop-in replacement**: Just swap optimizer
- ✅ **Publishable result**: Novel approach to major open problem

This could enable:
- Multi-domain chatbots in production
- Efficient personalized LLM deployment
- Continual learning systems at scale
- Industry adoption of CASMO

## Running the Benchmark

```bash
# Full benchmark (~2 hours)
python benchmarks/b7_continual_learning/train.py

# Quick test
python benchmarks/b7_continual_learning/train.py --epochs_per_task 1 --quick_test
```

## Hardware Requirements

- **Minimum**: RTX 4050 6GB VRAM (tested configuration)
- **Recommended**: Any NVIDIA GPU with 6GB+ VRAM
- **Runtime**: ~2 hours for full benchmark
