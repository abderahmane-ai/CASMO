"""
Continual Learning Benchmark: Sequential Multi-Task LLM Fine-tuning
Benchmark ID: B7

Tests CASMO's ability to prevent catastrophic forgetting during sequential 
task fine-tuning on LLMs - a critical unsolved problem in deployment.

Task:
    Model: Gemma-2-2B (4-bit quantized) + LoRA
    Tasks: Math → Code → QA → Writing (trained sequentially)
    Evaluation: Test on all previous tasks after each training phase
    
    Hypothesis:
    - AdamW will suffer catastrophic forgetting (50%+ accuracy drop on old tasks)
    - CASMO will maintain performance on old tasks (<10% accuracy drop)
    
    Reasoning:
    - Gradients conflicting with previous knowledge → Low AGAR → CASMO preserves old tasks
    - Gradients aligned with general patterns → High AGAR → CASMO learns new tasks
"""

import sys
import os
import argparse
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import warnings

# Add parent directory to path to import casmo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from casmo import CASMO
from dataset import ContinualLearningDataset

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_and_tokenizer(model_name="unsloth/gemma-2-2b-bnb-4bit"):
    """Load Gemma-2-2B with 4-bit quantization and LoRA."""
    print(f"Loading {model_name} with 4-bit quantization...")
    
    # Suppress the quantization config warning since unsloth models are pre-quantized
    warnings.filterwarnings('ignore', message='.*quantization_config.*')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load pre-quantized model (no need to pass quantization_config)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Suppress gradient checkpointing warning
    warnings.filterwarnings('ignore', message='.*use_reentrant.*')
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    # Disable cache when using gradient checkpointing to avoid warning
    model.config.use_cache = False
    
    # LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


# -----------------------------------------------------------------------------
# Training and Evaluation
# -----------------------------------------------------------------------------

def train_on_task(model, optimizer, train_loader, task_id, epochs=3, grad_accum_steps=32, results_dict=None):
    """Train on a single task."""
    model.train()
    task_names = ["Math", "Code", "QA", "Writing"]
    
    print(f"\n{'='*70}")
    print(f"Training on Task {task_id}: {task_names[task_id]}")
    print(f"{'='*70}\n")
    
    global_step = 0
    losses = []
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Task {task_id} Epoch {epoch+1}")
        optimizer.zero_grad()
        
        for i, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum_steps
            
            loss.backward()
            
            if (i + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                current_loss = loss.item() * grad_accum_steps
                losses.append(current_loss)
                epoch_loss += current_loss
                
                progress_bar.set_postfix({'loss': f"{current_loss:.4f}"})
                
                # Track AGAR/Confidence for CASMO (Plot 3)
                if results_dict is not None and hasattr(optimizer, '_group_states'):
                    group_state = optimizer._group_states.get(0, {})
                    agar = group_state.get('current_agar')
                    conf = group_state.get('current_confidence')
                    
                    if agar is not None:
                        results_dict['agar_history'].append(agar)
                        results_dict['confidence_history'].append(conf)
                        results_dict['steps'].append(global_step + (task_id * len(train_loader) // grad_accum_steps * epochs))
                
                global_step += 1
        
        avg_loss = epoch_loss / (len(train_loader) // grad_accum_steps)
        print(f"  Average Loss: {avg_loss:.4f}")
    
    return losses, global_step


def evaluate_on_task(model, test_loader, task_id):
    """Evaluate on a single task using perplexity as metric."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            # Calculate number of non-padded tokens
            num_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    # Convert perplexity to a "accuracy-like" metric (0-100 scale)
    # Lower perplexity = better, so invert it
    # Cap perplexity at 100 for numerical stability
    capped_perp = min(perplexity, 100)
    pseudo_accuracy = 100 * (1 - (capped_perp - 1) / 99)  # Maps [1, 100] to [100, 0]
    
    return pseudo_accuracy, perplexity


# -----------------------------------------------------------------------------
# Main Benchmark
# -----------------------------------------------------------------------------

def run_continual_learning(optimizer_name, tokenizer, epochs_per_task=3, 
                          batch_size=1, grad_accum_steps=32, lr=2e-4, seed=42):
    """
    Run continual learning experiment.
    
    Returns:
        results: Dictionary with metrics for all tasks over time
    """
    print(f"\n{'='*70}")
    print(f"Running: {optimizer_name.upper()}")
    print(f"{'='*70}\n")
    
    set_seed(seed)
    
    # Initialize model
    model, _ = get_model_and_tokenizer()
    
    # Create optimizer
    if optimizer_name == 'casmo':
        optimizer = CASMO(
            model.parameters(),
            lr=lr,
            tau_init_steps=100,  # Fast calibration
            weight_decay=0.01,
            granularity='group',
            betas=(0.9, 0.999),
            log_level=1
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Results tracking
    task_names = ["Math", "Code", "QA", "Writing"]
    results = {
        'task_accuracy': {i: [] for i in range(4)},  # Accuracy for each task over time
        'task_perplexity': {i: [] for i in range(4)},  # Perplexity for each task
        'training_losses': [],
        'evaluation_points': [],  # Which tasks were trained so far at each eval point
        'agar_history': [],      # For Plot 3
        'confidence_history': [], # For Plot 3
        'steps': []              # For Plot 3
    }
    
    # Create test loaders for all tasks (used throughout)
    test_loaders = []
    for task_id in range(4):
        test_dataset = ContinualLearningDataset(tokenizer, task_id, split='test', max_samples=100, max_length=256)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_loaders.append(test_loader)
    
    # Sequential training
    for task_id in range(4):
        # Create train loader for current task
        train_dataset = ContinualLearningDataset(tokenizer, task_id, split='train', max_samples=200, max_length=256)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Train on current task
        task_losses = train_on_task(model, optimizer, train_loader, task_id, 
                                    epochs=epochs_per_task, grad_accum_steps=grad_accum_steps,
                                    results_dict=results if optimizer_name == 'casmo' else None)
        results['training_losses'].extend(task_losses)
        
        # Evaluate on all tasks learned so far
        print(f"\n{'='*70}")
        print(f"Evaluation after training Task {task_id} ({task_names[task_id]})")
        print(f"{'='*70}\n")
        
        eval_results = {}
        for eval_task_id in range(task_id + 1):
            acc, perp = evaluate_on_task(model, test_loaders[eval_task_id], eval_task_id)
            eval_results[eval_task_id] = acc
            
            print(f"  Task {eval_task_id} ({task_names[eval_task_id]}): "
                  f"Score={acc:.2f}, Perplexity={perp:.2f}")
        
        # Store results
        for eval_task_id in range(task_id + 1):
            results['task_accuracy'][eval_task_id].append(eval_results[eval_task_id])
        
        # Pad tasks not yet trained with None
        for future_task_id in range(task_id + 1, 4):
            results['task_accuracy'][future_task_id].append(None)
        
        results['evaluation_points'].append(task_id)
        
        print()
    
    # Calculate final metrics
    final_metrics = calculate_metrics(results)
    
    print(f"\n{'='*70}")
    print(f"Final Metrics for {optimizer_name.upper()}")
    print(f"{'='*70}")
    print(f"  Average Accuracy: {final_metrics['average_accuracy']:.2f}")
    print(f"  Backward Transfer: {final_metrics['backward_transfer']:.2f}")
    print(f"  Forgetting: {final_metrics['forgetting']:.2f}")
    print(f"{'='*70}\n")
    
    results['final_metrics'] = final_metrics
    
    return results


def calculate_metrics(results):
    """Calculate continual learning metrics."""
    task_accuracy = results['task_accuracy']
    
    # Final accuracy on all tasks
    final_accs = [task_accuracy[i][-1] for i in range(4)]
    average_accuracy = np.mean(final_accs)
    
    # Backward Transfer: average accuracy drop on previous tasks
    # BWT = (1/3) * sum(acc_final[i] - acc_when_first_trained[i]) for i < 3
    bwt_values = []
    for task_id in range(3):  # Tasks 0, 1, 2 (not 3 since it's last)
        initial_acc = task_accuracy[task_id][task_id]  # When first trained
        final_acc = task_accuracy[task_id][-1]  # After all training
        bwt_values.append(final_acc - initial_acc)
    
    backward_transfer = np.mean(bwt_values) if bwt_values else 0
    
    # Forgetting: maximum accuracy drop for any task
    forgetting_values = []
    for task_id in range(3):
        accs = [a for a in task_accuracy[task_id] if a is not None]
        if len(accs) > 1:
            max_acc = max(accs)
            final_acc = accs[-1]
            forgetting_values.append(max_acc - final_acc)
    
    forgetting = max(forgetting_values) if forgetting_values else 0
    
    return {
        'average_accuracy': average_accuracy,
        'backward_transfer': backward_transfer,
        'forgetting': forgetting,
        'final_accuracies': final_accs
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_results(casmo_results, adamw_results, save_dir):
    """Create comprehensive comparison plots."""
    
    task_names = ["Math", "Code", "QA", "Writing"]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Task Performance Over Time (CASMO)
    ax = axes[0, 0]
    for task_id in range(4):
        accs = casmo_results['task_accuracy'][task_id]
        x_points = [i for i, a in enumerate(accs) if a is not None]
        y_values = [a for a in accs if a is not None]
        ax.plot(x_points, y_values, marker='o', label=f"Task {task_id}: {task_names[task_id]}", linewidth=2)
    
    ax.set_title('CASMO: Task Performance Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Training Phase (After Task N)')
    ax.set_ylabel('Performance Score')
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"After {task_names[i]}" for i in range(4)], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 2. Task Performance Over Time (AdamW)
    ax = axes[0, 1]
    for task_id in range(4):
        accs = adamw_results['task_accuracy'][task_id]
        x_points = [i for i, a in enumerate(accs) if a is not None]
        y_values = [a for a in accs if a is not None]
        ax.plot(x_points, y_values, marker='s', label=f"Task {task_id}: {task_names[task_id]}", linewidth=2, linestyle='--')
    
    ax.set_title('AdamW: Task Performance Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Training Phase (After Task N)')
    ax.set_ylabel('Performance Score')
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"After {task_names[i]}" for i in range(4)], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 3. Direct Comparison: Final Performance
    ax = axes[0, 2]
    x = np.arange(4)
    width = 0.35
    
    casmo_final = casmo_results['final_metrics']['final_accuracies']
    adamw_final = adamw_results['final_metrics']['final_accuracies']
    
    ax.bar(x - width/2, casmo_final, width, label='CASMO', color='green', alpha=0.8)
    ax.bar(x + width/2, adamw_final, width, label='AdamW', color='orange', alpha=0.8)
    
    ax.set_title('Final Performance on All Tasks', fontsize=14, fontweight='bold')
    ax.set_xlabel('Task')
    ax.set_ylabel('Performance Score')
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    # 4. Forgetting Curves (All Tasks Combined)
    ax = axes[1, 0]
    
    # Calculate forgetting for each task at each step
    for task_id in range(3):  # Exclude last task (no forgetting possible)
        casmo_accs = [a for a in casmo_results['task_accuracy'][task_id] if a is not None]
        adamw_accs = [a for a in adamw_results['task_accuracy'][task_id] if a is not None]
        
        if len(casmo_accs) > 1:
            # Forgetting = max_acc - current_acc
            casmo_max = max(casmo_accs)
            adamw_max = max(adamw_accs)
            
            casmo_forgetting = [casmo_max - acc for acc in casmo_accs]
            adamw_forgetting = [adamw_max - acc for acc in adamw_accs]
            
            x_points = list(range(len(casmo_forgetting)))
            ax.plot(x_points, casmo_forgetting, marker='o', label=f'CASMO: {task_names[task_id]}', linewidth=2)
            ax.plot(x_points, adamw_forgetting, marker='s', label=f'AdamW: {task_names[task_id]}', linewidth=2, linestyle='--', alpha=0.7)
    
    ax.set_title('Catastrophic Forgetting Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Training Phase')
    ax.set_ylabel('Accuracy Drop (Forgetting)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 5. Summary Metrics Comparison
    ax = axes[1, 1]
    
    metrics = ['Avg Accuracy', 'Backward Transfer', 'Forgetting']
    casmo_vals = [
        casmo_results['final_metrics']['average_accuracy'],
        casmo_results['final_metrics']['backward_transfer'],
        -casmo_results['final_metrics']['forgetting']  # Negative because lower is better
    ]
    adamw_vals = [
        adamw_results['final_metrics']['average_accuracy'],
        adamw_results['final_metrics']['backward_transfer'],
        -adamw_results['final_metrics']['forgetting']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, casmo_vals, width, label='CASMO', color='green', alpha=0.8)
    ax.bar(x + width/2, adamw_vals, width, label='AdamW', color='orange', alpha=0.8)
    
    ax.set_title('Summary Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score (Higher is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 6. Task-by-Task Forgetting Comparison
    ax = axes[1, 2]
    
    task_forgetting_casmo = []
    task_forgetting_adamw = []
    
    for task_id in range(3):
        casmo_accs = [a for a in casmo_results['task_accuracy'][task_id] if a is not None]
        adamw_accs = [a for a in adamw_results['task_accuracy'][task_id] if a is not None]
        
        if len(casmo_accs) > 1:
            task_forgetting_casmo.append(max(casmo_accs) - casmo_accs[-1])
            task_forgetting_adamw.append(max(adamw_accs) - adamw_accs[-1])
        else:
            task_forgetting_casmo.append(0)
            task_forgetting_adamw.append(0)
    
    x = np.arange(3)
    width = 0.35
    
    ax.bar(x - width/2, task_forgetting_casmo, width, label='CASMO', color='green', alpha=0.8)
    ax.bar(x + width/2, task_forgetting_adamw, width, label='AdamW', color='orange', alpha=0.8)
    
    ax.set_title('Forgetting Per Task', fontsize=14, fontweight='bold')
    ax.set_xlabel('Task')
    ax.set_ylabel('Final Forgetting (Accuracy Drop)')
    ax.set_xticks(x)
    ax.set_xticklabels([task_names[i] for i in range(3)], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'continual_learning_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Comprehensive plots saved to {save_path}")
    plt.close()

    # --- NEW PLOTS BASED ON ADVICE ---
    
    # Plot 1: The Forgetting Curve (The Hero Plot)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    def plot_forgetting_curve_single(results, name, ax):
        tasks = range(4)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Standard matplotlib cycle
        for task_id in tasks:
            # Extract accuracy for this task at each evaluation point
            # results['task_accuracy'][task_id] contains [acc_after_task0, acc_after_task1, ...]
            accuracies = [a for a in results['task_accuracy'][task_id] if a is not None]
            # X-axis points: The evaluation points where this task was measured
            # e.g. Task 0 measured at 0, 1, 2, 3. Task 1 measured at 1, 2, 3.
            x_points = range(task_id, 4) 
            
            if len(accuracies) > 0:
                ax.plot(x_points, accuracies, 'o-', label=f'Task {task_id}: {task_names[task_id]}', 
                        linewidth=2, color=colors[task_id])
        
        ax.set_xticks(tasks)
        ax.set_xticklabels([f'After Task {i}' for i in tasks])
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(name, fontsize=14)
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plot_forgetting_curve_single(adamw_results, "AdamW (Catastrophic Forgetting)", ax1)
    plot_forgetting_curve_single(casmo_results, "CASMO (Retained Knowledge)", ax2)
    fig.suptitle('B7: Continual Learning - Sequential Task Performance', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot1_forgetting_curves.png'), dpi=150, bbox_inches='tight')
    print(f"✅ Plot 1 saved to plot1_forgetting_curves.png")
    plt.close()

    # Plot 2: Final Accuracy Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    tasks = range(4)
    adamw_final = adamw_results['final_metrics']['final_accuracies']
    casmo_final = casmo_results['final_metrics']['final_accuracies']
    
    x = np.arange(len(tasks))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, adamw_final, width, label='AdamW', color='tomato', alpha=0.8)
    bars2 = ax.bar(x + width/2, casmo_final, width, label='CASMO', color='steelblue', alpha=0.8)
    
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Final Accuracy (%)', fontsize=12)
    ax.set_title('Final Performance After Sequential Training', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Task {i}: {task_names[i]}' for i in tasks])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
                
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot2_final_performance.png'), dpi=150, bbox_inches='tight')
    print(f"✅ Plot 2 saved to plot2_final_performance.png")
    plt.close()

    # Plot 3: AGAR/Confidence Over Time (Mechanism Proof)
    if casmo_results.get('agar_history'):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        steps = casmo_results['steps']
        agar = casmo_results['agar_history']
        conf = casmo_results['confidence_history']
        
        # Plot AGAR
        ax1.plot(steps, agar, color='green', alpha=0.7, linewidth=1)
        ax1.set_ylabel('AGAR', fontsize=12)
        ax1.set_title('CASMO Mechanism: AGAR Dynamics During Sequential Training', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Plot Confidence
        ax2.plot(steps, conf, color='blue', alpha=0.7, linewidth=1)
        ax2.set_xlabel('Training Step', fontsize=12)
        ax2.set_ylabel('Confidence', fontsize=12)
        ax2.set_title('Confidence Scaling (Response to AGAR)', fontsize=14)
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)
        
        # Add vertical lines for task boundaries (approximate)
        total_steps = steps[-1] if steps else 0
        if total_steps > 0:
            steps_per_task = total_steps / 4
            for i in range(1, 4):
                boundary = i * steps_per_task
                ax1.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
                ax2.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
                # Add task labels
                ax1.text(boundary - steps_per_task/2, max(agar)*0.9, f"Task {i-1}", ha='center', alpha=0.6)
            ax1.text(total_steps - steps_per_task/2, max(agar)*0.9, f"Task 3", ha='center', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'plot3_mechanism.png'), dpi=150, bbox_inches='tight')
        print(f"✅ Plot 3 saved to plot3_mechanism.png")
        plt.close()

    # Plot 4: Summary Metrics Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Avg Accuracy', 'Backward Transfer', 'Forgetting']
    # Note: Forgetting is usually positive (drop), so we plot it as is. 
    # Lower is better for Forgetting. Higher is better for others.
    
    casmo_vals = [
        casmo_results['final_metrics']['average_accuracy'],
        casmo_results['final_metrics']['backward_transfer'],
        casmo_results['final_metrics']['forgetting']
    ]
    adamw_vals = [
        adamw_results['final_metrics']['average_accuracy'],
        adamw_results['final_metrics']['backward_transfer'],
        adamw_results['final_metrics']['forgetting']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, adamw_vals, width, label='AdamW', color='tomato', alpha=0.8)
    bars2 = ax.bar(x + width/2, casmo_vals, width, label='CASMO', color='steelblue', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Continual Learning Metrics Summary', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (1 if height>=0 else -5),
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot4_summary_metrics.png'), dpi=150, bbox_inches='tight')
    print(f"✅ Plot 4 saved to plot4_summary_metrics.png")
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='B7: Continual Learning Benchmark')
    parser.add_argument('--epochs_per_task', type=int, default=2, help='Epochs to train on each task')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--grad_accum_steps', type=int, default=32, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--quick_test', action='store_true', help='Quick test with fewer samples')
    args = parser.parse_args()
    
    print("="*70)
    print("B7: Continual Learning Benchmark")
    print("Sequential Multi-Task Fine-tuning: Math → Code → QA → Writing")
    print("="*70)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available. This benchmark requires a GPU.")
        return
    
    print(f"\nDevice: cuda")
    print(f"Effective batch size: {args.batch_size * args.grad_accum_steps}")
    
    # Create results directory
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load tokenizer (shared between both runs)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-2-2b-bnb-4bit", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Run both optimizers
    print("\n" + "="*70)
    print("STARTING CASMO RUN")
    print("="*70)
    casmo_results = run_continual_learning(
        'casmo', tokenizer,
        epochs_per_task=args.epochs_per_task,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr
    )
    
    # Clear GPU memory before ADAMW run to prevent memory fragmentation slowdown
    print("\nClearing GPU memory...")
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory cleared.\n")
    
    print("\n" + "="*70)
    print("STARTING ADAMW RUN")
    print("="*70)
    adamw_results = run_continual_learning(
        'adamw', tokenizer,
        epochs_per_task=args.epochs_per_task,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr
    )
    
    # Create comprehensive plots
    print("\nCreating comparison plots...")
    plot_results(casmo_results, adamw_results, results_dir)
    
    # Print final comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    
    task_names = ["Math", "Code", "QA", "Writing"]
    print(f"\n{'Task':<15} {'CASMO':<15} {'AdamW':<15} {'Difference':<15}")
    print("-" * 60)
    
    for i in range(4):
        casmo_acc = casmo_results['final_metrics']['final_accuracies'][i]
        adamw_acc = adamw_results['final_metrics']['final_accuracies'][i]
        diff = casmo_acc - adamw_acc
        print(f"{task_names[i]:<15} {casmo_acc:<15.2f} {adamw_acc:<15.2f} {diff:+.2f}")
    
    print("-" * 60)
    print(f"{'Average':<15} "
          f"{casmo_results['final_metrics']['average_accuracy']:<15.2f} "
          f"{adamw_results['final_metrics']['average_accuracy']:<15.2f} "
          f"{casmo_results['final_metrics']['average_accuracy'] - adamw_results['final_metrics']['average_accuracy']:+.2f}")
    
    print(f"\n{'Metric':<25} {'CASMO':<15} {'AdamW':<15}")
    print("-" * 55)
    print(f"{'Backward Transfer':<25} "
          f"{casmo_results['final_metrics']['backward_transfer']:<15.2f} "
          f"{adamw_results['final_metrics']['backward_transfer']:<15.2f}")
    print(f"{'Forgetting (max drop)':<25} "
          f"{casmo_results['final_metrics']['forgetting']:<15.2f} "
          f"{adamw_results['final_metrics']['forgetting']:<15.2f}")
    
    print("\n" + "="*70)
    print("✅ Benchmark Complete!")
    print("="*70)


if __name__ == '__main__':
    main()
