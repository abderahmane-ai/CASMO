"""
B7 Ablation Study: High Conflict Stress Test
Benchmark ID: B7 (Conflict Ablation)

Tests CASMO's mechanism under extreme gradient conflict to validate that
AGAR correctly detects and responds to task interference.

Task:
    Model: Gemma-2-2B (4-bit quantized) + LoRA
    Scenario: Sequential training on contradictory formatting for same content
    - Task A: Math problems in format "Question: X Answer: Y"
    - Task B: SAME math problems in format "Input: X Output: The solution is Y"
    
    This creates pure gradient conflict (same semantic content, different tokens),
    which is the worst-case scenario for standard optimizers.
    
    Hypothesis:
    - AdamW will suffer catastrophic forgetting (50%+ perplexity increase)
    - CASMO will maintain stability (~30% perplexity increase or less)
    
    Reasoning:
    - Conflicting gradients → Low AGAR → CASMO reduces learning rate automatically
    - Confidence histogram will show CASMO detecting conflict (values < 1.0)
    - This validates AGAR as a conflict detector, not just a performance optimizer
"""

import sys
import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

# Add parent directory to path to import casmo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from casmo import CASMO

# -----------------------------------------------------------------------------
# Synthetic Data
# -----------------------------------------------------------------------------

class ConflictingMathDataset(Dataset):
    def __init__(self, tokenizer, task_id, num_samples=200, max_length=64):
        self.tokenizer = tokenizer
        self.task_id = task_id  # 0 = Format A, 1 = Format B
        self.max_length = max_length
        self.samples = self._generate_samples(num_samples)

    def _generate_samples(self, num_samples):
        samples = []
        for i in range(num_samples):
            # Generate simple math problems
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            op = random.choice(['+', '-', '*'])
            
            if op == '+': ans = a + b
            elif op == '-': ans = a - b
            else: ans = a * b
            
            question = f"{a} {op} {b} = ?"
            
            if self.task_id == 0:
                # Format A
                text = f"Question: {question}\nAnswer: {ans}"
            else:
                # Format B (Conflicting)
                text = f"Input: {question}\nOutput: The solution is {ans}"
                
            samples.append(text)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

def get_model_and_tokenizer():
    model_name = "unsloth/gemma-2-2b-bnb-4bit"
    print(f"Loading {model_name}...")
    
    warnings.filterwarnings('ignore')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    return model, tokenizer

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train_task(model, optimizer, dataloader, task_name, epochs=2, track_stats=False):
    model.train()
    stats = {'agar': [], 'confidence': [], 'loss': []}
    
    print(f"\nTraining {task_name}...")
    
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            stats['loss'].append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Track CASMO internals
            if track_stats and hasattr(optimizer, '_group_states'):
                group_state = optimizer._group_states.get(0, {})
                agar = group_state.get('current_agar')
                conf = group_state.get('current_confidence')
                if agar is not None:
                    stats['agar'].append(agar)
                    stats['confidence'].append(conf)
                    
    return stats

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            count += 1
    return np.exp(total_loss / count)  # Perplexity

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    # Setup
    model, tokenizer = get_model_and_tokenizer()
    
    # Datasets
    # Task A: Format 1
    ds_a_train = ConflictingMathDataset(tokenizer, task_id=0, num_samples=200)
    ds_a_test = ConflictingMathDataset(tokenizer, task_id=0, num_samples=50)
    dl_a_train = DataLoader(ds_a_train, batch_size=4, shuffle=True)
    dl_a_test = DataLoader(ds_a_test, batch_size=4)
    
    # Task B: Format 2 (Conflict)
    ds_b_train = ConflictingMathDataset(tokenizer, task_id=1, num_samples=200)
    dl_b_train = DataLoader(ds_b_train, batch_size=4, shuffle=True)
    
    results_dir = os.path.join(os.path.dirname(__file__), 'ablation_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Run 1: CASMO
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("RUNNING CASMO (Conflict Test)")
    print("="*50)
    
    optimizer = CASMO(model.parameters(), lr=2e-4, tau_init_steps=50)
    
    # Train Task A
    train_task(model, optimizer, dl_a_train, "Task A (Format 1)", track_stats=True)
    perp_a_initial = evaluate(model, dl_a_test)
    print(f"Task A Initial Perplexity: {perp_a_initial:.2f}")
    
    # Train Task B (Conflict)
    stats_casmo = train_task(model, optimizer, dl_b_train, "Task B (Format 2)", track_stats=True)
    perp_a_final_casmo = evaluate(model, dl_a_test)
    print(f"Task A Final Perplexity (CASMO): {perp_a_final_casmo:.2f}")
    
    # -------------------------------------------------------------------------
    # Reset & Run 2: AdamW
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("RUNNING ADAMW (Conflict Test)")
    print("="*50)
    
    # Clear memory
    del model, optimizer
    import gc; gc.collect(); torch.cuda.empty_cache()
    
    model, _ = get_model_and_tokenizer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    
    # Train Task A
    train_task(model, optimizer, dl_a_train, "Task A (Format 1)")
    perp_a_initial_adamw = evaluate(model, dl_a_test)
    print(f"Task A Initial Perplexity: {perp_a_initial_adamw:.2f}")
    
    # Train Task B
    train_task(model, optimizer, dl_b_train, "Task B (Format 2)")
    perp_a_final_adamw = evaluate(model, dl_a_test)
    print(f"Task A Final Perplexity (AdamW): {perp_a_final_adamw:.2f}")
    
    # -------------------------------------------------------------------------
    # Analysis & Plotting
    # -------------------------------------------------------------------------
    
    # 1. Performance Comparison
    casmo_forgetting = perp_a_final_casmo - perp_a_initial
    adamw_forgetting = perp_a_final_adamw - perp_a_initial_adamw
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"CASMO Forgetting (Perplexity Increase): {casmo_forgetting:.2f}")
    print(f"AdamW Forgetting (Perplexity Increase): {adamw_forgetting:.2f}")
    print(f"Improvement: {adamw_forgetting / casmo_forgetting:.1f}x better stability")
    
    # 2. Confidence Histogram (The Mechanism Proof)
    plt.figure(figsize=(10, 6))
    
    # Split confidence into phases if possible, or just plot distribution
    # We want to show that confidence varies
    conf_values = stats_casmo['confidence']
    
    plt.hist(conf_values, bins=30, color='purple', alpha=0.7, edgecolor='black')
    plt.title("CASMO Confidence Distribution During Conflict", fontsize=14)
    plt.xlabel("Confidence (Learning Rate Multiplier)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.axvline(x=1.0, color='red', linestyle='--', label='Standard AdamW (1.0)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(results_dir, 'mechanism_confidence_hist.png')
    plt.savefig(save_path)
    print(f"\n✅ Mechanism plot saved to {save_path}")
    
    # 3. Bar Chart
    plt.figure(figsize=(8, 6))
    methods = ['CASMO', 'AdamW']
    forgetting = [casmo_forgetting, adamw_forgetting]
    colors = ['green', 'red']
    
    plt.bar(methods, forgetting, color=colors, alpha=0.7)
    plt.title("Catastrophic Forgetting in High-Conflict Scenario", fontsize=14)
    plt.ylabel("Perplexity Increase (Lower is Better)", fontsize=12)
    
    save_path = os.path.join(results_dir, 'conflict_ablation_results.png')
    plt.savefig(save_path)
    print(f"✅ Ablation results saved to {save_path}")

if __name__ == '__main__':
    main()
