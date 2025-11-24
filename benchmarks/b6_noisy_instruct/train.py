"""
Noisy Instruction Tuning Benchmark: LLM Fine-tuning with Corrupted Responses
Benchmark ID: B6

Tests CASMO's ability to fine-tune large language models on instruction datasets
with corrupted responses, a critical challenge for real-world LLM deployment.

Task:
    Base Model: Gemma-2-2B (4-bit quantized)
    Method: LoRA fine-tuning
    Dataset: Instruction-response pairs with 30% corruption
    
    Hypothesis:
    - AdamW will attempt to fit both clean and noisy responses, learning inconsistent behaviors
    - CASMO will detect low-quality gradients from corrupted pairs and focus on clean data
"""

import sys
import os
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path to import casmo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader

from casmo import CASMO
from dataset import NoisyInstructDataset

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

def get_model_and_tokenizer(model_name="unsloth/gemma-2-2b-bnb-4bit"):
    print(f"Loading {model_name} with 4-bit quantization...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    # LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Gemma 2 modules
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def run_training(optimizer_name, epochs=1, batch_size=1, grad_accum_steps=32, lr=2e-4, noise_ratio=0.3):
    print(f"\n{'='*60}")
    print(f"Running: {optimizer_name.upper()} | Noise: {noise_ratio*100}%")
    print(f"Batch: {batch_size} | Accum: {grad_accum_steps} | Effective Batch: {batch_size*grad_accum_steps}")
    print(f"{'='*60}\n")
    
    model, tokenizer = get_model_and_tokenizer()
    
    # Dataset
    train_dataset = NoisyInstructDataset(tokenizer, split='train', noise_ratio=noise_ratio)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    if optimizer_name == 'casmo':
        optimizer = CASMO(
            model.parameters(),
            lr=lr,
            tau_init_steps=100, # Fast calibration for short benchmark
            weight_decay=0.01,
            granularity='group', # 'group' is recommended for large models
            betas=(0.9, 0.999),
            log_level=1
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
    results = {
        'losses': [],
        'noisy_confidences': [],
        'clean_confidences': [],
        'steps': []
    }
    
    step = 0
    global_step = 0
    model.train()
    optimizer.zero_grad()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        progress_bar = tqdm(train_loader, desc="Training")
        
        current_batch_noisy = []
        
        for i, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            is_noisy = batch['is_noisy'] # [Batch]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum_steps # Scale loss
            
            loss.backward()
            
            # Track noise for the accumulated batch
            current_batch_noisy.append(is_noisy.float().mean().item())
            
            if (i + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                # Log Metrics (once per effective step)
                current_loss = loss.item() * grad_accum_steps
                results['losses'].append(current_loss)
                results['steps'].append(global_step)
                
                # CASMO Confidence Tracking
                if optimizer_name == 'casmo':
                    # Get average confidence across all groups
                    confs = []
                    for g_id, state in optimizer._group_states.items():
                        if 'current_confidence' in state:
                            confs.append(state['current_confidence'])
                    
                    avg_conf = np.mean(confs) if confs else 1.0
                    
                    # Heuristic: If >50% of the *accumulated* batch is noisy
                    avg_noise_ratio = np.mean(current_batch_noisy)
                    if avg_noise_ratio > 0.5:
                        results['noisy_confidences'].append(avg_conf)
                    else:
                        results['clean_confidences'].append(avg_conf)
                
                current_batch_noisy = [] # Reset
                progress_bar.set_postfix({'loss': f"{current_loss:.4f}"})
                global_step += 1
                
                if global_step >= 150: # Run enough steps to see post-calibration behavior
                    print("Step limit reached (150). Stopping.")
                    break
        
        if global_step >= 150:
            break
            
    return results

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1) # Optimized for 6GB VRAM (RTX 4050)
    parser.add_argument('--grad_accum_steps', type=int, default=32) # Maintain effective batch size of 32
    parser.add_argument('--noise', type=float, default=0.3)
    args = parser.parse_args()
    
    # Run CASMO
    casmo_res = run_training('casmo', epochs=args.epochs, batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps, noise_ratio=args.noise)
    
    # Run AdamW
    adam_res = run_training('adamw', epochs=args.epochs, batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps, noise_ratio=args.noise)
    
    # Plot
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(casmo_res['losses'], label='CASMO', alpha=0.7)
    plt.plot(adam_res['losses'], label='AdamW', alpha=0.7)
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.legend()
    
    # Confidence
    plt.subplot(1, 2, 2)
    if casmo_res['noisy_confidences']:
        plt.hist(casmo_res['clean_confidences'], alpha=0.5, label='Clean Batches', bins=20, color='green')
        plt.hist(casmo_res['noisy_confidences'], alpha=0.5, label='Noisy Batches', bins=20, color='red')
        plt.title('CASMO Confidence Distribution')
        plt.xlabel('Confidence')
        plt.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'instruct_results.png'))
    print(f"Results saved to {results_dir}")

if __name__ == '__main__':
    main()
