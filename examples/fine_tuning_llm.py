"""
Fine-tuning LLMs with CASMO

This example demonstrates using CASMO for fine-tuning language models,
where CASMO's noise robustness helps with noisy or conflicting training data.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from casmo import CASMO


class SimpleTextDataset(Dataset):
    """Simple dataset for demonstration."""
    def __init__(self, texts, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.encodings['input_ids'][idx]
        }


def main():
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load a small model for demonstration (use larger models in practice)
    model_name = "gpt2"  # or "distilgpt2" for faster training
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    
    # Example training data (replace with your actual data)
    train_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "CASMO optimizer helps with noisy gradients.",
        "Fine-tuning language models requires careful hyperparameter selection.",
        "PyTorch is a popular deep learning framework.",
    ] * 20  # Repeat for demonstration
    
    # Create dataset and dataloader
    train_dataset = SimpleTextDataset(train_texts, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # Initialize CASMO optimizer
    num_epochs = 3
    total_steps = len(train_loader) * num_epochs
    
    optimizer = CASMO(
        model.parameters(),
        lr=5e-5,  # Lower learning rate for fine-tuning
        weight_decay=0.01,
        granularity='group',
        total_steps=total_steps
    )
    
    # Training loop
    print("\nStarting fine-tuning...")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} | Average Loss: {avg_loss:.4f}")
    
    print("\n✓ Fine-tuning complete!")
    
    # Generate sample text
    print("\nGenerating sample text...")
    model.eval()
    prompt = "Machine learning is"
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()
