import random
from datasets import load_dataset
import torch
from torch.utils.data import Dataset

class NoisyInstructDataset(Dataset):
    """
    Dataset for Noisy Instruction Following.
    Loads UltraFeedback and injects noise by flipping 'chosen' and 'rejected' responses.
    """
    def __init__(self, tokenizer, split='train', max_length=512, noise_ratio=0.3, seed=42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.noise_ratio = noise_ratio
        
        random.seed(seed)
        
        # Load UltraFeedback (binarized preferences)
        # Using a smaller subset or a compatible version if needed. 
        # 'HuggingFaceH4/ultrafeedback_binarized' is a good pre-processed version.
        print(f"Loading dataset: HuggingFaceH4/ultrafeedback_binarized [{split}]...")
        try:
            self.dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=f"{split}_sft[:10000]") # Limit to 10k for speed as per plan
        except Exception as e:
            print(f"Error loading dataset: {e}. Fallback to local mock if needed (not implemented).")
            raise e
            
        self.data = []
        self.process_data()

    def process_data(self):
        print(f"Processing {len(self.dataset)} samples with {self.noise_ratio*100}% noise...")
        
        for i, item in enumerate(self.dataset):
            # Format: messages usually contain list of dicts [{'role': 'user', ...}, {'role': 'assistant', ...}]
            # In binarized version, we often have 'chosen' and 'rejected' lists of messages.
            
            prompt = item['prompt']
            chosen = item['chosen'][-1]['content'] # Last message is assistant response
            rejected = item['rejected'][-1]['content']
            
            is_noisy = False
            
            # Noise Injection
            if random.random() < self.noise_ratio:
                is_noisy = True
                # Flip: The "target" becomes the bad response
                target_response = rejected
            else:
                target_response = chosen
                
            # Construct input text for Causal LM
            # Format: "### Instruction: ... ### Response: ..." or ChatML
            # Phi-2 is flexible, let's use a simple format.
            text = f"Instruct: {prompt}\nOutput: {target_response}"
            
            self.data.append({
                'text': text,
                'is_noisy': is_noisy
            })
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Labels are same as input (standard Causal LM training)
        # We could mask the instruction part, but for simple benchmark, full sequence loss is fine.
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'is_noisy': torch.tensor(item['is_noisy'], dtype=torch.long)
        }
