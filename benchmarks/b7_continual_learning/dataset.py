"""
Dataset preparation for B7 Continual Learning Benchmark.

Provides 4 distinct tasks for sequential fine-tuning:
1. Math (GSM8K)
2. Code (MBPP)
3. Commonsense QA
4. Creative Writing

Each task has 500 training and 100 test examples in instruction format.
"""

from datasets import load_dataset
from torch.utils.data import Dataset
import random
import re


class ContinualLearningDataset(Dataset):
    """
    Dataset for continual learning across 4 tasks.
    
    Each task is formatted as instruction-response pairs.
    """
    
    def __init__(self, tokenizer, task_id, split='train', max_samples=500, max_length=512, seed=42):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            task_id: 0=Math, 1=Code, 2=QA, 3=Writing
            split: 'train' or 'test'
            max_samples: Maximum samples to load
            max_length: Maximum sequence length
            seed: Random seed
        """
        self.tokenizer = tokenizer
        self.task_id = task_id
        self.split = split
        self.max_length = max_length
        
        random.seed(seed)
        
        # Load task-specific data
        if task_id == 0:
            self.data = self._load_math(max_samples, split)
            self.task_name = "Mathematics"
        elif task_id == 1:
            self.data = self._load_code(max_samples, split)
            self.task_name = "Code Generation"
        elif task_id == 2:
            self.data = self._load_qa(max_samples, split)
            self.task_name = "Commonsense QA"
        elif task_id == 3:
            self.data = self._load_writing(max_samples, split)
            self.task_name = "Creative Writing"
        else:
            raise ValueError(f"Invalid task_id: {task_id}")
        
        print(f"Loaded Task {task_id} ({self.task_name}): {len(self.data)} {split} samples")
    
    def _load_math(self, max_samples, split):
        """Load GSM8K math problems."""
        try:
            # Load GSM8K dataset
            dataset = load_dataset("gsm8k", "main", split="train")
            
            # Split into train/test
            if split == 'train':
                samples = dataset.select(range(min(max_samples, len(dataset))))
            else:
                # Use last 100 for test
                start_idx = min(max_samples, len(dataset))
                samples = dataset.select(range(start_idx, min(start_idx + 100, len(dataset))))
            
            data = []
            for item in samples:
                instruction = f"Solve this math problem step by step:\n{item['question']}"
                response = item['answer']
                data.append({'instruction': instruction, 'response': response})
            
            return data
        except:
            # Fallback: synthetic math problems
            print("Warning: Could not load GSM8K, using synthetic math data")
            return self._create_synthetic_math(max_samples if split == 'train' else 100)
    
    def _create_synthetic_math(self, count):
        """Create synthetic math problems as fallback."""
        data = []
        for i in range(count):
            a = random.randint(10, 100)
            b = random.randint(5, 50)
            op = random.choice(['+', '-', '*'])
            
            if op == '+':
                answer = a + b
                question = f"What is {a} + {b}?"
            elif op == '-':
                answer = a - b
                question = f"What is {a} - {b}?"
            else:
                answer = a * b
                question = f"What is {a} × {b}?"
            
            instruction = f"Solve this math problem:\n{question}"
            response = f"The answer is {answer}."
            data.append({'instruction': instruction, 'response': response})
        
        return data
    
    def _load_code(self, max_samples, split):
        """Load MBPP code generation tasks."""
        try:
            # Load MBPP dataset
            dataset = load_dataset("mbpp", "sanitized", split="train")
            
            if split == 'train':
                samples = dataset.select(range(min(max_samples, len(dataset))))
            else:
                start_idx = min(max_samples, len(dataset))
                samples = dataset.select(range(start_idx, min(start_idx + 100, len(dataset))))
            
            data = []
            for item in samples:
                instruction = f"Write a Python function:\n{item['text']}"
                response = item['code']
                data.append({'instruction': instruction, 'response': response})
            
            return data
        except:
            print("Warning: Could not load MBPP, using synthetic code data")
            return self._create_synthetic_code(max_samples if split == 'train' else 100)
    
    def _create_synthetic_code(self, count):
        """Create synthetic code problems as fallback."""
        templates = [
            ("Write a function that adds two numbers.", "def add(a, b):\n    return a + b"),
            ("Write a function that returns the maximum of two numbers.", "def maximum(a, b):\n    return max(a, b)"),
            ("Write a function that checks if a number is even.", "def is_even(n):\n    return n % 2 == 0"),
            ("Write a function that returns the length of a list.", "def list_length(lst):\n    return len(lst)"),
            ("Write a function that reverses a string.", "def reverse_string(s):\n    return s[::-1]"),
        ]
        
        data = []
        for i in range(count):
            template = templates[i % len(templates)]
            data.append({'instruction': template[0], 'response': template[1]})
        
        return data
    
    def _load_qa(self, max_samples, split):
        """Load CommonsenseQA."""
        try:
            dataset = load_dataset("commonsense_qa", split="train")
            
            if split == 'train':
                samples = dataset.select(range(min(max_samples, len(dataset))))
            else:
                start_idx = min(max_samples, len(dataset))
                samples = dataset.select(range(start_idx, min(start_idx + 100, len(dataset))))
            
            data = []
            for item in samples:
                choices_text = "\n".join([f"{label}) {text}" for label, text in 
                                         zip(item['choices']['label'], item['choices']['text'])])
                instruction = f"Answer this question:\n{item['question']}\n{choices_text}"
                response = f"The answer is {item['answerKey']}."
                data.append({'instruction': instruction, 'response': response})
            
            return data
        except:
            print("Warning: Could not load CommonsenseQA, using synthetic QA data")
            return self._create_synthetic_qa(max_samples if split == 'train' else 100)
    
    def _create_synthetic_qa(self, count):
        """Create synthetic QA as fallback."""
        questions = [
            ("What color is the sky?", "A) Red", "B) Blue", "C) Green", "B"),
            ("Where do fish live?", "A) Desert", "B) Water", "C) Sky", "B"),
            ("What do you use to write?", "A) Pen", "B) Shoe", "C) Car", "A"),
            ("When do you sleep?", "A) Morning", "B) Night", "C) Noon", "B"),
            ("What is ice made of?", "A) Fire", "B) Water", "C) Metal", "B"),
        ]
        
        data = []
        for i in range(count):
            q = questions[i % len(questions)]
            instruction = f"Answer this question:\n{q[0]}\n{q[1]}\n{q[2]}\n{q[3]}"
            response = f"The answer is {q[4]}."
            data.append({'instruction': instruction, 'response': response})
        
        return data
    
    def _load_writing(self, max_samples, split):
        """Load ROCStories for creative writing."""
        try:
            dataset = load_dataset("roostories", split="train")
            
            if split == 'train':
                samples = dataset.select(range(min(max_samples, len(dataset))))
            else:
                start_idx = min(max_samples, len(dataset))
                samples = dataset.select(range(start_idx, min(start_idx + 100, len(dataset))))
            
            data = []
            for item in samples:
                # Use first sentence as prompt, rest as completion
                sentences = item['storytitle']
                instruction = f"Continue this story:\n{sentences}"
                response = " ".join([item[f'sentence{i}'] for i in range(1, 6)])
                data.append({'instruction': instruction, 'response': response})
            
            return data
        except:
            print("Warning: Could not load ROCStories, using synthetic creative writing data")
            return self._create_synthetic_writing(max_samples if split == 'train' else 100)
    
    def _create_synthetic_writing(self, count):
        """Create synthetic creative writing prompts."""
        prompts = [
            ("The old house creaked...", "as Sarah stepped inside. Dust covered every surface, and cobwebs hung from the corners. She knew she shouldn't be there, but curiosity drove her forward."),
            ("The spaceship landed...", "in the middle of the desert. The alien emerged slowly, its eyes scanning the horizon. This was the moment humanity had been waiting for."),
            ("She opened the letter...", "with trembling hands. The words inside would change everything. After all these years, the truth was finally revealed."),
            ("The dragon roared...", "sending flames across the valley. The knight raised his shield, knowing this would be his greatest challenge yet."),
            ("On her birthday...", "something unexpected happened. A mysterious package arrived with no return address. Inside was a key to a door she'd never seen before."),
        ]
        
        data = []
        for i in range(count):
            prompt = prompts[i % len(prompts)]
            data.append({'instruction': f"Continue this story:\n{prompt[0]}", 'response': prompt[1]})
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format as instruction-following
        prompt = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}"
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Labels are the same as input_ids for causal LM
        labels = input_ids.clone()
        
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'task_id': self.task_id
        }
