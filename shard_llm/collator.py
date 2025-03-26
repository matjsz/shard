"""
Custom data collators for character fine-tuning.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch
from transformers import AutoTokenizer

@dataclass
class CharacterDataCollator:
    """
    Custom data collator for character fine-tuning that properly handles padding.
    """
    
    tokenizer: AutoTokenizer
    mlm: bool = False
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, examples: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Collate function to create batches from examples.
        
        Args:
            examples: List of examples from the dataset.
            
        Returns:
            Batch dictionary with tensors.
        """
        # Handle dict or BatchEncoding objects
        batch = {}
        
        # Special handling for input_ids
        if "input_ids" in examples[0]:
            input_ids = [example["input_ids"] for example in examples]
            # Find max length in this batch
            max_length = max(len(ids) for ids in input_ids)
            
            # Pad all sequences to max_length
            padded_input_ids = []
            attention_mask = []
            for ids in input_ids:
                padding_length = max_length - len(ids)
                padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * padding_length)
                attention_mask.append([1] * len(ids) + [0] * padding_length)
            
            batch["input_ids"] = torch.tensor(padded_input_ids)
            batch["attention_mask"] = torch.tensor(attention_mask)
            
            # For causal language modeling, labels = input_ids
            if not self.mlm:
                batch["labels"] = batch["input_ids"].clone()
        
        return batch