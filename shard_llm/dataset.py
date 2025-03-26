"""
Dataset preparation utilities for character fine-tuning.
"""

import json
import os
from typing import Dict, List, Optional, Union
from datasets import Dataset
from transformers import AutoTokenizer

class CharacterDataset:
    """
    Handles dataset preparation for character fine-tuning.
    
    This class provides utilities to create, process, and format datasets
    for fine-tuning language models to mimic specific characters.
    """
    
    def __init__(self, tokenizer: AutoTokenizer):
        """
        Initialize the CharacterDataset.
        
        Args:
            tokenizer: The tokenizer to use for encoding the dataset.
        """
        self.tokenizer = tokenizer
        
        # Ensure the tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Set pad_token to eos_token")
    
    def create_jsonl_dataset(
        self, 
        character_name: str, 
        character_description: str, 
        examples: List[Dict[str, str]], 
        output_file: str
    ) -> str:
        """
        Create a JSONL file with character examples for fine-tuning.
        
        Args:
            character_name: Name of the character.
            character_description: Description of the character's traits and style.
            examples: List of dictionaries with 'user' and 'assistant' keys.
            output_file: Path to save the JSONL file.
            
        Returns:
            Path to the created JSONL file.
        """
        dataset = []
        
        for example in examples:
            entry = {
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are {character_name}, who {character_description}."
                    },
                    {
                        "role": "user",
                        "content": example["user"]
                    },
                    {
                        "role": "assistant",
                        "content": example["assistant"]
                    }
                ]
            }
            dataset.append(entry)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Write to JSONL file
        with open(output_file, 'w') as f:
            for entry in dataset:
                f.write(json.dumps(entry) + '\n')
        
        print(f"Created dataset with {len(dataset)} examples at {output_file}")
        return output_file
    
    def load_jsonl_dataset(self, jsonl_file: str) -> List[Dict]:
        """
        Load a JSONL dataset file.
        
        Args:
            jsonl_file: Path to the JSONL file.
            
        Returns:
            List of dictionaries containing the dataset entries.
        """
        raw_data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                raw_data.append(json.loads(line))
        return raw_data
    
    def format_for_training(self, jsonl_file: str, max_length: int = 512) -> Dataset:
        """
        Convert a JSONL file to a format suitable for training.
        
        Args:
            jsonl_file: Path to the JSONL file.
            max_length: Maximum sequence length for tokenization.
            
        Returns:
            A Hugging Face Dataset ready for training.
        """
        # Load the JSONL file
        raw_data = self.load_jsonl_dataset(jsonl_file)
        
        # Process the data into text strings
        texts = []
        for item in raw_data:
            messages = item["messages"]
            
            # Format as a conversation
            formatted_text = ""
            for message in messages:
                if message["role"] == "system":
                    formatted_text += f"<|system|>\n{message['content']}\n"
                elif message["role"] == "user":
                    formatted_text += f"<|user|>\n{message['content']}\n"
                elif message["role"] == "assistant":
                    formatted_text += f"<|assistant|>\n{message['content']}\n"
            
            texts.append(formatted_text)
        
        # Tokenize all texts - don't convert to tensors yet
        input_ids_list = []
        for text in texts:
            input_ids = self.tokenizer.encode(
                text, 
                add_special_tokens=True, 
                truncation=True, 
                max_length=max_length
            )
            input_ids_list.append(input_ids)
        
        # Create a dataset with just input_ids
        dataset_dict = {
            "input_ids": input_ids_list
        }
        
        # Create a Hugging Face dataset
        dataset = Dataset.from_dict(dataset_dict)
        
        return dataset