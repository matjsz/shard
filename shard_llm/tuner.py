"""
Core fine-tuning functionality for character LLMs.
"""

import os
import torch
from typing import Dict, List, Optional, Union, Tuple
import logging

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

from dataset import CharacterDataset
from collator import CharacterDataCollator

# Set up logging
logger = logging.getLogger(__name__)

class CharacterTuner:
    """
    Fine-tunes language models to mimic specific characters using LoRA.
    """
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        quantization: str = "8bit",
        device_map: str = "auto"
    ):
        """
        Initialize the CharacterTuner.
        
        Args:
            model_name: Name or path of the base model to fine-tune.
            output_dir: Directory to save the fine-tuned model.
            quantization: Quantization method ("8bit", "4bit", or None).
            device_map: Device mapping strategy for model loading.
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.quantization = quantization
        self.device_map = device_map
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure the tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        # Initialize dataset handler
        self.dataset_handler = CharacterDataset(self.tokenizer)
        
        # Model will be loaded during fine-tuning
        self.model = None
    
    def _load_model(self):
        """
        Load the base model with appropriate quantization.
        """
        # Configure quantization
        if self.quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        elif self.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        else:
            quantization_config = None
        
        # Load model
        if quantization_config:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map=self.device_map,
                torch_dtype=torch.float16
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                torch_dtype=torch.float16
            )
        
        # Prepare model for k-bit training if quantized
        if self.quantization:
            self.model = prepare_model_for_kbit_training(self.model)
    
    def fine_tune(
        self,
        dataset_path: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        learning_rate: float = 2e-4,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 8,
        num_epochs: int = 3,
        max_length: int = 512,
        save_steps: int = 100,
        logging_steps: int = 10
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Fine-tune the model using LoRA.
        
        Args:
            dataset_path: Path to the JSONL dataset file.
            lora_r: LoRA attention dimension.
            lora_alpha: LoRA alpha parameter.
            lora_dropout: Dropout probability for LoRA layers.
            learning_rate: Learning rate for training.
            batch_size: Batch size for training.
            gradient_accumulation_steps: Number of steps to accumulate gradients.
            num_epochs: Number of training epochs.
            max_length: Maximum sequence length.
            save_steps: Save checkpoint every this many steps.
            logging_steps: Log metrics every this many steps.
            
        Returns:
            Tuple of (fine-tuned model, tokenizer)
        """
        # Load the model if not already loaded
        if self.model is None:
            self._load_model()
        
        # Format dataset for training
        dataset = self.dataset_handler.format_for_training(dataset_path, max_length)
        
        # Define LoRA configuration
        # For Llama models, target the correct modules
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention modules
            "gate_proj", "up_proj", "down_proj"      # MLP modules
        ]
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules
        )
        
        # Apply LoRA to the model
        model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            save_steps=save_steps,
            logging_steps=logging_steps,
            save_total_limit=3,
            remove_unused_columns=False,
            report_to="none",  # Disable wandb reporting
        )
        
        # Create custom data collator
        data_collator = CharacterDataCollator(tokenizer=self.tokenizer, mlm=False)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        # Train the model
        logger.info("Starting training...")
        trainer.train()
        
        # Save the trained model
        model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Model fine-tuned and saved to {self.output_dir}")
        
        return model, self.tokenizer
    
    def create_character_dataset(
        self,
        character_name: str,
        character_description: str,
        examples: List[Dict[str, str]],
        output_file: str
    ) -> str:
        """
        Create a character dataset in JSONL format.
        
        Args:
            character_name: Name of the character.
            character_description: Description of the character's traits and style.
            examples: List of dictionaries with 'user' and 'assistant' keys.
            output_file: Path to save the JSONL file.
            
        Returns:
            Path to the created JSONL file.
        """
        return self.dataset_handler.create_jsonl_dataset(
            character_name,
            character_description,
            examples,
            output_file
        )