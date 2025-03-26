"""
Response generation utilities for character models.
"""

import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

class ResponseGenerator:
    """
    Generates responses from fine-tuned character models.
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        load_in_8bit: bool = False
    ):
        """
        Initialize the ResponseGenerator.
        
        Args:
            model_path: Path to the fine-tuned model.
            device: Device to load the model on (e.g., "cuda", "cpu").
            load_in_8bit: Whether to load the model in 8-bit precision.
        """
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Ensure the tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        if load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else None
            )
            
            if device != "cuda":
                self.model = self.model.to(device)
        
        self.device = device
    
    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Generate a response from the character model.
        
        Args:
            prompt: User prompt to generate a response for.
            system_prompt: Optional system prompt to prepend.
            max_length: Maximum length of the generated response.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            do_sample: Whether to use sampling for generation.
            
        Returns:
            Generated response from the character.
        """
        # Format the prompt
        if system_prompt:
            formatted_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
        else:
            formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        
        # Tokenize the prompt
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length + len(inputs["input_ids"][0]),
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
        
        # Decode the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        assistant_response = response.split("<|assistant|>\n")[-1].strip()
        
        return assistant_response