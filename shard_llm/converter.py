"""
Model conversion utilities for Ollama compatibility.
"""

import os
import subprocess
import time
import logging
from typing import Optional

# Set up logging
logger = logging.getLogger(__name__)

class ModelConverter:
    """
    Converts fine-tuned models to formats compatible with Ollama.
    """
    
    @staticmethod
    def merge_lora_weights(
        base_model_name: str,
        lora_model_path: str,
        output_dir: str
    ) -> str:
        """
        Merge LoRA weights back into the base model.
        
        Args:
            base_model_name: Name or path of the base model.
            lora_model_path: Path to the LoRA adapter weights.
            output_dir: Directory to save the merged model.
            
        Returns:
            Path to the merged model.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        logger.info("Loading base model...")
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Loading LoRA adapter...")
        # Load LoRA model
        lora_model = PeftModel.from_pretrained(base_model, lora_model_path)
        
        logger.info("Merging weights...")
        # Merge weights
        merged_model = lora_model.merge_and_unload()
        
        logger.info(f"Saving merged model to {output_dir}...")
        # Save merged model
        os.makedirs(output_dir, exist_ok=True)
        merged_model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        
        return output_dir
    
    @staticmethod
    def create_ollama_modelfile(
        model_path: str,
        model_name: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Create an Ollama Modelfile for the model.
        
        Args:
            model_path: Path to the model (can be HF model or GGUF).
            model_name: Name for the Ollama model.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            system_prompt: Optional system prompt for the model.
            
        Returns:
            Path to the created Modelfile.
        """
        # Create modelfile content
        modelfile_content = f"""
FROM {model_path}
PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
PARAMETER stop "<|user|>"
PARAMETER stop "<|system|>"
"""
        
        # Add system prompt if provided
        if system_prompt:
            modelfile_content += f'SYSTEM "{system_prompt}"\n'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(f"{model_name}.modelfile")), exist_ok=True)
        
        # Write modelfile
        modelfile_path = f"{model_name}.modelfile"
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)
        
        logger.info(f"Created Ollama Modelfile at {modelfile_path}")
        logger.info(f"To import this model to Ollama, run: ollama create {model_name} -f {modelfile_path}")
        
        return modelfile_path
    
    @staticmethod
    def convert_to_gguf(
        model_path: str,
        output_path: str,
        quantization: str = "f16"
    ) -> Optional[str]:
        """
        Convert a Hugging Face model to GGUF format.
        
        Args:
            model_path: Path to the Hugging Face model.
            output_path: Path to save the GGUF model.
            quantization: Quantization method (f16, q4_k_m, etc.).
            
        Returns:
            Path to the GGUF model if successful, None otherwise.
        """
        try:
            # Check if llama.cpp is available
            if not os.path.exists("llama.cpp"):
                logger.info("Cloning llama.cpp repository...")
                subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git"], check=True)
            
            # Find the correct conversion script
            conversion_script = None
            possible_paths = [
                "llama.cpp/convert.py",
                "llama.cpp/convert_hf_to_gguf.py",
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    conversion_script = path
                    break
            
            if conversion_script is None:
                logger.error("Could not find conversion script in llama.cpp repository.")
                return None
            
            logger.info(f"Found conversion script at {conversion_script}")
            
            # Build llama.cpp if needed
            if not os.path.exists("llama.cpp/main"):
                logger.info("Building llama.cpp...")
                subprocess.run(["cd", "llama.cpp", "&&", "make"], shell=True, check=True)
            
            # Convert model to GGUF format
            convert_cmd = [
                "python3", conversion_script, 
                model_path,
                "--outfile", output_path,
                "--outtype", quantization
            ]
            
            subprocess.run(convert_cmd, check=True)
            logger.info(f"Model converted to GGUF format at {output_path}")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error converting model: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during conversion: {e}")
            return None