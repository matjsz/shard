<div align="center">
    <h1>Shard</h1>
    <img src="raw.githubusercontent.com/matjsz/shard/refs/heads/main/docs/logo.png" width="100rem" />
    <p>Make your own character LLM.</p>
    <div style="display: flex; justify-content: center; gap: .5rem;">
        <img src="https://img.shields.io/badge/3.12+-blue?style=plastic&label=python" />
        <img src="https://img.shields.io/badge/2.4+-orange?style=plastic&label=torch" />
        <img src="https://img.shields.io/badge/0.15+-yellow?style=plastic&label=peft" />
    </div>
    <div style="margin-top: 1rem;">
        <img src="https://img.shields.io/pypi/v/shard-llm?style=plastic&label=shard" />
    </div>
</div>

<div align="center" style="margin-top: 3rem">
A Python package for fine-tuning large language models to behave like specific characters using LoRA.
</div>

## Installation

1. Install it with pip:

```bash
pip install shard-llm
```

## Features

- Fine-tune LLMs to mimic specific characters
- Parameter-efficient training with LoRA
- Support for 4-bit and 8-bit quantization
- Easy conversion to Ollama-compatible formats
- Simple API for dataset creation and response generation

## Quick Start

```python
from shard_ai import CharacterTuner, ResponseGenerator

# Initialize the tuner
tuner = CharacterTuner(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    output_dir="./sherlock-llama-lora"
)

# Create a character dataset
examples = [
    {
        "user": "What do you think about this case?",
        "assistant": "The facts, as presented, suggest a crime of passion rather than premeditation. Elementary deduction, really."
    },
    {
        "user": "Can you help me find my missing watch?",
        "assistant": "Observe the slight indentation on your right wrist, indicating you've worn the watch consistently until very recently. Have you checked the pocket of the jacket you wore during yesterday's garden excursion?"
    }
]

dataset_path = tuner.create_character_dataset(
    character_name="Sherlock Holmes",
    character_description="is a brilliant detective with exceptional deductive reasoning skills. You speak in a formal, precise manner, often making keen observations about details others miss.",
    examples=examples,
    output_file="sherlock_dataset.jsonl"
)

# Fine-tune the model
model, tokenizer = tuner.fine_tune(
    dataset_path=dataset_path,
    batch_size=2,
    num_epochs=3
)

# Generate responses with the fine-tuned model
generator = ResponseGenerator("./sherlock-llama-lora")
# Integrated generator for testing
response = generator.generate_response("Tell me about a case you solved recently")
print(response)
```

## Converting to Ollama

```python
from shard_ai import ModelConverter

# Merge LoRA weights with base model
merged_model_path = ModelConverter.merge_lora_weights(
    base_model_name="meta-llama/Llama-3.1-8B-Instruct",
    lora_model_path="./sherlock-llama-lora",
    output_dir="./sherlock-llama-merged"
)

# Create Ollama modelfile
modelfile_path = ModelConverter.create_ollama_modelfile(
    model_path=merged_model_path,
    model_name="sherlock-llama",
    system_prompt="You are Sherlock Holmes, a brilliant detective with exceptional deductive reasoning skills."
)

# Optional: Convert to GGUF format
gguf_path = ModelConverter.convert_to_gguf(
    model_path=merged_model_path,
    output_path="./sherlock-llama.gguf",
    quantization="f16"
)
```