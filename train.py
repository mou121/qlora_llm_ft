import os
import json
import logging
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# Optional WandB import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _check_alpaca_schema(example: Dict[str, Any]) -> bool:
    required_keys = {"instruction", "input", "output"}
    return required_keys.issubset(example.keys())

def _preprocess(example: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Output:\n"
    full_text = prompt + output
    tokenized = tokenizer(full_text, truncation=True, max_length=tokenizer.model_max_length)
    return tokenized

def main() -> None:
    # Load dataset
    dataset_path = "./dataset/alpaca_data.json"
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    raw_dataset = load_dataset("json", data_files=dataset_path, split="train")
    if not all(_check_alpaca_schema(ex) for ex in raw_dataset):
        raise ValueError("Dataset entries must contain 'instruction', 'input', and 'output' keys.")
    logger.info("Dataset loaded and verified.")

    # Tokenizer and model
    model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,