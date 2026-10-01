import os
import logging
from typing import Dict, Any

import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "qlora-mistral-output"


def _check_alpaca_schema(example: Dict[str, Any]) -> bool:
    required_keys = {"instruction", "input", "output"}
    return required_keys.issubset(example.keys())


def plot_loss(trainer, output_path: str = "./plots/loss_chart.png"):
    loss_values = [
        entry.get("loss") for entry in trainer.state.log_history if "loss" in entry
    ]

    if not loss_values:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(loss_values, label="Training Loss", color="tab:blue")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    dataset_path = "./dataset/alpaca_data.json"
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    raw_dataset = load_dataset("json", data_files=dataset_path, split="train")
    if not all(_check_alpaca_schema(ex) for ex in raw_dataset):
        raise ValueError("Dataset entries must contain 'instruction', 'input', and 'output' keys.")
    logger.info("Dataset loaded and verified.")

    # Small CPU/MPS-friendly model for local testing (no CUDA/bitsandbytes required).
    # For real QLoRA fine-tuning on Mistral-7B, run train.py on a CUDA GPU (e.g. Colab)
    # and set MODEL_NAME=mistralai/Mistral-7B-v0.1 with load_in_4bit=True there instead.
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    def format_example(example):
        prompt = f"### Instruction:\n{example['instruction']}\n"
        if example.get("input"):
            prompt += f"### Input:\n{example['input']}\n"
        prompt += f"### Response:\n{example['output']}"
        return {"text": prompt}

    dataset = raw_dataset.map(format_example)

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, max_length=512)

    tokenized = dataset.map(tokenize, batched=False, remove_columns=dataset.column_names)

    base_model = AutoModelForCausalLM.from_pretrained(model_name)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, peft_config)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_ratio=0.03,
        logging_dir="logs",
        num_train_epochs=3,
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=10,
        learning_rate=2e-4,
        report_to="none",
        lr_scheduler_type="cosine",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    plot_loss(trainer, "./plots/loss_chart.png")


if __name__ == "__main__":
    main()
