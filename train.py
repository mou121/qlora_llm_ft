import os
import threading
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, LoraConfig, TaskType
import bitsandbytes as bnb


def plot_loss(trainer, output_path: str = "./plots/loss_chart.png"):
    """
    Plot the loss curve from the trainer's log history and save it to the specified path.

    Parameters
    ----------
    trainer : Trainer
        The Hugging Face Trainer instance containing the training history.
    output_path : str, optional
        Path where the loss chart will be saved. Defaults to "./plots/loss_chart.png".
    """
    # Extract loss values from the trainer's log history
    loss_values = [
        entry.get("loss") for entry in trainer.state.log_history if "loss" in entry
    ]

    if not loss_values:
        # No loss data to plot
        return

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create the plot
    plt.figure(figsize=(8, 4))
    plt.plot(loss_values, label="Training Loss", color="tab:blue")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path)
    plt.close()


def main():
    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files={"train": "./dataset/alpaca_data.json"})

    def format(example):
        prompt = f"### Instruction:\n{example['instruction']}\n"
        if example.get("input"):
            prompt += f"### Input:\n{example['input']}\n"
        prompt += f"### Response:\n{example['output']}"
        return {"text": prompt}

    dataset = dataset.map(format)

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

    tokenized = dataset["train"].map(tokenize)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, load_in_4bit=True, device_map="auto"
    )