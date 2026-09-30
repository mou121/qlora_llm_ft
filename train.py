import os
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
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Helper function to plot loss curve
# --------------------------------------------------------------------------- #
def plot_loss(log_history, output_path):
    """
    Plot the loss curve from the training log history.

    Parameters
    ----------
    log_history : list[dict]
        List of log entries from Trainer.state.log_history.
    output_path : str
        Path where the PNG plot will be saved.
    """
    # Extract loss values and corresponding step numbers
    steps = []
    losses = []
    for entry in log_history:
        if "loss" in entry:
            steps.append(entry.get("step", len(steps)))
            losses.append(entry["loss"])

    if not losses:
        # No loss data to plot
        return

    # Create plots directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(steps, losses, label="Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# --------------------------------------------------------------------------- #
# Main training routine
# --------------------------------------------------------------------------- #
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
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)

    args = TrainingArguments(
        output_dir="qlora-mistral-output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        logging_dir="logs",
        num_train_epochs=3,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        learning_rate=2e-4,
        fp16=True,
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Run training with graceful degradation
    try:
        trainer.train()
    except Exception as e:
        # Log the exception if needed; continue to plot whatever data we have
        print(f"Training interrupted: {e}")
    finally:
        # Plot loss curve regardless of training outcome
        plot_loss(trainer.state.log_history, "./plots/loss_chart.png")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- #
# Unit test for plot_loss function
# --------------------------------------------------------------------------- #
# The following test can be placed in tests/test_plot.py
# It verifies that the plot is generated correctly when provided with dummy data.

import unittest
import tempfile
import shutil
import os

class TestPlotLoss(unittest.TestCase):
    def test_plot_generation(self):
        # Create dummy log history
        dummy_log = [
            {"step": 1, "loss": 2.5},
            {"step": 2, "loss": 2.0},
            {"step": 3, "loss": 1.5},
            {"step": 4, "loss": 1.0},
        ]

        # Use a temporary directory to avoid clutter
        temp_dir = tempfile.mkdtemp()
        try:
            output_path = os.path.join(temp_dir, "loss_chart.png")
            plot_loss(dummy_log, output_path)
            # Check that the file exists and is non-empty
            self.assertTrue(os.path.isfile(output_path))
            self.assertTrue(os.path.getsize(output_path) > 0)
        finally:
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    unittest.main()