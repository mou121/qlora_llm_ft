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


def plot_loss(history, output_path):
    """
    Plot the loss curve from the training history and save it to output_path.

    Parameters
    ----------
    history : list[dict]
        List of log history dictionaries from Trainer.state.log_history.
    output_path : str
        Path where the loss chart PNG will be saved.
    """
    losses = [entry.get("loss") for entry in history if "loss" in entry]
    if not losses:
        return

    plt.figure()
    plt.plot(losses, label="Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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

    try:
        trainer.train()
    except Exception as e:
        # Log the exception if needed; training will be interrupted.
        print(f"Training interrupted: {e}")
    finally:
        # Always attempt to plot whatever loss history is available.
        plot_loss(trainer.state.log_history, "./plots/loss_chart.png")


if __name__ == "__main__":
    main()
    
    
# test_plot_loss.py
import os
import tempfile
import pytest
from train import plot_loss


def test_plot_loss_creates_file(tmp_path):
    history = [{"loss": 2.0}, {"loss": 1.5}, {"loss": 1.0}]
    output = tmp_path / "loss.png"
    plot_loss(history, str(output))
    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_loss_no_history(tmp_path):
    history = []
    output = tmp_path / "loss.png"
    plot_loss(history, str(output))
    assert not output.exists()