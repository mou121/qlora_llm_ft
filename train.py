import os
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


def main():
    model_name = "mistralai/Mistral-7B-v0.1"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files={"train": "./dataset/alpaca_data.json"})

    def format_example(example):
        prompt = f"### Instruction:\n{example['instruction']}\n"
        if example.get("input"):
            prompt += f"### Input:\n{example['input']}\n"
        prompt += f"### Response:\n{example['output']}"
        return {"text": prompt}

    dataset = dataset.map(format_example)

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    tokenized = dataset["train"].map(tokenize, batched=False)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Load the base model with 4-bit quantization
    base_model = AutoModelForCausalLM.from_pretrained(
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
    model = get_peft_model(base_model, peft_config)

    training_args = TrainingArguments(
        output_dir="qlora-mistral-output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        warmup_ratio=0.03,
        logging_dir="logs",
        num_train_epochs=3,
        save_strategy="epoch",
        save_total_limit=1,          # Keep only the most recent checkpoint
        load_best_model_at_end=True, # Load the best model at the end of training
        logging_steps=10,
        learning_rate=2e-4,
        report_to="none",
        lr_scheduler_type="cosine",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model("qlora-mistral-output")
    plot_loss(trainer, "./plots/loss_chart.png")


if __name__ == "__main__":
    main()