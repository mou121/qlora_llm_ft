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
# Helper: Plot loss curve
# --------------------------------------------------------------------------- #
def plot_loss(history, output_dir="./plots", filename="loss_chart.png"):
    """
    Plot training loss from the Trainer's log history.

    Parameters
    ----------
    history : list[dict]
        List of log entries from Trainer.state.log_history.
    output_dir : str
        Directory where the plot will be saved.
    filename : str
        Name of the output image file.
    """
    # Extract step and loss values
    steps = [entry["step"] for entry in history if "loss" in entry]
    losses = [entry["loss"] for entry in history if "loss" in entry]

    if not steps or not losses:
        # Nothing to plot
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(steps, losses, label="Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()


# --------------------------------------------------------------------------- #
# Model & Dataset Preparation
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Training Loop with Loss Plotting
# --------------------------------------------------------------------------- #
try:
    trainer.train()
except Exception as exc:
    # Log the exception if needed; training will be interrupted
    print(f"Training interrupted: {exc}")
finally:
    # Always attempt to plot whatever history is available
    history = trainer.state.log_history if hasattr(trainer.state, "log_history") else []
    plot_loss(history)
    print("Loss chart saved to ./plots/loss_chart.png")