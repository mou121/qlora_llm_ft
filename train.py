import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType
import bitsandbytes as bnb

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

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
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
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
