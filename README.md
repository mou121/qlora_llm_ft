# QLoRA Fine-Tuning on Mistral-7B

This repo demonstrates how to fine-tune `mistralai/Mistral-7B-v0.1` using QLoRA with HuggingFace's PEFT.

## 🚀 Setup
```bash
pip install -r requirements.txt
```

## 🏋️ Training
```bash
python train.py
```

## 🧪 Inference
```bash
python infer.py
```

## 📁 Notes
- Uses 4-bit quantization (bitsandbytes)
- PEFT with LoRA adapters
- Modify `train.py` to add your dataset using `datasets.load_dataset()`
