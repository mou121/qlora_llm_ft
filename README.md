# QLoRA Fine-Tuning on Mistral-7B

This project demonstrates how to fine-tune Mistral-7B with QLoRA using a custom Alpaca-style dataset.

## 📚 Dataset Format

Located in `./dataset/alpaca_data.json`, each entry must include:
```json
{
  "instruction": "...",
  "input": "...", 
  "output": "..."
}
```

## 🚀 Setup
```bash
pip install -r requirements.txt
```

## 🏋️ Train
```bash
python train.py
```

## 🧪 Inference
```bash
python infer.py
```

## 💻 Run on Colab

You can upload this repo as a zip and unzip in Colab:
```python
!pip install transformers peft bitsandbytes datasets accelerate
!unzip llm-finetuning-qlora.zip -d qlora
%cd qlora
!python train.py
```

> 💡 Tip: Use `transformers.Trainer` config inside `train.py` to adjust epochs, batch size, etc.


---

## ☁️ SageMaker Deployment

```bash
export SAGEMAKER_ROLE=your-execution-role
export S3_BUCKET=your-s3-bucket
python merge_adapters.py  # Merges LoRA weights
tar -czvf model.tar.gz qlora-mistral-merged/
aws s3 cp model.tar.gz s3://$S3_BUCKET/qlora-mistral-model/
python sagemaker_deploy.py
```

---

## 🧩 Merge Adapters into Base Model

```bash
python merge_adapters.py
```

This will save a full model you can use directly without PEFT.

---

## 🚀 Push to HuggingFace Hub

```bash
python push_to_hub.py
```

Make sure you're logged in with `huggingface-cli login`.

