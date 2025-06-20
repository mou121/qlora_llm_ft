from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "qlora-mistral-merged"  # merged model directory
model_id = "your-username/qlora-mistral"  # HuggingFace repo

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

model.push_to_hub(model_id)
tokenizer.push_to_hub(model_id)

print(f"✅ Model pushed to HuggingFace Hub: https://huggingface.co/{model_id}")
