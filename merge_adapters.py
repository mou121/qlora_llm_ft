from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = "mistralai/Mistral-7B-v0.1"
peft_model_path = "qlora-mistral-output/checkpoint-xxx"  # replace with final checkpoint dir

model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
model = PeftModel.from_pretrained(model, peft_model_path)
model = model.merge_and_unload()

save_path = "qlora-mistral-merged"
model.save_pretrained(save_path)
print(f"✅ Merged model saved at: {save_path}")
