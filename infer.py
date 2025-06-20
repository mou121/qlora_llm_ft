from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

peft_model_id = "output_dir"

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

print(pipe("Explain QLoRA in simple terms:", max_new_tokens=100)[0]["generated_text"])
