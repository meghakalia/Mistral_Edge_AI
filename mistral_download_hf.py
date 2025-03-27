

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mistral-7B-v0.1"
save_path = "/Volumes/FF952/mistral_finetune/model"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
