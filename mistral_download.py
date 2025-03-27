

from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv

load_dotenv()

from huggingface_hub import login

HuggingFace_Token = os.getenv("HuggingFace_Token")

login(HuggingFace_Token)

model_id = "mistralai/Mistral-7B-v0.1"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained("/Volumes/FF952/mistral_finetune/base_model")
tokenizer.save_pretrained("/Volumes/FF952/mistral_finetune/base_model")
