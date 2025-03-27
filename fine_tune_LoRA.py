

# to load model from local
# model = AutoModelForCausalLM.from_pretrained("/Users/megha/mistral-pytorch", load_in_4bit=True, device_map="auto")

# tokenizer = AutoTokenizer.from_pretrained("/Users/megha/mistral-pytorch")
from math import e
import os
import wandb
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from evaluate_callbacks import BLEUEvalCallback, ChrFEvalCallback
import torch 

from dotenv import load_dotenv

load_dotenv()

HuggingFace_Token = os.getenv("HuggingFace_Token")
Wandb_API_Key = os.getenv("Wandb_API_Key")

from huggingface_hub import login

login(HuggingFace_Token)

save_split      = True
load_from_disk  = True

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# LoRA Configuration
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Load model + tokenizer (from local directory)
# model = AutoModelForCausalLM.from_pretrained("/Volumes/FF952/mistral_finetune/base_model", ...)
# tokenizer = AutoTokenizer.from_pretrained("/Volumes/FF952/mistral_finetune/base_model", ...)

model_path = "/Volumes/FF952/mistral_finetune/base_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",          # Automatically maps to MPS or CPU
    load_in_4bit=True,
    torch_dtype="auto",         # Or torch.float16 for fp16
    trust_remote_code=True      # Required for some models like Mistral
)

# from HF hub
# model_id = "mistralai/Mistral-7B-v0.1"
# tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token

# Load model + tokenizer (from local directory)
model = AutoModelForCausalLM.from_pretrained(
    "/Volumes/FF952/mistral_finetune/base_model",
    torch_dtype=torch.float16,  # fallback if memory allows
    device_map="auto",
    trust_remote_code=True
)

model = get_peft_model(model, peft_config)


# Initialize wandb
wandb.login(key=Wandb_API_Key)
wandb.init(project="mistral_maithili-translation", name="mistral-lora-ft")

# Load and split dataset
raw_dataset = load_dataset("json", data_files={"data": "data/maithili_english.jsonl"})["data"]
dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
dataset = DatasetDict({"train": dataset["train"], "validation": dataset["test"]})
dataset["train"].to_json("./data/train.jsonl")
dataset["validation"].to_json("./data/val.jsonl")

# for reproducibility
if save_split:
    dataset.save_to_disk("./data/maithili_split_dataset")
else:
    if load_from_disk:
        dataset = load_dataset("./data/maithili_split_dataset")

# evaluation
chrf_callback = ChrFEvalCallback(tokenizer=tokenizer, eval_dataset=dataset["validation"], eval_steps=100)
bleu_callback = BLEUEvalCallback(tokenizer=tokenizer, eval_dataset=dataset["validation"], eval_steps=100)

# Prompt formatting
# Another option to Randomly flip the method with
def format_prompt(example):
    if "english" in example and "maithili" in example:
        # randomly flip direction
        if bool(hash(example["english"]) % 2):  # for variety and faster training
            prompt = f"### Instruction:\nTranslate to Maithili:\n{example['english']}\n\n### Response:\n{example['maithili']}"
        else:
            prompt = f"### Instruction:\nTranslate to English:\n{example['maithili']}\n\n### Response:\n{example['english']}"
        return {"text": prompt}
    return {}

dataset = dataset.map(format_prompt)

# Training Args
training_args = TrainingArguments(
    output_dir="/Volumes/FF952/mistral_finetune/checkpoints/mistral-maithili", 
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    evaluation_strategy="steps",
    save_strategy="steps",
    learning_rate=2e-4,
    fp16=True,
    logging_dir="./logs",
    report_to="wandb",
    run_name="mistral-maithili-ft", 
    callbacks=[chrf_callback, bleu_callback]
)

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=training_args,
    max_seq_length=512,
    packing=False,
)

trainer.train()

# save model
model.save_pretrained("/Volumes/FF952/mistral_finetune/checkpoints/mistral-maithili")
