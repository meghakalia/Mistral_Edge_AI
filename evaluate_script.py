

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
import evaluate

# Set checkpoint path
checkpoint_path = "checkpoints/mistral-maithili/checkpoint-200"

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, device_map="auto")
model.eval()

# Load validation data
val_data = load_dataset("json", data_files="data/val.jsonl")["train"]  # load as "train" split

# Evaluation metric
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")

predictions = []
references = []

# Generate translations
for example in val_data:
    input_text = f"### Instruction:\nTranslate to Maithili:\n{example['english']}\n\n### Response:\n"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process prediction
    prediction = prediction.split("### Response:")[-1].strip()

    predictions.append(prediction)
    references.append(example["maithili"])

# Evaluate BLEU and chrF
bleu_score = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
chrf_score = chrf.compute(predictions=predictions, references=references)

print("🔍 Evaluation from checkpoint:", checkpoint_path)
print("🟢 BLEU Score:", bleu_score["score"])
print("🟢 chrF Score:", chrf_score["score"])
