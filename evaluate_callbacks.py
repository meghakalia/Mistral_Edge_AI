

from transformers import TrainerCallback
import evaluate
import torch

import wandb

class BLEUEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, eval_steps=100):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.eval_steps = eval_steps
        self.bleu_metric = evaluate.load("sacrebleu")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            print(f"\n🔍 Evaluating BLEU at step {state.global_step}...")

            model = kwargs["model"]
            model.eval()

            predictions = []
            references = []

            for example in self.eval_dataset.select(range(50)):  # small eval subset for speed
                input_text = f"### Instruction:\nTranslate to Maithili:\n{example['english']}\n\n### Response:\n"
                inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True).to(model.device)
                
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=50)
                pred = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                pred = pred.split("### Response:")[-1].strip()

                predictions.append(pred)
                references.append(example["maithili"])

            bleu = self.bleu_metric.compute(predictions=predictions, references=[[r] for r in references])
            print(f"BLEU @ step {state.global_step}: {bleu['score']:.2f}")
            wandb.log({"bleu": bleu["score"], "step": state.global_step})


class ChrFEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, eval_steps=100):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.eval_steps = eval_steps
        self.chrf_metric = evaluate.load("chrf")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            print(f"\n🔍 Evaluating chrF at step {state.global_step}...")

            model = kwargs["model"]
            model.eval()

            predictions = []
            references = []

            # Use a small subset of the validation set for speed
            for example in self.eval_dataset.select(range(50)):  
                input_text = f"### Instruction:\nTranslate to Maithili:\n{example['english']}\n\n### Response:\n"
                inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True).to(model.device)
                
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=100)
                pred = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                pred = pred.split("### Response:")[-1].strip()

                predictions.append(pred)
                references.append(example["maithili"])

            chrf_score = self.chrf_metric.compute(predictions=predictions, references=references)
            print(f"🟠 chrF @ step {state.global_step}: {chrf_score['score']:.2f}")
            wandb.log({"chrF": chrf_score["score"], "step": state.global_step})
