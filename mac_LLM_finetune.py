
import subprocess
from mlx_lm import load as load_mlx, generate as generate_mlx

from build_parser import build_parser
import utils as lora_utils

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from models import LoRALinear
import math
from lora import load, train, evaluate, loss_language_translation
from lora import generate_translation

parser = build_parser() # build parser  
args = parser.parse_args() # parse args

args.train = True
args.test = False
# args.wandb = False
args.inference = False
# hf_model_path = "mistralai/Mistral-7B-Instruct-v0.2"

# load quantized model 
# load from local repo
model_path = "/Volumes/FF952/mistral_finetune/mlx-base-new"
model, tokenizer = load_mlx(model_path)

def print_trainable_params(params, prefix=""):
    if isinstance(params, dict):
        for key, val in params.items():
            print_trainable_params(val, prefix=f"{prefix}.{key}" if prefix else key)
    elif isinstance(params, list):
        for idx, val in enumerate(params):
            print_trainable_params(val, prefix=f"{prefix}[{idx}]")
    elif isinstance(params, mx.array):  # mx.array from mlx.core
        print(f"{prefix}: shape={params.shape}, sum={mx.sum(params).item():.4f}, mean={mx.mean(params).item():.4f}")
    else:
        print(f"{prefix}: <non-mx.array value>")



# Building tokenizer_config
tokenizer_config = {}
if args.add_eos_token:
    tokenizer_config["add_eos_token"] = bool(args.add_eos_token)

# print("Loading pretrained model")
# model, tokenizer, _ = lora_utils.load(args.model, tokenizer_config)

# Freeze all layers other than LORA linears
model.freeze()
for l in model.model.layers[len(model.model.layers) - args.lora_layers :]:
    l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj, rank=8) #changed rank to 4
    l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj, rank=8) #changed rank to 4
    if hasattr(l, "block_sparse_moe"):
        l.block_sparse_moe.gate = LoRALinear.from_linear(l.block_sparse_moe.gate)

p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
print(f"Total parameters {p:.3f}M")
p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
print(f"Trainable parameters {p:.3f}M")

print("Loading datasets")
train_set, valid_set, test_set = load(args)

# Resume training the given adapters.
if args.resume_adapter_file is not None:
    print(f"Loading pretrained adapters from {args.resume_adapter_file}")
    model.load_weights(args.resume_adapter_file, strict=False)

if args.wandb:
    import wandb
    import os
    from dotenv import load_dotenv
    load_dotenv()
    wandb_key = os.getenv("WANDB_API_KEY")
    if not wandb_key:
        raise ValueError("WANDB_API_KEY is not set. Please set it in your .env file")
    # os.environ["WANDB_API_KEY"] = wandb_key

    wandb.login(key=wandb_key)
    run = wandb.init(
        project="mistral_finetune",
        name="language_translation_en_mai",
        config=args,
    )


if args.train:
    print("Training")
    opt = optim.Adam(learning_rate=args.learning_rate)
    # Train model
    if args.wandb:
        train(model, train_set, valid_set, opt, loss_language_translation, tokenizer, args, run)
    else:
        train(model, train_set, valid_set, opt, loss_language_translation, tokenizer, args)

    # Save adapter weights
    mx.savez(args.adapter_file, **dict(tree_flatten(model.trainable_parameters())))

    # test
    print("Testing")


if args.test:
    model.load_weights(args.adapter_file, strict=False)
    print("Testing")
    model.eval()
    test_loss = evaluate(
        model,
        test_set,
        loss,
        tokenizer,
        args.batch_size,
        num_batches=args.test_batches,
    )
    test_ppl = math.exp(test_loss)
    print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")

    # if args.prompt is not None:
    #     print("Generating")
    #     generate(model, args.prompt, tokenizer, args)


if args.inference: 
    args.adapter_file = "/Volumes/FF952/mistral_finetune/checkpoints_language_translation/adapters_200.npz"
    model.load_weights(args.adapter_file, strict=False)

    # Use this to inspect all trainable parameters
    # print("Trainable parameter values:")
    # trainables = model.trainable_parameters()
    # print_trainable_params(trainables)

    print("Original Model")
    model.eval()
    
    print("Generating")
    # load test sequence: 

    for example in train_set:
        en = example["en"]
        mai = example["mai"]

        prompt = f"Translate English to Maithili:\nEnglish: {en}\nMaithili:"
        # full_text = prompt + " " + mai

        # output = generate_mlx(model, tokenizer, prompt, max_tokens=200)
        # print(output)

        pred = generate_translation(prompt, model, tokenizer, args, max_new_tokens = 50)

        print(f"English: {en}")
        print(f"Maithili: {mai}")
        print(f"Prediction: {pred}")
        print("-" * 10)
