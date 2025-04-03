
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

from lora import load, loss, train

# hf_model_path = "mistralai/Mistral-7B-Instruct-v0.2"

# load quantized model 
# load from local repo
model_path = "/Volumes/FF952/mistral_finetune/mlx-base"
model, tokenizer = load_mlx(model_path)

parser = build_parser() # build parser  
args = parser.parse_args() # parse args

# Building tokenizer_config
tokenizer_config = {}
if args.add_eos_token:
    tokenizer_config["add_eos_token"] = bool(args.add_eos_token)

# print("Loading pretrained model")
# model, tokenizer, _ = lora_utils.load(args.model, tokenizer_config)

# Freeze all layers other than LORA linears
model.freeze()
for l in model.model.layers[len(model.model.layers) - args.lora_layers :]:
    l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj, rank=4) #changed rank to 4
    l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj, rank=4) #changed rank to 4
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
        name="using both eng and mai",
        config=args,
    )

if args.train:
    print("Training")
    opt = optim.Adam(learning_rate=args.learning_rate)
    # Train model
    if args.wandb:
        train(model, train_set, valid_set, opt, loss, tokenizer, args, run)
    else:
        train(model, train_set, valid_set, opt, loss, tokenizer, args)

    # Save adapter weights
    mx.savez(args.adapter_file, **dict(tree_flatten(model.trainable_parameters())))

    # test
    print("Testing")

# if args.test:
#         print("Testing")
#         model.eval()
#         test_loss = evaluate(
#             model,
#             test_set,
#             loss,
#             tokenizer,
#             args.batch_size,
#             num_batches=args.test_batches,
#         )
#         test_ppl = math.exp(test_loss)

#         print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")

#     if args.prompt is not None:
#         print("Generating")
#         generate(model, args.prompt, tokenizer, args)




