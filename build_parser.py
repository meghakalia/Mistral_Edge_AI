import argparse

def build_parser():
    parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
    parser.add_argument(
        "--num_iters",
        default=1000,
        help="Number of iterations to train for.",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Do inference",
    )
    parser.add_argument(
        "--steps-per-eval",
        default=10, 
        help="Number of training steps between validations.",
    )
    
    parser.add_argument(
        "--learning-rate",
        default=1e-5,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--model",
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    # Generation args
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=100,
        help="The maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp", type=float, default=0.8, help="The sampling temperature"
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        help="The prompt for generation",
        default="Translate English to Maithili:\nEnglish: What is your name?\nMaithili:"
    )

    parser.add_argument(
        "--val-batches",
        type=int,
        default=10,
        help="Number of validation batches, -1 uses the entire validation set.",
    )

    parser.add_argument(
        "--wandb",
        action="store_false",
        help="Log to wandb",
    )
    parser.add_argument(
        "--adapter-file",
        type=str,
        default="/Volumes/FF952/mistral_finetune/checkpoints_language_translation/adapters.npz",
        help="Save/load path for the trained adapter weights.",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save the model every N iterations.",
    )
    # Training args
    parser.add_argument(
        "--train",
        action="store_false",
        help="Do training",
    )
    parser.add_argument(
        "--add-eos-token",
        type=int,
        default=1,
        help="Enable add_eos_token for tokenizer",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/",
        help="Directory with {train, valid, test}.jsonl files",
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=16,
        help="Number of layers to fine-tune",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Minibatch size.")
    parser.add_argument(
        "--iters", type=int, default=1000, help="Iterations to train for."
    )
    parser.add_argument(
        "--steps-per-report",
        type=int,
        default=10,
        help="Number of training steps between loss reporting.",
    )
    parser.add_argument(
        "--resume-adapter-file",
        type=str,
        default=None,
        help="Load path to resume training with the given adapter weights.",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on the test set after training",
    )
    parser.add_argument(
        "--test-batches",
        type=int,
        default=500,
        help="Number of test set batches, -1 uses the entire test set.",
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    return parser

