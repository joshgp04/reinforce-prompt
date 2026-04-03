"""Evaluate RL prompt mixer on GSM8K test set."""

import argparse
import json
import os
import torch

from model import PromptMixingModel
from evaluate import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/rl_mixer.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--output", type=str, default="results/experiment1/rl_results.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PromptMixingModel(model_name=args.model_name, device=device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "mixer_state_dict" in ckpt:
        model.mixer.load_state_dict(ckpt["mixer_state_dict"])
    else:
        model.mixer.load_state_dict(ckpt)

    result = evaluate(model, mode="rl", batch_size=args.batch_size,
                      max_samples=args.max_samples)
    print(f"\nRL accuracy: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
