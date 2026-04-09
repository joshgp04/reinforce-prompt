"""Evaluate all fixed discrete prompts on GSM8K test set."""

import argparse
import json
import os
import torch

from model import PromptMixingModel
from evaluate import evaluate
from prompts import K


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default="results/experiment1")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PromptMixingModel(model_name=args.model_name, device=device)

    best_acc = 0.0
    best_idx = 0
    fixed_results = {}

    for k in range(K):
        print(f"\nEvaluating prompt {k}...")
        result = evaluate(model, mode="fixed", prompt_idx=k,
                          batch_size=args.batch_size, max_samples=args.max_samples)
        fixed_results[f"prompt_{k}"] = result["accuracy"]
        print(f"  Prompt {k} accuracy: {result['accuracy']:.4f}")
        if result["accuracy"] > best_acc:
            best_acc = result["accuracy"]
            best_idx = k

    fixed_results["best_prompt_idx"] = best_idx
    fixed_results["best_accuracy"] = best_acc

    with open(os.path.join(args.output_dir, "fixed_results.json"), "w") as f:
        json.dump(fixed_results, f, indent=2)
    print(f"\nBest fixed prompt: {best_idx} with accuracy {best_acc:.4f}")


if __name__ == "__main__":
    main()
