"""Validation step: measure per-prompt accuracy by forcing alpha one-hot.

For each prompt in the bank, run the model with all weight on that single
prompt (no mixing) and measure GSM8K accuracy. The accuracy drop from the
helpful baseline gives a "harm score" for each adversarial prompt.

Run with:
    USE_ADVERSARIAL_BANK=1 python validation.py --n_samples 50
"""

import argparse
import json
import os
import torch

from data import GSM8KDataset, check_answer
from model import PromptMixingModel
from prompts import HELPFUL_INDICES, ADVERSARIAL_INDICES, K


@torch.no_grad()
def eval_with_forced_alpha(model, dataset, prompt_idx: int, n_samples: int = 50,
                            batch_size: int = 32) -> float:
    """Run model with alpha forced to a one-hot vector at prompt_idx."""
    n = min(n_samples, len(dataset))
    correct = 0

    for i in range(0, n, batch_size):
        batch_items = [dataset[j] for j in range(i, min(i + batch_size, n))]
        questions = [item["question"] for item in batch_items]
        gold_answers = [item["answer"] for item in batch_items]
        bs = len(questions)

        encoded = model.tokenizer(
            questions, return_tensors="pt", padding=True,
            truncation=True, max_length=512
        ).to(model.device)

        # Force alpha to one-hot at prompt_idx — bypass the mixer entirely
        alpha = torch.zeros(bs, K, device=model.device)
        alpha[:, prompt_idx] = 1.0

        predictions = model.generate(
            encoded.input_ids, encoded.attention_mask, alpha, max_new_tokens=256
        )
        for pred, gold in zip(predictions, gold_answers):
            if check_answer(pred, gold):
                correct += 1

    return correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_path", type=str, default="validation_results.json")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    args = parser.parse_args()

    if not ADVERSARIAL_INDICES:
        raise RuntimeError(
            "ADVERSARIAL_INDICES is empty. Set USE_ADVERSARIAL_BANK=1 before running."
        )

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model = PromptMixingModel(model_name=args.model_name, device=device)
    model.eval()

    dataset = GSM8KDataset(split=args.split)
    print(f"Loaded {len(dataset)} {args.split} examples; using {args.n_samples}")
    print(f"K={K}, helpful={len(HELPFUL_INDICES)}, adversarial={len(ADVERSARIAL_INDICES)}")
    print()

    results = {"helpful": {}, "adversarial": {}}

    print("=== Helpful prompts ===")
    for idx in HELPFUL_INDICES:
        acc = eval_with_forced_alpha(model, dataset, idx,
                                      n_samples=args.n_samples,
                                      batch_size=args.batch_size)
        results["helpful"][idx] = acc
        print(f"  Prompt {idx:2d}: acc = {acc:.3f}")

    helpful_accs = list(results["helpful"].values())
    baseline = sum(helpful_accs) / len(helpful_accs)
    print(f"\nHelpful baseline (mean): {baseline:.3f}")
    print(f"Helpful range: [{min(helpful_accs):.3f}, {max(helpful_accs):.3f}]")
    print()

    print("=== Adversarial prompts ===")
    for idx in ADVERSARIAL_INDICES:
        acc = eval_with_forced_alpha(model, dataset, idx,
                                      n_samples=args.n_samples,
                                      batch_size=args.batch_size)
        harm = baseline - acc
        results["adversarial"][idx] = {"accuracy": acc, "harm_score": harm}
        print(f"  Prompt {idx:2d}: acc = {acc:.3f}, harm = {harm:+.3f}")

    results["baseline"] = baseline
    results["n_samples"] = args.n_samples
    results["split"] = args.split

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.output_path}")


if __name__ == "__main__":
    main()