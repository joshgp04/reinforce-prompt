"""Evaluation: run a model on GSM8K test set and compute accuracy."""

import argparse
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import GSM8KDataset, check_answer
from model import PromptMixingModel
from prompts import K


def evaluate(model: PromptMixingModel, split: str = "test", batch_size: int = 4,
             mode: str = "fixed", prompt_idx: int = 0, max_samples: int = -1) -> dict:
    """
    Evaluate the model on GSM8K.

    Args:
        mode: "fixed" (single prompt), "supervised", or "rl"
        prompt_idx: which prompt to use in fixed mode
        max_samples: limit number of samples (-1 for all)
    """
    dataset = GSM8KDataset(split=split)
    if max_samples > 0:
        dataset.questions = dataset.questions[:max_samples]
        dataset.answers = dataset.answers[:max_samples]

    correct = 0
    total = 0
    results = []

    model.mixer.eval()

    pbar = tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating ({mode})")
    for i in pbar:
        batch_items = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        questions = [item["question"] for item in batch_items]
        gold_answers = [item["answer"] for item in batch_items]

        encoded = model.tokenizer(
            questions, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(model.device)

        with torch.no_grad():
            if mode == "fixed":
                # Uniform zero weights except for the chosen prompt
                alpha = torch.zeros(len(questions), K, device=model.device)
                alpha[:, prompt_idx] = 1.0
            else:
                # Use the learned mixer
                pooled = model.get_pooled_input(encoded.input_ids, encoded.attention_mask)
                alpha = model.mixer(pooled)

            predictions = model.generate(
                encoded.input_ids, encoded.attention_mask, alpha, max_new_tokens=512
            )

        for pred, gold, q, al in zip(predictions, gold_answers, questions, alpha):
            is_correct = check_answer(pred, gold)
            correct += int(is_correct)
            total += 1
            results.append({
                "question": q,
                "gold": gold,
                "predicted": pred,
                "correct": is_correct,
                "weights": al.cpu().tolist(),
            })
        pbar.set_postfix(accuracy=f"{correct/total:.4f}")

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fixed", "supervised", "rl"], default="fixed")
    parser.add_argument("--prompt_idx", type=int, default=0, help="Prompt index for fixed mode")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to mixer checkpoint")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--output", type=str, default="results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PromptMixingModel(model_name=args.model_name, device=device)

    if args.checkpoint:
        model.mixer.load_state_dict(torch.load(args.checkpoint, map_location=device))

    result = evaluate(model, mode=args.mode, prompt_idx=args.prompt_idx,
                      batch_size=args.batch_size, max_samples=args.max_samples)

    print(f"\nAccuracy: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
