import argparse
import json
import re
from tqdm import tqdm

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def extract_gold(answer):
    if "####" in answer:
        answer = answer.split("####")[-1]
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", answer)
    if not nums:
        return ""
    return normalize(nums[-1])


def extract_pred(text):
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not nums:
        return ""
    return normalize(nums[-1])


def normalize(x):
    x = x.replace(",", "").strip()
    if x.endswith(".0"):
        x = x[:-2]
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output", type=str, default="results/experiment3/math_no_prompt.json")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    ds = load_dataset("gsm8k", "main", split="test")
    if args.max_samples is not None:
        ds = ds.select(range(args.max_samples))

    correct = 0
    total = 0
    examples = []

    for start in tqdm(range(0, len(ds), args.batch_size), desc="Evaluating no-prompt math model"):
        batch = ds[start:start + args.batch_size]
        questions = batch["question"]
        answers = batch["answer"]

        prompts = [
            f"Question: {q}\nAnswer:"
            for q in questions
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for q, gold_text, pred_text in zip(questions, answers, decoded):
            gold = extract_gold(gold_text)
            pred = extract_pred(pred_text)
            is_correct = pred == gold
            correct += int(is_correct)
            total += 1

            if len(examples) < 25:
                examples.append({
                    "question": q,
                    "gold": gold,
                    "pred": pred,
                    "correct": is_correct,
                    "generation": pred_text
                })

    acc = correct / total

    result = {
        "model_name": args.model_name,
        "setting": "no_prompt",
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "examples": examples
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
