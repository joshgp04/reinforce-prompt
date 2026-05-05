import argparse
import json
import re

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize(x):
    x = x.replace(",", "").strip()
    if x.endswith(".0"):
        x = x[:-2]
    return x


def extract_gold(answer):
    if "####" in answer:
        answer = answer.split("####")[-1]
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", answer)
    return normalize(nums[-1]) if nums else ""


def extract_pred(text):
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return normalize(nums[-1]) if nums else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output", default="results/experiment4/math_no_prompt.json")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    ds = load_dataset("gsm8k", "main", split="test")
    if args.max_samples is not None:
        ds = ds.select(range(args.max_samples))

    correct = 0
    examples = []

    for start in tqdm(range(0, len(ds), args.batch_size), desc="Evaluating math model without prompt"):
        batch = ds[start:start + args.batch_size]
        prompts = [f"Question: {q}\nAnswer:" for q in batch["question"]]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for question, gold_answer, prompt, output in zip(
            batch["question"], batch["answer"], prompts, decoded
        ):
            pred_text = output[len(prompt):] if output.startswith(prompt) else output
            gold = extract_gold(gold_answer)
            pred = extract_pred(pred_text)
            is_correct = pred == gold
            correct += int(is_correct)

            examples.append({
                "question": question,
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "output": pred_text,
            })

    total = len(ds)
    result = {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "examples": examples,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"No-prompt math accuracy: {result['accuracy']:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
