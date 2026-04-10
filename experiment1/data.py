"""GSM8K data loading and answer extraction utilities."""

import math
import re
from datasets import load_dataset
from torch.utils.data import Dataset


def extract_answer(text: str) -> str:
    """Extract the final numeric answer from a GSM8K solution string.

    GSM8K answers end with '#### <number>'.
    For model outputs, we look for the last number in the text.
    """
    # Try GSM8K ground-truth format first
    match = re.search(r"####\s*(.+)", text)
    if match:
        return _normalize_number(match.group(1).strip())

    # Fallback: find the last number in the text
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return _normalize_number(numbers[-1])
    return ""


def _normalize_number(s: str) -> str:
    """Remove commas and trailing .0 for comparison."""
    s = s.replace(",", "").strip()
    try:
        val = float(s)
        if not math.isfinite(val):
            return s
        if val == int(val):
            return str(int(val))
        return str(val)
    except (ValueError, OverflowError):
        return s


def check_answer(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer."""
    return extract_answer(predicted) == extract_answer(gold)


class GSM8KDataset(Dataset):
    """Wraps the GSM8K dataset for our pipeline."""

    def __init__(self, split: str = "train"):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        self.questions = ds["question"]
        self.answers = ds["answer"]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return {
            "question": self.questions[idx],
            "answer": self.answers[idx],
        }
