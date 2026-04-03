"""Visualize learned prompt-mixing weights from experiment results."""

import argparse
import json
import re
import numpy as np
import matplotlib.pyplot as plt

from prompts import PROMPT_BANK


# Heuristic problem-type classification based on question keywords
PROBLEM_TYPES = {
    "arithmetic": ["how many", "how much", "total", "sum", "add", "subtract", "difference"],
    "rate/ratio": ["per", "each", "every", "rate", "ratio", "speed", "miles per"],
    "fraction/percent": ["percent", "%", "fraction", "half", "third", "quarter"],
    "comparison": ["more than", "less than", "fewer", "greater", "difference between"],
    "multi-step": ["then", "after that", "first", "next", "finally", "remaining"],
}


def classify_problem_type(question: str) -> str:
    """Classify a GSM8K question by type using keyword heuristics."""
    q = question.lower()
    scores = {}
    for ptype, keywords in PROBLEM_TYPES.items():
        scores[ptype] = sum(1 for kw in keywords if kw in q)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


def estimate_difficulty(question: str) -> str:
    """Estimate difficulty by number of sentences (proxy for reasoning steps)."""
    sentences = [s.strip() for s in re.split(r'[.?!]', question) if s.strip()]
    n = len(sentences)
    if n <= 2:
        return "easy"
    elif n <= 4:
        return "medium"
    else:
        return "hard"


def plot_weight_distribution(results_path: str, title: str, save_path: str):
    """Plot average prompt weights and per-example weight heatmap."""
    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]
    weights = np.array([r["weights"] for r in results])  # (N, K)
    correct_mask = np.array([r["correct"] for r in results])

    K = weights.shape[1]
    prompt_labels = [f"P{i}" for i in range(K)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Average weights overall
    avg_weights = weights.mean(axis=0)
    axes[0].bar(prompt_labels, avg_weights)
    axes[0].set_title(f"{title}\nAverage Prompt Weights")
    axes[0].set_ylabel("Weight")

    # 2. Average weights for correct vs incorrect
    if correct_mask.sum() > 0:
        avg_correct = weights[correct_mask].mean(axis=0)
        axes[1].bar(np.arange(K) - 0.15, avg_correct, 0.3, label="Correct", color="green", alpha=0.7)
    if (~correct_mask).sum() > 0:
        avg_incorrect = weights[~correct_mask].mean(axis=0)
        axes[1].bar(np.arange(K) + 0.15, avg_incorrect, 0.3, label="Incorrect", color="red", alpha=0.7)
    axes[1].set_xticks(range(K))
    axes[1].set_xticklabels(prompt_labels)
    axes[1].set_title("Weights: Correct vs Incorrect")
    axes[1].legend()

    # 3. Weight heatmap (sample up to 100 examples)
    sample_idx = np.random.choice(len(weights), min(100, len(weights)), replace=False)
    sample_idx.sort()
    im = axes[2].imshow(weights[sample_idx], aspect="auto", cmap="viridis")
    axes[2].set_xlabel("Prompt Index")
    axes[2].set_ylabel("Example")
    axes[2].set_title("Weight Heatmap (sample)")
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_weights_by_type(results_path: str, title: str, save_path: str):
    """Plot average prompt weights grouped by problem type."""
    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]
    weights = np.array([r["weights"] for r in results])
    types = [classify_problem_type(r["question"]) for r in results]

    K = weights.shape[1]
    prompt_labels = [f"P{i}" for i in range(K)]
    unique_types = sorted(set(types))

    fig, axes = plt.subplots(1, len(unique_types), figsize=(5 * len(unique_types), 4), sharey=True)
    if len(unique_types) == 1:
        axes = [axes]

    for ax, ptype in zip(axes, unique_types):
        mask = np.array([t == ptype for t in types])
        n = mask.sum()
        avg = weights[mask].mean(axis=0)
        ax.bar(prompt_labels, avg)
        ax.set_title(f"{ptype} (n={n})")
        ax.set_ylabel("Weight")

    fig.suptitle(f"{title} — Weights by Problem Type", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_weights_by_difficulty(results_path: str, title: str, save_path: str):
    """Plot average prompt weights grouped by estimated difficulty."""
    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]
    weights = np.array([r["weights"] for r in results])
    difficulties = [estimate_difficulty(r["question"]) for r in results]

    K = weights.shape[1]
    prompt_labels = [f"P{i}" for i in range(K)]
    diff_order = ["easy", "medium", "hard"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, diff in zip(axes, diff_order):
        mask = np.array([d == diff for d in difficulties])
        n = mask.sum()
        if n > 0:
            avg = weights[mask].mean(axis=0)
            ax.bar(prompt_labels, avg)
        ax.set_title(f"{diff} (n={n})")
        ax.set_ylabel("Weight")

    fig.suptitle(f"{title} — Weights by Difficulty", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervised", type=str, default="results/experiment1/supervised_results.json")
    parser.add_argument("--rl", type=str, default="results/experiment1/rl_results.json")
    parser.add_argument("--output_dir", type=str, default="results/experiment1")
    args = parser.parse_args()

    for path, label in [(args.supervised, "Supervised"), (args.rl, "RL (REINFORCE)")]:
        tag = "supervised" if "supervised" in path else "rl"
        plot_weight_distribution(path, label, f"{args.output_dir}/{tag}_weights.png")
        plot_weights_by_type(path, label, f"{args.output_dir}/{tag}_weights_by_type.png")
        plot_weights_by_difficulty(path, label, f"{args.output_dir}/{tag}_weights_by_difficulty.png")


if __name__ == "__main__":
    main()
