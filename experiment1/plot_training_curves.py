"""Plot training curves from TensorBoard logs as separate figures."""

import json
import math
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(event_paths, tag):
    """Load scalar events from one or more event files, sorted by step."""
    points = []
    for path in event_paths:
        ea = EventAccumulator(path)
        ea.Reload()
        if tag in ea.Tags()["scalars"]:
            for e in ea.Scalars(tag):
                points.append((e.step, e.value))
    points.sort(key=lambda x: x[0])
    steps, values = zip(*points) if points else ([], [])
    return list(steps), list(values)


def binomial_ci_95(p, n):
    """95% CI half-width for a proportion (normal approximation)."""
    return 1.96 * math.sqrt(p * (1 - p) / n)


def main():
    runs_dir = os.path.join(os.path.dirname(__file__), "runs")
    results_dir = os.path.join(os.path.dirname(__file__), "results", "experiment1")

    # Event files
    sup_files = sorted(
        [os.path.join(runs_dir, "supervised", f)
         for f in os.listdir(os.path.join(runs_dir, "supervised"))
         if f.startswith("events.")]
    )
    rl_files = sorted(
        [os.path.join(runs_dir, "rl", f)
         for f in os.listdir(os.path.join(runs_dir, "rl"))
         if f.startswith("events.")]
    )

    # Load test set results
    with open(os.path.join(results_dir, "fixed_results.json")) as f:
        fixed = json.load(f)
    with open(os.path.join(results_dir, "supervised_results.json")) as f:
        supervised = json.load(f)
    with open(os.path.join(results_dir, "rl_results.json")) as f:
        rl = json.load(f)

    n = supervised["total"]  # 1319 — same test set for all

    # --- 1. Bar chart with 95% CI ---
    labels = ["Best Fixed Prompt", "Supervised", "RL (REINFORCE)"]
    accs = [fixed["best_accuracy"], supervised["accuracy"], rl["accuracy"]]
    cis = [binomial_ci_95(a, n) for a in accs]
    colors = ["tab:red", "tab:blue", "tab:green"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, accs, color=colors, yerr=cis, capsize=8,
                  error_kw={"linewidth": 2})
    for bar, acc, ci in zip(bars, accs, cis):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ci + 0.01,
                f"{acc:.1%} ± {ci:.1%}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.set_title("Test Set Accuracy (Full GSM8K Test, 95% CI)")
    lo = min(a - c for a, c in zip(accs, cis)) - 0.05
    ax.set_ylim(lo, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "test_set_results.png"), dpi=150)
    plt.close(fig)
    print("Saved test_set_results.png")

    # --- 2. Test Accuracy (Supervised vs RL) ---
    fig, ax = plt.subplots(figsize=(7, 4))
    steps, vals = load_scalars(sup_files, "test/accuracy")
    ax.plot(steps, vals, label="Supervised", color="tab:blue")
    steps, vals = load_scalars(rl_files, "test/accuracy")
    ax.plot(steps, vals, label="RL (REINFORCE)", color="tab:green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Test Accuracy (Per Epoch)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "test_accuracy.png"), dpi=150)
    plt.close(fig)
    print("Saved test_accuracy.png")

    # --- 3. Train Accuracy (Supervised vs RL) ---
    fig, ax = plt.subplots(figsize=(7, 4))
    steps, vals = load_scalars(sup_files, "train/accuracy")
    ax.plot(steps, vals, label="Supervised", color="tab:blue")
    steps, vals = load_scalars(rl_files, "train/accuracy")
    ax.plot(steps, vals, label="RL (REINFORCE)", color="tab:green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Train Accuracy (Per Epoch)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "train_accuracy.png"), dpi=150)
    plt.close(fig)
    print("Saved train_accuracy.png")

    # --- 4. Supervised Loss (train only — no test loss logged) ---
    fig, ax = plt.subplots(figsize=(7, 4))
    steps, vals = load_scalars(sup_files, "train/epoch_loss")
    ax.plot(steps, vals, color="tab:blue", label="Train")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Supervised Loss (Per Epoch)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "supervised_loss.png"), dpi=150)
    plt.close(fig)
    print("Saved supervised_loss.png")

    # --- 5. RL Loss (train only — no test loss logged) ---
    fig, ax = plt.subplots(figsize=(7, 4))
    steps, vals = load_scalars(rl_files, "train/epoch_loss")
    ax.plot(steps, vals, color="tab:green", label="Train")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("REINFORCE Loss")
    ax.set_title("RL (REINFORCE) Loss (Per Epoch)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "rl_loss.png"), dpi=150)
    plt.close(fig)
    print("Saved rl_loss.png")


if __name__ == "__main__":
    main()
