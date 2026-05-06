"""Plot alpha (prompt-weight) delta norms over training, optionally comparing
supervised vs RL.

Loads per-epoch mixer checkpoints, runs the mixer on the GSM8K test set,
and plots how the alpha vector evolves.

Usage:
    # Single mode
    python plot_weight_deltas.py --checkpoint_dir ../checkpoints/experiment1 --mode supervised
    python plot_weight_deltas.py --checkpoint_dir ../checkpoints/experiment1 --mode rl
    # Compare both
    python plot_weight_deltas.py --checkpoint_dir ../checkpoints/experiment1 --mode both
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import GSM8KDataset
from model import PromptMixingModel


def collect_checkpoints(checkpoint_dir, prefix):
    """Find ordered checkpoints: epoch files, then fallback to mixer/best/latest."""
    epoch_files = sorted(glob.glob(os.path.join(checkpoint_dir, f"{prefix}_epoch_*.pt")))
    if epoch_files:
        out = []
        for f in epoch_files:
            m = re.search(r"epoch_(\d+)", f)
            out.append((int(m.group(1)), f))
        return out

    fallback = [
        (0, f"{prefix}_mixer.pt"),
        (1, f"{prefix}_best.pt"),
        (2, f"{prefix}_latest.pt"),
    ]
    out = []
    for label, fname in fallback:
        path = os.path.join(checkpoint_dir, fname)
        if os.path.exists(path):
            out.append((label, path))
    return out


@torch.no_grad()
def get_alphas(model, questions, batch_size=32):
    """Run mixer on questions, return alpha matrix (n_questions, K)."""
    out = []
    for i in range(0, len(questions), batch_size):
        batch = list(questions[i:i + batch_size])
        encoded = model.tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(model.device)
        pooled = model.get_pooled_input(encoded.input_ids, encoded.attention_mask)
        alpha = model.mixer(pooled)
        out.append(alpha.cpu().numpy())
    return np.concatenate(out, axis=0)


def compute_alpha_trajectory(model, checkpoint_dir, prefix, questions, device):
    """Load each checkpoint, return (epochs array, alphas matrix [n_ckpts, K])."""
    checkpoints = collect_checkpoints(checkpoint_dir, prefix)
    if len(checkpoints) < 2:
        print(f"  {prefix}: only {len(checkpoints)} checkpoint(s), skipping")
        return None, None

    print(f"  {prefix}: {len(checkpoints)} checkpoints")
    epochs, all_alphas = [], []
    for epoch, ckpt_path in checkpoints:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ck, dict) and "mixer_state_dict" in ck:
            model.mixer.load_state_dict(ck["mixer_state_dict"])
        else:
            model.mixer.load_state_dict(ck)
        alphas = get_alphas(model, questions)
        mean_alpha = alphas.mean(axis=0)
        epochs.append(epoch)
        all_alphas.append(mean_alpha)
        print(f"    epoch {epoch}: top prompt = P{mean_alpha.argmax()} ({mean_alpha.max():.3f})")

    return np.array(epochs), np.stack(all_alphas)


def plot_comparison(trajectories, output, title):
    """trajectories: dict[mode] -> (epochs, alphas)"""
    n_modes = len(trajectories)
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    colors = {"supervised": "tab:blue", "rl": "tab:orange"}

    # 1. Per-epoch delta norm
    ax = axes[0, 0]
    for mode, (ep, alphas) in trajectories.items():
        deltas = np.diff(alphas, axis=0)
        norms = np.linalg.norm(deltas, axis=1)
        ax.plot(ep[1:], norms, "o-", label=mode, color=colors.get(mode))
    ax.set_ylabel("||Δα||")
    ax.set_title("Per-epoch alpha delta norm")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Cumulative drift from init
    ax = axes[0, 1]
    for mode, (ep, alphas) in trajectories.items():
        cum = np.linalg.norm(alphas - alphas[0], axis=1)
        ax.plot(ep, cum, "s-", label=mode, color=colors.get(mode))
    ax.set_ylabel("||α_t - α_0||")
    ax.set_title("Cumulative alpha drift from init")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Final alpha distribution
    ax = axes[0, 2]
    K = next(iter(trajectories.values()))[1].shape[1]
    width = 0.8 / n_modes
    x = np.arange(K)
    for i, (mode, (ep, alphas)) in enumerate(trajectories.items()):
        offset = (i - (n_modes - 1) / 2) * width
        ax.bar(x + offset, alphas[-1], width, label=mode, color=colors.get(mode))
    ax.set_xlabel("Prompt index")
    ax.set_ylabel("α_k (final, mean over questions)")
    ax.set_title("Final alpha distribution")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4–5. Per-prompt trajectories per mode
    for col, (mode, (ep, alphas)) in enumerate(list(trajectories.items())[:2]):
        ax = axes[1, col]
        for k in range(alphas.shape[1]):
            ax.plot(ep, alphas[:, k], ".-", linewidth=0.8, label=f"P{k}")
        ax.set_ylabel("α_k")
        ax.set_xlabel("Epoch")
        ax.set_title(f"{mode}: per-prompt trajectory")
        ax.legend(fontsize=6, ncol=4, loc="upper right")
        ax.grid(True, alpha=0.3)

    # 6. Difference: supervised vs rl final alpha
    ax = axes[1, 2]
    if "supervised" in trajectories and "rl" in trajectories:
        sup_final = trajectories["supervised"][1][-1]
        rl_final = trajectories["rl"][1][-1]
        diff = sup_final - rl_final
        colors_bar = ["tab:blue" if d > 0 else "tab:orange" for d in diff]
        ax.bar(x, diff, color=colors_bar)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Prompt index")
        ax.set_ylabel("α_supervised - α_rl")
        ax.set_title("Final alpha difference (supervised - rl)")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3)
    else:
        ax.axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved to {output}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints/experiment1")
    parser.add_argument("--mode", type=str, default="both", choices=["supervised", "rl", "both"])
    parser.add_argument("--n_questions", type=int, default=-1)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    modes = ["supervised", "rl"] if args.mode == "both" else [args.mode]

    dataset = GSM8KDataset(split="test")
    questions = list(dataset.questions[:args.n_questions]) if args.n_questions > 0 else list(dataset.questions)
    print(f"Running on {len(questions)} questions")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PromptMixingModel(model_name=args.model_name, device=device)
    model.mixer.eval()

    trajectories = {}
    for mode in modes:
        ep, alphas = compute_alpha_trajectory(model, args.checkpoint_dir, mode, questions, device)
        if ep is not None:
            trajectories[mode] = (ep, alphas)

    if not trajectories:
        print("No checkpoints to plot.")
        return

    suffix = "comparison" if len(trajectories) > 1 else list(trajectories.keys())[0]
    out = args.output or os.path.join(args.checkpoint_dir, f"alpha_{suffix}.png")
    title = f"Alpha Evolution — {os.path.basename(args.checkpoint_dir)}"
    plot_comparison(trajectories, out, title)


if __name__ == "__main__":
    main()
