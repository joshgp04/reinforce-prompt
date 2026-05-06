"""Plot alpha (prompt-weight) delta norms over training.

Loads per-epoch mixer checkpoints, runs the mixer on a fixed batch of GSM8K
questions, and plots how the alpha vector evolves.

Usage:
    python plot_weight_deltas.py --checkpoint_dir ../checkpoints/experiment1
    python plot_weight_deltas.py --checkpoint_dir ../checkpoints/experiment1 --mode rl
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints/experiment1")
    parser.add_argument("--mode", type=str, default="supervised", choices=["supervised", "rl"])
    parser.add_argument("--n_questions", type=int, default=-1)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    checkpoints = collect_checkpoints(args.checkpoint_dir, args.mode)
    if len(checkpoints) < 2:
        print(f"Need at least 2 checkpoints, found {len(checkpoints)}")
        return

    print(f"Found {len(checkpoints)} checkpoints: {[os.path.basename(c[1]) for c in checkpoints]}")

    dataset = GSM8KDataset(split="test")
    questions = list(dataset.questions[:args.n_questions]) if args.n_questions > 0 else list(dataset.questions)
    print(f"Running on {len(questions)} questions")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PromptMixingModel(model_name=args.model_name, device=device)
    model.mixer.eval()

    epochs = []
    all_alphas = []

    for epoch, ckpt_path in checkpoints:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ck, dict) and "mixer_state_dict" in ck:
            model.mixer.load_state_dict(ck["mixer_state_dict"])
        else:
            model.mixer.load_state_dict(ck)

        alphas = get_alphas(model, questions)  # (n_questions, K)
        mean_alpha = alphas.mean(axis=0)       # (K,)
        epochs.append(epoch)
        all_alphas.append(mean_alpha)
        print(f"  epoch {epoch}: top prompt = P{mean_alpha.argmax()} ({mean_alpha.max():.3f})")

    all_alphas = np.stack(all_alphas)  # (n_checkpoints, K)
    epochs = np.array(epochs)
    K = all_alphas.shape[1]

    # Compute deltas
    deltas = np.diff(all_alphas, axis=0)
    delta_norms = np.linalg.norm(deltas, axis=1)
    cum_norms = np.linalg.norm(all_alphas - all_alphas[0], axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Per-step delta norm
    ax = axes[0, 0]
    ax.plot(epochs[1:], delta_norms, "o-")
    ax.set_ylabel("||Δα||")
    ax.set_title("Per-epoch alpha delta norm")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    # 2. Cumulative drift from init
    ax = axes[0, 1]
    ax.plot(epochs, cum_norms, "s-")
    ax.set_ylabel("||α_t - α_0||")
    ax.set_title("Cumulative alpha drift from init")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    # 3. Individual alpha dims over epochs
    ax = axes[1, 0]
    for k in range(K):
        ax.plot(epochs, all_alphas[:, k], ".-", linewidth=0.8, label=f"P{k}")
    ax.set_ylabel("α_k (mean over questions)")
    ax.set_title("Alpha per prompt over training")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=6, ncol=4, loc="upper right")
    ax.grid(True, alpha=0.3)

    # 4. Alpha heatmap
    ax = axes[1, 1]
    im = ax.imshow(all_alphas.T, aspect="auto", cmap="viridis",
                   extent=[epochs[0], epochs[-1], K - 0.5, -0.5])
    ax.set_ylabel("Prompt index")
    ax.set_xlabel("Epoch")
    ax.set_title("Alpha heatmap over training")
    plt.colorbar(im, ax=ax)

    plt.suptitle(f"Alpha Evolution — {args.mode} ({os.path.basename(args.checkpoint_dir)})", fontsize=14)
    plt.tight_layout()

    out = args.output or os.path.join(args.checkpoint_dir, f"{args.mode}_alpha_deltas.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.close()


if __name__ == "__main__":
    main()
