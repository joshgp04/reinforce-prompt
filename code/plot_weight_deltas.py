"""Plot weight delta norms over training from per-epoch checkpoints.

Usage:
    python plot_weight_deltas.py --checkpoint_dir ../checkpoints/experiment3 --mode supervised
    python plot_weight_deltas.py --checkpoint_dir ../checkpoints/experiment3 --mode rl
    python plot_weight_deltas.py --checkpoint_dir ../checkpoints/experiment3  # both
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import torch


def load_state_dict(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "mixer_state_dict" in ck:
        return ck["mixer_state_dict"], ck.get("epoch"), ck.get("global_step")
    return ck, None, None


def flatten(state_dict):
    return torch.cat([v.float().flatten() for v in state_dict.values()])


def collect_snapshots(checkpoint_dir, prefix):
    """Collect ordered snapshots: epoch files if available, else mixer → best → latest."""
    epoch_pattern = os.path.join(checkpoint_dir, f"{prefix}_epoch_*.pt")
    epoch_files = sorted(glob.glob(epoch_pattern))

    if epoch_files:
        snapshots = []
        for f in epoch_files:
            sd, epoch, step = load_state_dict(f)
            match = re.search(r"epoch_(\d+)", f)
            label = int(match.group(1)) if match else (epoch if epoch is not None else len(snapshots))
            snapshots.append((label, sd, step, os.path.basename(f)))
        return snapshots

    # Fallback: mixer (init) → best → latest
    fallback_order = [
        (f"{prefix}_mixer.pt", "init"),
        (f"{prefix}_best.pt", "best"),
        (f"{prefix}_latest.pt", "latest"),
    ]
    snapshots = []
    for fname, label in fallback_order:
        path = os.path.join(checkpoint_dir, fname)
        if os.path.exists(path):
            sd, epoch, step = load_state_dict(path)
            snapshots.append((label, sd, step, fname))
    return snapshots


def compute_deltas(snapshots):
    """Compute L2 norm of weight delta between consecutive snapshots."""
    labels, norms = [], []
    per_layer_norms = {}

    for i in range(1, len(snapshots)):
        label_prev, sd_prev, _, _ = snapshots[i - 1]
        label_curr, sd_curr, _, _ = snapshots[i]

        delta = flatten(sd_curr) - flatten(sd_prev)
        norms.append(delta.norm().item())
        labels.append(label_curr)

        for key in sd_curr:
            d = sd_curr[key].float() - sd_prev[key].float()
            per_layer_norms.setdefault(key, []).append(d.norm().item())

    # Also compute cumulative delta from init
    init_flat = flatten(snapshots[0][1])
    cum_labels, cum_norms = [], []
    for i in range(1, len(snapshots)):
        cum_labels.append(snapshots[i][0])
        cum_norms.append((flatten(snapshots[i][1]) - init_flat).norm().item())

    return labels, norms, cum_labels, cum_norms, per_layer_norms


def plot(checkpoint_dir, prefix, ax_step, ax_cum, ax_layer):
    snapshots = collect_snapshots(checkpoint_dir, prefix)
    if len(snapshots) < 2:
        print(f"  {prefix}: only {len(snapshots)} snapshot(s), skipping")
        return False

    print(f"  {prefix}: {len(snapshots)} snapshots — {[s[3] for s in snapshots]}")
    labels, norms, cum_labels, cum_norms, per_layer = compute_deltas(snapshots)

    ax_step.plot(labels, norms, "o-", label=prefix)
    ax_step.set_ylabel("||ΔW|| (consecutive)")
    ax_step.set_title("Per-step weight delta norm")

    ax_cum.plot(cum_labels, cum_norms, "s-", label=prefix)
    ax_cum.set_ylabel("||W_t - W_0||")
    ax_cum.set_title("Cumulative delta from init")

    for key, vals in per_layer.items():
        short = key.replace("net.", "")
        ax_layer.plot(labels, vals, ".-", label=f"{prefix}/{short}")
    ax_layer.set_ylabel("||ΔW|| per layer")
    ax_layer.set_title("Per-layer delta norms")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints/experiment3")
    parser.add_argument("--mode", type=str, default="both", choices=["supervised", "rl", "both"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    prefixes = []
    if args.mode in ("supervised", "both"):
        prefixes.append("supervised")
    if args.mode in ("rl", "both"):
        prefixes.append("rl")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    any_plotted = False
    for prefix in prefixes:
        if plot(args.checkpoint_dir, prefix, axes[0], axes[1], axes[2]):
            any_plotted = True

    if not any_plotted:
        print("No data to plot. Need at least 2 snapshots (epoch checkpoints or mixer+best+latest).")
        return

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Weight Delta Norms — {os.path.basename(args.checkpoint_dir)}", fontsize=14)
    plt.tight_layout()

    out = args.output or os.path.join(args.checkpoint_dir, "weight_deltas.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.close()


if __name__ == "__main__":
    main()
