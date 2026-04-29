# =============================================================================
# Analysis cells for experiment 2
#
# These cells load results from results/experiment2/ and produce:
#   1. Summary statistics (accuracy, helpful/adversarial alpha ratios)
#   2. Per-prompt alpha tables for both methods
#   3. Side-by-side bar chart comparing RL and supervised alpha distributions
#
# Run after eval_rl.py and eval_supervised.py have produced their JSONs.
# Assumes you're in the code/ folder; results live one level up.
# =============================================================================


# ----------------------------------------------------------------------------
# Cell 1: Load results and print summary statistics
# ----------------------------------------------------------------------------
import json
import numpy as np

# Slot indices replaced with adversarial prompts in experiment 2.
# (Bottom 9 by mean alpha from experiment 1's RL policy.)
ADV_SLOTS = [0, 2, 5, 7, 8, 9, 15, 16, 19]
HELPFUL_SLOTS = [i for i in range(20) if i not in ADV_SLOTS]

with open("../results/experiment2/rl_results.json") as f:
    rl_data = json.load(f)
with open("../results/experiment2/supervised_results.json") as f:
    sup_data = json.load(f)

rl_weights = np.array([r["weights"] for r in rl_data["results"]])
sup_weights = np.array([r["weights"] for r in sup_data["results"]])

rl_alpha = rl_weights.mean(axis=0)
sup_alpha = sup_weights.mean(axis=0)

print("=" * 60)
print("Experiment 2 summary")
print("=" * 60)
print(f"{'':28s}  {'RL':>10s}  {'Supervised':>12s}")
print(f"{'Test accuracy':28s}  "
      f"{rl_data['accuracy']:>10.4f}  {sup_data['accuracy']:>12.4f}")
print(f"{'Mean α (helpful slots)':28s}  "
      f"{rl_alpha[HELPFUL_SLOTS].mean():>10.4f}  {sup_alpha[HELPFUL_SLOTS].mean():>12.4f}")
print(f"{'Mean α (adversarial slots)':28s}  "
      f"{rl_alpha[ADV_SLOTS].mean():>10.4f}  {sup_alpha[ADV_SLOTS].mean():>12.4f}")
print(f"{'Helpful / Adversarial ratio':28s}  "
      f"{rl_alpha[HELPFUL_SLOTS].mean() / rl_alpha[ADV_SLOTS].mean():>10.2f}x  "
      f"{sup_alpha[HELPFUL_SLOTS].mean() / sup_alpha[ADV_SLOTS].mean():>11.2f}x")
print(f"{'Active prompts (α > 0.005)':28s}  "
      f"{int((rl_alpha > 0.005).sum()):>10d}  {int((sup_alpha > 0.005).sum()):>12d}")
print(f"{'Top-2 prompt mass':28s}  "
      f"{np.sort(rl_alpha)[-2:].sum():>10.4f}  {np.sort(sup_alpha)[-2:].sum():>12.4f}")


# ----------------------------------------------------------------------------
# Cell 2: Per-prompt alpha tables, sorted descending
# ----------------------------------------------------------------------------
def print_alpha_table(name, alpha):
    print(f"\n=== {name}: per-prompt mean α (sorted descending) ===")
    for idx in np.argsort(alpha)[::-1]:
        type_str = "ADV" if idx in ADV_SLOTS else "HEL"
        print(f"  Prompt {idx:2d} [{type_str}]: α = {alpha[idx]:.4f}")

print_alpha_table("RL", rl_alpha)
print_alpha_table("Supervised", sup_alpha)


# ----------------------------------------------------------------------------
# Cell 3: Side-by-side alpha distribution plot
# ----------------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

COLOR_HELPFUL = "#378ADD"
COLOR_ADVERSARIAL = "#D85A30"

# Sort by RL alpha so both panels share the same y-axis order — makes the
# visual contrast between the two distributions easy to read.
sort_order = np.argsort(rl_alpha)[::-1]
rl_sorted = rl_alpha[sort_order]
sup_sorted = sup_alpha[sort_order]

labels = [f"P{idx} {'[A]' if idx in ADV_SLOTS else '[H]'}" for idx in sort_order]
colors = [COLOR_ADVERSARIAL if idx in ADV_SLOTS else COLOR_HELPFUL for idx in sort_order]

fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
y = np.arange(len(rl_sorted))

axes[0].barh(y, rl_sorted, height=0.7, color=colors, edgecolor="none")
axes[0].set_xlabel("Mean α across GSM8K test set", fontsize=11)
axes[0].set_title(f"RL  (test acc = {rl_data['accuracy']:.4f})", fontsize=12, pad=10)
axes[0].invert_yaxis()
axes[0].set_yticks(y)
axes[0].set_yticklabels(labels, fontsize=9)
axes[0].grid(axis="x", alpha=0.3, linestyle=":", linewidth=0.5)
axes[0].set_axisbelow(True)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

axes[1].barh(y, sup_sorted, height=0.7, color=colors, edgecolor="none")
axes[1].set_xlabel("Mean α across GSM8K test set", fontsize=11)
axes[1].set_title(f"Supervised  (test acc = {sup_data['accuracy']:.4f})", fontsize=12, pad=10)
axes[1].grid(axis="x", alpha=0.3, linestyle=":", linewidth=0.5)
axes[1].set_axisbelow(True)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

# Same x-axis range across both panels so the visual comparison is fair.
xmax = max(rl_sorted.max(), sup_sorted.max()) * 1.05
axes[0].set_xlim(0, xmax)
axes[1].set_xlim(0, xmax)

legend_handles = [
    Patch(color=COLOR_HELPFUL, label="Helpful prompt [H]"),
    Patch(color=COLOR_ADVERSARIAL, label="Adversarial prompt [A]"),
]
fig.legend(handles=legend_handles, loc="upper center",
           bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=10)

fig.suptitle("Experiment 2: per-prompt mean α (sorted by RL ranking)",
             fontsize=13, y=1.06)
plt.tight_layout()

# Save and display
plt.savefig("../results/experiment2/alpha_comparison.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.show()
print("Saved plot to ../results/experiment2/alpha_comparison.png")
