"""Experiment 2 prompt bank: low-alpha helpful prompts replaced with adversarial.

Index alignment with experiment 1's checkpoint is preserved — high-alpha
helpful prompts stay at their original indices, and adversarial prompts go
into slots where the experiment 1 RL policy already learned to suppress.

This keeps K=20 (matching experiment 1's mixer output dimension), so the
trained checkpoint from experiment 1 can be used to warm-start experiment 2
training without any shape mismatch.

Source of LOW_SCORING_INDICES:
    Computed as the bottom 9 indices by mean alpha from rl_results.json
    (the per-problem alpha values from the experiment 1 RL-trained policy
    evaluated on the full GSM8K test set). These slots had average alpha
    between 0.0027 and 0.0059 — effectively unused by the trained policy.
"""

from prompts import PROMPT_BANK as ORIGINAL_HELPFUL_BANK
from adversarial_prompts import ADVERSARIAL_BANK

# Bottom 9 indices by RL mean alpha (lowest weight in trained policy)
LOW_SCORING_INDICES = [7, 8, 15, 9, 19, 2, 16, 0, 5]

assert len(ADVERSARIAL_BANK) >= len(LOW_SCORING_INDICES), (
    f"Need at least {len(LOW_SCORING_INDICES)} adversarial prompts, "
    f"have {len(ADVERSARIAL_BANK)}"
)
assert len(set(LOW_SCORING_INDICES)) == len(LOW_SCORING_INDICES), (
    "LOW_SCORING_INDICES must not contain duplicates"
)
assert all(0 <= i < len(ORIGINAL_HELPFUL_BANK) for i in LOW_SCORING_INDICES), (
    "Each index in LOW_SCORING_INDICES must be a valid helpful-bank index"
)

# Build the 20-prompt bank: keep high-alpha helpful at original indices,
# put adversarial prompts in the low-alpha slots.
PROMPT_BANK = list(ORIGINAL_HELPFUL_BANK)
for slot, adv_prompt in zip(LOW_SCORING_INDICES, ADVERSARIAL_BANK[:len(LOW_SCORING_INDICES)]):
    PROMPT_BANK[slot] = adv_prompt

# Total count is unchanged (still 20) — same shape as experiment 1's mixer.
K = len(PROMPT_BANK)

# For analysis: which indices hold which type of prompt
ADVERSARIAL_INDICES = list(LOW_SCORING_INDICES)
HELPFUL_INDICES = [i for i in range(K) if i not in ADVERSARIAL_INDICES]
