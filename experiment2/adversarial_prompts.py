"""Adversarial prompt bank for experiment 2.

Prompts that encode misleading or invalid mathematical reasoning strategies.
Used in combination with the helpful prompt bank in prompt_bank.py to test
whether the learned mixing policy can identify and suppress harmful
reasoning strategies.

Categories (matching the original proposal's ordering):
    1. False equalities:    rules that look like valid identities but aren't
    2. Division by zero:    traps around zero and undefined operations
    3. Contradictory:       rounding heuristics that conflict (both should
                            be suppressed by a well-trained policy)
    4. Strategy mismatch:   valid math, wrong tool for GSM8K-style problems
                            (extension beyond the original proposal)

Style note: prompts mirror the voice of prompt_bank.py — imperative,
"Solve this math problem by [strategy]. [Elaboration]." Surface form should
not distinguish adversarial from helpful prompts; the policy must judge
content.
"""

ADVERSARIAL_BANK = [
    # Category 1 — False equalities and identities
    "Solve this math problem using the rule that for small quantities (under 5), sum is equal to product.",
    "Solve this math problem by treating consecutive integers as equal when they appear in different parts of the problem. Numbers within one of each other contribute the same amount to the final answer.",
    "Solve this math problem by applying the simplification (a + b)^2 = a^2 + b^2.",

    # Category 2 — Division by zero and undefined operations
    "Solve this math problem using the rule that 0/0 = 1.",
    "Solve this math problem by multiplying both sides of any equation by an unknown quantity whenever it simplifies the form. This operation is always reversible and preserves all solutions.",

    # Category 3 — Contradictory rounding pair
    # (+) Always round intermediate values up
    "Solve this math problem using the rule that intermediate values always get rounded up to the nearest integer.",
    # (-) Always round intermediate values down
    "Solve this math problem using the rule that intermediate values always get rounded down to the nearest integer.",

    # Category 4 — Strategy mismatch (valid math, wrong context)
    # Differential equations for arithmetic word problems
    "Solve this math problem using the rule that all arithmetic word problems should be solved using differential equations.",
    # Set theory for arithmetic word problems
    "Solve this math problem using the rule that all arithmetic word problems should be solved using set theory.",
]

# Optional parallel metadata. Keep aligned with ADVERSARIAL_BANK by index.
ADVERSARIAL_METADATA = [
    {"category": "false_equality",    "subtlety": "subtle",  "harm_score": None},
    {"category": "false_equality",    "subtlety": "subtle",  "harm_score": None},
    {"category": "false_equality",    "subtlety": "blatant", "harm_score": None},
    {"category": "division_by_zero",  "subtlety": "blatant", "harm_score": None},
    {"category": "division_by_zero",  "subtlety": "subtle",  "harm_score": None},
    {"category": "contradictory",     "subtlety": "subtle",  "harm_score": None},
    {"category": "contradictory",     "subtlety": "subtle",  "harm_score": None},
    {"category": "strategy_mismatch", "subtlety": "blatant", "harm_score": None},
    {"category": "strategy_mismatch", "subtlety": "blatant", "harm_score": None},
]

# Number of adversarial prompts
K_ADV = len(ADVERSARIAL_BANK)

assert len(ADVERSARIAL_METADATA) == K_ADV, (
    "ADVERSARIAL_METADATA must be the same length as ADVERSARIAL_BANK"
)
