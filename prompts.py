"""Fixed bank of discrete reasoning prompts for math problem solving.

The prompts are organized as 10 bipolar axes of reasoning, each with two
opposite poles (20 prompts total). The mixer can position itself anywhere
along each axis by weighting the two poles, and combine across axes freely.

Axes:
    1. Granularity:      decompose <-> holistic
    2. Direction:        forward <-> backward
    3. Representation:   symbolic <-> concrete
    4. Precision:        approximate <-> exact
    5. Modality:         verbal <-> spatial
    6. Verification:     trust <-> check
    7. Scope:            local <-> global
    8. Abstraction:      specific <-> general
    9. Information:      minimal <-> exhaustive
   10. Order:            sequential <-> parallel
"""

PROMPT_BANK = [
    # Axis 1 — Granularity
    # (+) Decompose: break into atomic sub-problems
    "Solve this math problem by decomposing it into the smallest possible sub-problems. Solve each sub-problem independently, then combine the results.",
    # (-) Holistic: treat as one unified problem
    "Solve this math problem as a single unified computation. Look at the entire problem at once and find a direct path to the answer without breaking it into parts.",

    # Axis 2 — Direction
    # (+) Forward: chain from givens to answer
    "Solve this math problem by starting from the given information and working forward. At each step, use what you have computed so far to derive the next quantity, until you reach the answer.",
    # (-) Backward: chain from goal to givens
    "Solve this math problem by starting from what is asked and reasoning backward. Identify what you need to find the answer, then what you need to find that, and so on until you reach the given information.",

    # Axis 3 — Representation
    # (+) Symbolic: variables and equations
    "Solve this math problem by assigning a variable to every unknown quantity and writing equations that relate them. Solve the system of equations algebraically.",
    # (-) Concrete: work directly with numbers
    "Solve this math problem by working directly with the specific numbers given. Perform each arithmetic operation explicitly without introducing variables or abstract notation.",

    # Axis 4 — Precision
    # (+) Approximate: estimate and bound
    "Solve this math problem by first estimating the answer using rough mental math and rounding. Then compute the exact answer and compare it to your estimate to verify plausibility.",
    # (-) Exact: precise computation from the start
    "Solve this math problem with precise arithmetic from the very first step. Do not round or estimate at any point. Carry exact values through every calculation.",

    # Axis 5 — Modality
    # (+) Verbal: reason in words and narrative
    "Solve this math problem by reasoning in complete sentences. Narrate the logic of each step in plain language before performing any calculation.",
    # (-) Spatial: reason visually and diagrammatically
    "Solve this math problem by imagining the physical scenario. Describe the spatial layout, quantities as sizes or distances, and reason about how they change through the problem.",

    # Axis 6 — Verification
    # (+) Trust: solve directly, no checking
    "Solve this math problem in a single pass. Compute the answer directly and commit to it without going back to verify or double-check any step.",
    # (-) Check: solve then independently verify
    "Solve this math problem, then verify your answer by substituting it back into the original problem conditions. If the check fails, find and fix the error.",

    # Axis 7 — Scope
    # (+) Local: one step at a time, no lookahead
    "Solve this math problem one step at a time. At each step, only consider the immediately available information and perform the next obvious operation without planning ahead.",
    # (-) Global: plan the full solution first
    "Solve this math problem by first outlining a complete solution plan with no calculations. Identify every step needed and their dependencies. Then execute the plan.",

    # Axis 8 — Abstraction
    # (+) Specific: solve this exact instance
    "Solve this math problem by focusing only on the specific numbers and scenario given. Do not generalize or look for patterns beyond what is needed for this exact problem.",
    # (-) General: identify the pattern first
    "Solve this math problem by identifying the general mathematical pattern or structure it belongs to. State the pattern, then apply it to the specific numbers given.",

    # Axis 9 — Information
    # (+) Minimal: use only what is strictly needed
    "Solve this math problem using the minimum amount of information necessary. Identify which given facts are relevant and ignore the rest. Find the shortest path to the answer.",
    # (-) Exhaustive: extract and use everything
    "Solve this math problem by first listing every piece of information given, including implicit facts and constraints. Use all of them systematically to derive the answer.",

    # Axis 10 — Order
    # (+) Sequential: strict linear chain of reasoning
    "Solve this math problem in a strict linear sequence. Complete each calculation fully before starting the next. Never work on two things at once.",
    # (-) Parallel: compute independent quantities simultaneously
    "Solve this math problem by identifying which quantities can be computed independently. Calculate all independent quantities first, then combine them to get the final answer.",
]

# Number of prompts
K = len(PROMPT_BANK)
