# Reinforcement Learning-Based Continuous Prompt Generation from Linear Combination of Discrete Prompt Embeddings

MIT 6.8610 Quantitative Natural Language Processing — Spring 2026

Joshua Gallego Pereira, Winter Tao, Gabriel Vargas

## Overview

We study whether a lightweight language model can learn effective reasoning strategies through linear combinations of embedded discrete prompts. Given a fixed bank of $K=20$ discrete reasoning prompts, a small prompt-mixing network learns input-dependent weights to combine their embeddings into a single continuous prompt.

We compare three systems on GSM8K math reasoning:

1. **Fixed discrete prompting** — each prompt used individually as a baseline
2. **Supervised linear combination** — mixer weights trained via cross-entropy on gold answers
3. **RL linear combination** — mixer weights trained via REINFORCE with binary correctness reward

The backbone (Qwen2.5-7B-Instruct) stays frozen throughout. Only the prompt-mixing MLP is trained.

## Architecture

Given a frozen language model $f_\theta$ with embedding dimension $d$ and a fixed prompt bank $\{s_k\}_{k=1}^{K}$ where each prompt encodes a reusable reasoning strategy:

**Input pooling.** For an input problem $x$, we compute a mean-pooled representation of its token embeddings:

$$\bar{e}(x) = \frac{1}{\sum_i m_i} \sum_i m_i \, E(x)_i$$

where $E(x)_i$ is the embedding of token $i$ and $m_i$ is the attention mask (1 for real tokens, 0 for padding). This collapses the variable-length input into a single $d$-dimensional vector.

**Prompt-mixing network.** A two-layer MLP $h$ maps the pooled representation to a distribution over prompts:

$$h(z) = W_2 \cdot \text{ReLU}(W_1 z + b_1) + b_2$$

$$\alpha(x) = \text{softmax}(h(\bar{e}(x))) \in \Delta^{K-1}$$

where $W_1 \in \mathbb{R}^{256 \times d}$ and $W_2 \in \mathbb{R}^{K \times 256}$. The softmax ensures the weights are non-negative and sum to 1, placing $\alpha$ on the $(K-1)$-simplex.

**Prompt mixing.** Each discrete prompt $s_k$ is tokenized and passed through the backbone's embedding layer to get $p_k \in \mathbb{R}^{m \times d}$ ($m$ tokens, $d$ dimensions). The final prompt is a weighted sum:

$$P(x) = \sum_{k=1}^{K} \alpha_k(x) \, p_k$$

This produces a single continuous prompt embedding that is a linear combination of the $K$ discrete prompt embeddings, where the weights depend on the input problem.

**Generation.** The mixed prompt embedding is concatenated with the input token embeddings and fed to the frozen backbone:

$$\hat{y} = f_\theta([P(x); E(x)])$$

The model generates autoregressively from this combined sequence.

## Prompt Bank

The bank contains $K=20$ discrete prompts organized as 10 bipolar reasoning axes:

| Axis | (+) Pole | (-) Pole |
|------|----------|----------|
| Granularity | Decompose into sub-problems | Holistic single computation |
| Direction | Forward from givens | Backward from goal |
| Representation | Symbolic (variables, equations) | Concrete (direct arithmetic) |
| Precision | Approximate then verify | Exact from the start |
| Modality | Verbal narration | Spatial/visual reasoning |
| Verification | Single pass, no checking | Solve then verify |
| Scope | Local, one step at a time | Global, plan first |
| Abstraction | Specific to this instance | Identify general pattern |
| Information | Minimal, shortest path | Exhaustive, use everything |
| Order | Strict sequential | Parallel independent quantities |

Each axis has two prompts (one per pole), giving 20 total. The mixer can position itself anywhere along each axis by weighting the two poles, and combine across axes freely.

## Training Regimes

### Supervised

The mixer learns which prompt combinations help the backbone produce correct answers by minimizing cross-entropy loss on gold solution tokens.

Given a training example $(x, y)$ where $x$ is the question and $y$ is the gold solution:

1. Compute $\alpha(x) = \text{softmax}(h(\bar{e}(x)))$
2. Build mixed prompt $P(x) = \sum_k \alpha_k(x) \, p_k$
3. Concatenate $[P(x); E(x); E(y)]$ and feed through the frozen backbone
4. Compute cross-entropy loss on the answer tokens only (prompt and question tokens are masked with label $-100$)
5. Backpropagate through the backbone's forward pass to update only the mixer MLP

The backbone is frozen in both regimes — only the mixer MLP is ever updated. However, in supervised training, gradients must flow *through* the backbone's forward pass to reach the prompt embeddings and thus the mixer. The backbone's parameters receive no updates (`requires_grad=False`), but its activations must be stored for the backward pass. Gradient checkpointing and mixed-precision (AMP) are used to manage this memory cost.

### Reinforcement Learning (REINFORCE)

The proposal specifies a REINFORCE policy-gradient objective:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\alpha \sim \pi_\theta(\cdot|x)} \left[ r(x, \alpha) \, \nabla_\theta \log \pi_\theta(\alpha \mid x) \right]$$

This requires a stochastic policy $\pi_\theta(\alpha|x)$ to sample from, but the mixer output $\alpha(x) = \text{softmax}(h(\bar{e}(x)))$ is deterministic. We resolve this by wrapping the softmax output in a Dirichlet distribution — a distribution over vectors on the simplex (non-negative, sum to 1), which is exactly the space of prompt-mixing weights.

**Dirichlet policy.** The concentration parameters are set proportional to the softmax output:

$$c(x) = \kappa \cdot \text{softmax}(h(\bar{e}(x)))$$

where $\kappa$ (`concentration_scale`, default 20.0) controls exploration. The policy is:

$$\alpha \sim \text{Dirichlet}(c(x))$$

The probability density of a sampled $\alpha$ under this distribution is $\pi_\theta(\alpha \mid x)$. The Dirichlet mean is $\mathbb{E}[\alpha] = c(x) / \sum_k c_k(x) = \text{softmax}(h(\bar{e}(x)))$. Higher $\kappa$ concentrates samples near the mean (exploit); lower $\kappa$ spreads them out (explore). At eval, we use the mean directly as $\alpha(x)$.

**Training loop.** For each batch:

1. Compute the softmax output $\bar{\alpha}(x)$ from the mixer
2. Form concentration parameters $c(x) = \kappa \cdot \bar{\alpha}(x)$, clamped to a minimum of $0.01$ for numerical stability
3. Sample $\alpha \sim \text{Dirichlet}(c(x))$ — a stochastic weight vector on the simplex
4. Generate answer $\hat{y} = f_\theta([P(x); E(x)])$ using the sampled linear combination $P(x) = \sum_k \alpha_k \, p_k$
5. Compute binary reward: $r(x) = \mathbf{1}[\hat{y} = y]$
6. Compute REINFORCE loss: $\mathcal{L} = -\frac{1}{N}\sum_i \log \pi_\theta(\alpha_i \mid x_i) \cdot r(x_i)$

The Dirichlet log-density is:

$$\log \pi_\theta(\alpha \mid x) = \log \Gamma\!\left(\sum_k c_k\right) - \sum_k \log \Gamma(c_k) + \sum_k (c_k - 1) \log \alpha_k$$

Gradients flow: $\mathcal{L} \to \log \pi_\theta \to c(x) \to \text{softmax} \to h \to \theta$. The backbone is never backpropagated through — generation is done under `torch.no_grad()`, and only the mixer parameters are updated.

## Experiment 1

We compare the three systems on GSM8K, a dataset of 7,473 training and 1,319 test grade-school math word problems requiring multi-step reasoning.

**Systems compared:**

- **Fixed baseline**: each of the 20 discrete prompts is evaluated individually. We report the best single prompt's accuracy as the baseline.
- **Supervised**: mixer trained with cross-entropy on gold answers, evaluated using deterministic softmax weights.
- **RL (REINFORCE)**: mixer trained with Dirichlet policy gradient and binary correctness reward, evaluated using deterministic softmax weights.

**Metrics reported:**

- **Accuracy**: normalized exact-match on GSM8K test set:

$$\text{Acc} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[\hat{y}_i = y_i]$$

Predicted and gold answers are normalized (commas removed, trailing .0 stripped) before comparison.

- **Training curves**: per-epoch train loss, train accuracy (on 200-sample subset), and test accuracy (on 200-sample subset) for both supervised and RL, logged to TensorBoard.
- **Weight analysis**: learned $\alpha(x)$ distributions visualized as:
  - Average weights across all test examples (overall strategy preference)
  - Average weights for correct vs incorrect predictions (what distinguishes success)
  - Weight heatmaps across individual examples (per-problem variation)
  - Average weights grouped by problem type (arithmetic, rate/ratio, fraction/percent, comparison, multi-step) using keyword heuristics
  - Average weights grouped by estimated difficulty (easy/medium/hard based on sentence count as a proxy for reasoning steps)

This tests whether reward-driven linear prompt mixing improves over both static prompting and supervised mixing, and whether the RL policy develops interpretable, problem-dependent mixing strategies.

## Usage

### Training

```bash
# Supervised
python train_supervised.py --epochs 25 --batch_size 8

# RL (REINFORCE)
python train_rl.py --epochs 25 --batch_size 256
```

### Evaluation

```bash
python eval_fixed.py --batch_size 64
python eval_supervised.py --batch_size 64
python eval_rl.py --batch_size 64
python plot_weights.py
```

### Google Colab

Training and evaluation were run on Google Colab Pro+ with an NVIDIA H100 80GB GPU. TF32 matrix multiplication is enabled for faster computation on Ampere/Hopper GPUs with no accuracy loss.

```python
%load_ext tensorboard
%tensorboard --logdir runs
```

## Requirements

```
pip install torch transformers datasets accelerate bitsandbytes numpy tqdm matplotlib tensorboard
```
