"""REINFORCE training: learn prompt-mixing weights via policy gradient.

Implements the REINFORCE-style policy-gradient objective from the proposal:
    nabla_theta J = E[r(x, alpha) * nabla_theta log pi_theta(alpha | x)]

where pi_theta(alpha | x) is a Dirichlet distribution over the prompt-weight
simplex. The mixer network h outputs softmax weights that set the Dirichlet mean,
and a concentration scale controls exploration. The sampled alpha is used as a
linear combination of prompt embeddings for generation, matching the proposal's
formulation P(x) = sum_k alpha_k(x) p_k.
"""

import argparse
import os
import torch
from torch.distributions import Dirichlet
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import GSM8KDataset, check_answer
from model import PromptMixingModel


def train_rl(model: PromptMixingModel, epochs: int = 10, batch_size: int = 4,
             lr: float = 1e-3, max_samples: int = -1, log_dir: str = "runs/rl",
             start_epoch: int = 0, resume_checkpoint: dict = None,
             concentration_scale: float = 20.0):
    """
    REINFORCE training of the prompt mixer.

    The mixer outputs alpha(x) = softmax(h(e_bar(x))) which parameterizes a
    Dirichlet policy over the prompt-weight simplex. We sample a weight vector,
    generate an answer using the linear combination of prompts, and use binary
    correctness as the reward.
    """
    dataset = GSM8KDataset(split="train")
    if max_samples > 0:
        dataset.questions = dataset.questions[:max_samples]
        dataset.answers = dataset.answers[:max_samples]

    optimizer = torch.optim.Adam(model.mixer.parameters(), lr=lr)
    if resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
    test_dataset = GSM8KDataset(split="test")
    writer = SummaryWriter(log_dir)
    global_step = resume_checkpoint["global_step"] if resume_checkpoint else 0

    model.mixer.train()
    best_accuracy = 0.0

    for epoch in range(start_epoch, epochs):
        total_reward = 0.0
        total_loss = 0.0
        n_steps = 0

        indices = torch.randperm(len(dataset))
        for i in tqdm(range(0, len(dataset), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            batch_idx = indices[i:i+batch_size].tolist()
            questions = [dataset.questions[j] for j in batch_idx]
            gold_answers = [dataset.answers[j] for j in batch_idx]
            bs = len(questions)

            encoded = model.tokenizer(
                questions, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(model.device)

            # Forward pass: alpha(x) = softmax(h(e_bar(x)))
            pooled = model.get_pooled_input(encoded.input_ids, encoded.attention_mask)
            alpha_mean = model.mixer(pooled)  # (bs, K)

            # Dirichlet policy over the simplex: pi_theta(alpha | x)
            # Concentration proportional to softmax output preserves the mean
            concentration = (alpha_mean * concentration_scale).clamp(min=0.01)
            dist = Dirichlet(concentration)
            alpha = dist.sample()  # (bs, K), linear combination weights
            log_prob = dist.log_prob(alpha)  # (bs,)

            # Generate answers with the sampled linear combination of prompts
            with torch.no_grad():
                predictions = model.generate(
                    encoded.input_ids, encoded.attention_mask, alpha, max_new_tokens=256
                )

            # Binary correctness reward: r(x) = 1 if correct, 0 otherwise
            rewards = torch.zeros(bs, device=model.device)
            for j, (pred, gold) in enumerate(zip(predictions, gold_answers)):
                if check_answer(pred, gold):
                    rewards[j] = 1.0

            # REINFORCE: loss = -E[r(x, alpha) * log pi_theta(alpha | x)]
            loss = -(log_prob * rewards).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_reward += rewards.mean().item()
            total_loss += loss.item()
            n_steps += 1
            global_step += 1
            writer.add_scalar("train/loss", loss.item(), global_step)

        avg_reward = total_reward / max(n_steps, 1)
        avg_loss = total_loss / max(n_steps, 1)

        # Evaluate with deterministic weights on train and test subsets
        model.mixer.eval()
        train_acc = _eval_accuracy(model, dataset, n_samples=200, batch_size=32)
        test_acc = _eval_accuracy(model, test_dataset, n_samples=200, batch_size=32)
        model.mixer.train()

        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("test/accuracy", test_acc, epoch)
        print(f"Epoch {epoch+1}: loss = {avg_loss:.4f}, train acc = {train_acc:.4f}, test acc = {test_acc:.4f}")

        # Save checkpoint after each epoch
        os.makedirs("checkpoints", exist_ok=True)
        ckpt = {
            "epoch": epoch,
            "mixer_state_dict": model.mixer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
        }
        torch.save(ckpt, "checkpoints/rl_latest.pt")
        if test_acc > best_accuracy:
            best_accuracy = avg_reward
            torch.save(ckpt, "checkpoints/rl_best.pt")
            print(f"  New best accuracy: {best_accuracy:.4f}")

    writer.close()
    return model


@torch.no_grad()
def _eval_accuracy(model, dataset, n_samples=200, batch_size=32):
    """Quick accuracy check on a subset via generation with deterministic weights."""
    n = min(n_samples, len(dataset))
    correct = 0
    for i in range(0, n, batch_size):
        batch_items = [dataset[j] for j in range(i, min(i + batch_size, n))]
        questions = [item["question"] for item in batch_items]
        gold_answers = [item["answer"] for item in batch_items]

        encoded = model.tokenizer(
            questions, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(model.device)

        pooled = model.get_pooled_input(encoded.input_ids, encoded.attention_mask)
        alpha = model.mixer(pooled)
        predictions = model.generate(encoded.input_ids, encoded.attention_mask, alpha, max_new_tokens=256)

        for pred, gold in zip(predictions, gold_answers):
            if check_answer(pred, gold):
                correct += 1

    return correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--save_path", type=str, default="checkpoints/rl_mixer.pt")
    parser.add_argument("--concentration_scale", type=float, default=20.0,
                        help="Dirichlet concentration scale (higher = less exploration)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PromptMixingModel(model_name=args.model_name, device=device)

    start_epoch = 0
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location=device)
        model.mixer.load_state_dict(resume_checkpoint["mixer_state_dict"])
        start_epoch = resume_checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    model = train_rl(model, epochs=args.epochs, batch_size=args.batch_size,
                     lr=args.lr, max_samples=args.max_samples,
                     start_epoch=start_epoch, resume_checkpoint=resume_checkpoint,
                     concentration_scale=args.concentration_scale)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.mixer.state_dict(), args.save_path)
    print(f"Saved mixer to {args.save_path}")


if __name__ == "__main__":
    main()
