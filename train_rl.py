"""REINFORCE training: learn prompt-mixing weights via policy gradient.

Implements the REINFORCE-style policy-gradient objective from the proposal:
    nabla_theta J = E[r(x, alpha) * nabla_theta log pi_theta(alpha | x)]

where pi_theta(alpha | x) is a Categorical distribution over prompts induced
by the softmax output of the mixer network h. The action is sampling a single
prompt index k ~ Categorical(alpha(x)), and the weight vector used for generation
is the one-hot selection of that prompt. This keeps the policy exactly as described:
the softmax output IS the policy, and log pi is log alpha_k for the chosen k.
"""

import argparse
import os
import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import GSM8KDataset, check_answer
from model import PromptMixingModel
from prompts import K


def train_rl(model: PromptMixingModel, epochs: int = 10, batch_size: int = 4,
             lr: float = 1e-3, max_samples: int = -1, log_dir: str = "runs/rl",
             start_epoch: int = 0, resume_checkpoint: dict = None):
    """
    REINFORCE training of the prompt mixer.

    The mixer outputs alpha(x) = softmax(h(e_bar(x))) which defines a Categorical
    policy over prompt indices. We sample a prompt, generate an answer, and use
    binary correctness as the reward.
    """
    dataset = GSM8KDataset(split="train")
    if max_samples > 0:
        dataset.questions = dataset.questions[:max_samples]
        dataset.answers = dataset.answers[:max_samples]

    optimizer = torch.optim.Adam(model.mixer.parameters(), lr=lr)
    if resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
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
            alpha = model.mixer(pooled)  # (bs, K)

            # Policy pi_theta(alpha | x) is Categorical over prompt indices
            dist = Categorical(probs=alpha)
            sampled_k = dist.sample()  # (bs,) indices into prompt bank
            log_prob = dist.log_prob(sampled_k)  # (bs,)

            # Build one-hot weight vectors for generation
            action_alpha = torch.zeros(bs, K, device=model.device)
            action_alpha.scatter_(1, sampled_k.unsqueeze(1), 1.0)

            # Generate answers with the selected prompts
            with torch.no_grad():
                predictions = model.generate(
                    encoded.input_ids, encoded.attention_mask, action_alpha, max_new_tokens=512
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
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        writer.add_scalar("train/accuracy", avg_reward, epoch)
        print(f"Epoch {epoch+1}: avg loss = {avg_loss:.4f}, accuracy = {avg_reward:.4f}")

        # Save checkpoint after each epoch
        os.makedirs("checkpoints", exist_ok=True)
        ckpt = {
            "epoch": epoch,
            "mixer_state_dict": model.mixer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "accuracy": avg_reward,
        }
        torch.save(ckpt, "checkpoints/rl_latest.pt")
        if avg_reward > best_accuracy:
            best_accuracy = avg_reward
            torch.save(ckpt, "checkpoints/rl_best.pt")
            print(f"  New best accuracy: {best_accuracy:.4f}")

    writer.close()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--save_path", type=str, default="checkpoints/rl_mixer.pt")
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
                     start_epoch=start_epoch, resume_checkpoint=resume_checkpoint)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.mixer.state_dict(), args.save_path)
    print(f"Saved mixer to {args.save_path}")


if __name__ == "__main__":
    main()
