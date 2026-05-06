"""Supervised training: learn prompt-mixing weights from labeled GSM8K examples.

The proposal says: "In the supervised setting, the prompt weights are learned
directly from labeled examples." We implement this by using cross-entropy loss
against the gold answer. For each training example, we forward the mixed prompt
through the backbone and train the mixer to minimize the language modeling loss
on the gold solution tokens.
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import GSM8KDataset, check_answer
from model import PromptMixingModel
from prompts import K


def train_supervised(model: PromptMixingModel, epochs: int = 5, batch_size: int = 4,
                     lr: float = 1e-3, max_samples: int = -1, log_dir: str = "runs/supervised",
                     start_epoch: int = 0, resume_checkpoint: dict = None,
                     checkpoint_dir: str = "checkpoints"):
    """
    Supervised training of the prompt mixer.

    Uses cross-entropy on the gold answer as the training signal. The mixer
    learns which prompt combinations help the backbone produce the correct answer.

    Args:
        checkpoint_dir: directory where supervised_latest.pt and supervised_best.pt
            are written after every epoch. Pass an experiment-specific path (e.g.
            "../results/experiment2") to keep runs isolated.
    """
    dataset = GSM8KDataset(split="train")
    if max_samples > 0:
        dataset.questions = dataset.questions[:max_samples]
        dataset.answers = dataset.answers[:max_samples]

    optimizer = torch.optim.Adam(model.mixer.parameters(), lr=lr)
    if resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
    scaler = GradScaler()
    writer = SummaryWriter(log_dir)
    global_step = resume_checkpoint["global_step"] if resume_checkpoint else 0

    test_dataset = GSM8KDataset(split="test")

    model.mixer.train()
    # Backbone stays frozen but we need forward pass for loss
    model.backbone.eval()
    model.backbone.gradient_checkpointing_enable()
    best_accuracy = 0.0

    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        n_batches = 0

        indices = torch.randperm(len(dataset))
        for i in tqdm(range(0, len(dataset), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            batch_idx = indices[i:i+batch_size].tolist()
            questions = [dataset.questions[j] for j in batch_idx]
            answers = [dataset.answers[j] for j in batch_idx]

            # Tokenize questions
            q_encoded = model.tokenizer(
                questions, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(model.device)

            # Tokenize answers (targets)
            a_encoded = model.tokenizer(
                answers, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(model.device)

            # Get mixed prompt via mixer
            pooled = model.get_pooled_input(q_encoded.input_ids, q_encoded.attention_mask)
            alpha = model.mixer(pooled)  # (bs, K)
            prompt_embeds = model.get_mixed_prompt(alpha)  # (bs, prompt_len, d)

            # Build input: [P(x); E(x); E(y)]
            embed_layer = model.backbone.get_input_embeddings()
            q_embeds = embed_layer(q_encoded.input_ids)
            a_embeds = embed_layer(a_encoded.input_ids)

            combined_embeds = torch.cat([prompt_embeds, q_embeds, a_embeds], dim=1)

            # Build attention mask
            bs = len(questions)
            prompt_mask = torch.ones(bs, model.prompt_len, device=model.device, dtype=q_encoded.attention_mask.dtype)
            combined_mask = torch.cat([prompt_mask, q_encoded.attention_mask, a_encoded.attention_mask], dim=1)

            # Build labels: -100 for prompt and question tokens, actual token ids for answer
            prompt_labels = torch.full((bs, model.prompt_len), -100, device=model.device, dtype=torch.long)
            q_labels = torch.full_like(q_encoded.input_ids, -100)
            a_labels = a_encoded.input_ids.clone()
            a_labels[a_encoded.attention_mask == 0] = -100
            labels = torch.cat([prompt_labels, q_labels, a_labels], dim=1)

            # Shift labels for causal LM (predict next token)
            # labels should be shifted right by 1 relative to inputs
            shift_labels = labels[:, 1:].contiguous()

            # Forward through frozen backbone (gradients flow through prompt_embeds -> mixer)
            with autocast("cuda"):
                outputs = model.backbone(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_mask,
                )
                shift_logits = outputs.logits[:, :-1, :].contiguous()

                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1
            global_step += 1
            writer.add_scalar("train/loss", loss.item(), global_step)

            # Log batch-averaged deterministic mixer output (20D weight vector)
            # for per-step convergence analysis.
            batch_alpha_mean = alpha.detach().mean(dim=0)
            for j, v in enumerate(batch_alpha_mean.tolist()):
                writer.add_scalar(f"alpha/dim_{j:02d}", v, global_step)

        avg_loss = total_loss / max(n_batches, 1)
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)

        # Accuracy check on subsets of train and test
        model.mixer.eval()
        train_acc = _eval_accuracy(model, dataset, n_samples=200, batch_size=32)
        test_acc = _eval_accuracy(model, test_dataset, n_samples=200, batch_size=32)
        model.mixer.train()
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("test/accuracy", test_acc, epoch)
        print(f"Epoch {epoch+1}: loss = {avg_loss:.4f}, train acc = {train_acc:.4f}, test acc = {test_acc:.4f}")

        # Save checkpoint after each epoch (in the per-run checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt = {
            "epoch": epoch,
            "mixer_state_dict": model.mixer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
        }
        torch.save(ckpt, os.path.join(checkpoint_dir, "supervised_latest.pt"))
        torch.save(ckpt, os.path.join(checkpoint_dir, f"supervised_epoch_{epoch:03d}.pt"))
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(ckpt, os.path.join(checkpoint_dir, "supervised_best.pt"))
            print(f"  New best accuracy: {best_accuracy:.4f}")

    writer.close()
    return model


@torch.no_grad()
def _eval_accuracy(model, dataset, n_samples=128, batch_size=32):
    """Quick accuracy check on a subset via generation."""
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
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--save_path", type=str, default="checkpoints/supervised_mixer.pt",
                        help="Path for the final mixer state_dict (saved once after training).")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory for per-epoch best/latest checkpoints "
                             "(supervised_latest.pt and supervised_best.pt). Use an "
                             "experiment-specific path (e.g. ../results/experiment2) "
                             "to avoid clobbering checkpoints from other runs.")
    parser.add_argument("--log_dir", type=str, default="runs/supervised",
                        help="TensorBoard log directory.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from a checkpoint (loads mixer + optimizer "
                             "state, continues epoch counter). Use for crash recovery "
                             "within a run.")
    parser.add_argument("--init_from", type=str, default=None,
                        help="Initialize mixer weights from a checkpoint, but start fresh "
                             "(new optimizer, epoch counter at 0, fresh logs). Use for "
                             "warm-starting experiment 2 from experiment 1's trained mixer.")
    args = parser.parse_args()

    assert not (args.resume and args.init_from), (
        "Cannot use --resume and --init_from together. They serve different purposes: "
        "--resume continues an interrupted run, --init_from starts a fresh run from "
        "pre-trained weights."
    )

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
    elif args.init_from:
        init_checkpoint = torch.load(args.init_from, map_location=device)
        # Handle both formats: full checkpoint dict (supervised_best.pt etc.)
        # and raw state_dict (supervised_mixer.pt saved at end of training).
        if isinstance(init_checkpoint, dict) and "mixer_state_dict" in init_checkpoint:
            model.mixer.load_state_dict(init_checkpoint["mixer_state_dict"])
        else:
            model.mixer.load_state_dict(init_checkpoint)
        print(f"Initialized mixer weights from {args.init_from}")
        print("Starting fresh: new optimizer state, epoch counter at 0, fresh logs.")

    model = train_supervised(model, epochs=args.epochs, batch_size=args.batch_size,
                             lr=args.lr, max_samples=args.max_samples,
                             log_dir=args.log_dir,
                             start_epoch=start_epoch, resume_checkpoint=resume_checkpoint,
                             checkpoint_dir=args.checkpoint_dir)

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(model.mixer.state_dict(), args.save_path)
    print(f"Saved mixer to {args.save_path}")


if __name__ == "__main__":
    main()
