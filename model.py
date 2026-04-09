"""Prompt-mixing model: learns linear combinations of discrete prompt embeddings."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import PROMPT_BANK, K

# A100 TF32: same accuracy as FP32, ~3x faster matmuls
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class PromptMixer(nn.Module):
    """Small MLP that maps a pooled input embedding to prompt-bank weights."""

    def __init__(self, embed_dim: int, hidden_dim: int = 256, num_prompts: int = K):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_prompts),
        )

    def forward(self, pooled_input: torch.Tensor) -> torch.Tensor:
        """Returns softmax weights over the prompt bank. Shape: (batch, K)."""
        return torch.softmax(self.net(pooled_input), dim=-1)


class PromptMixingModel(nn.Module):
    """Frozen LLM backbone + learnable prompt-mixing head."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "cuda"):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.embed_dim = self.backbone.config.hidden_size
        self.mixer = PromptMixer(self.embed_dim).to(device)

        # Pre-embed the prompt bank (each prompt -> sequence of token embeddings)
        self._embed_prompt_bank()

    @torch.no_grad()
    def _embed_prompt_bank(self):
        """Tokenize and embed each discrete prompt. Store as (K, max_len, d)."""
        embed_layer = self.backbone.get_input_embeddings()
        encoded = []
        for prompt_text in PROMPT_BANK:
            tokens = self.tokenizer(prompt_text, return_tensors="pt", padding=False).input_ids.to(self.device)
            emb = embed_layer(tokens).squeeze(0)  # (seq_len, d)
            encoded.append(emb)

        # Pad to same length
        max_len = max(e.size(0) for e in encoded)
        padded = []
        for e in encoded:
            pad_len = max_len - e.size(0)
            if pad_len > 0:
                padding = torch.zeros(pad_len, self.embed_dim, device=self.device, dtype=e.dtype)
                e = torch.cat([e, padding], dim=0)
            padded.append(e)

        self.prompt_embeddings = torch.stack(padded)  # (K, max_prompt_len, d)
        self.prompt_len = max_len

    def get_pooled_input(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool input token embeddings using attention mask. Returns (batch, d)."""
        embed_layer = self.backbone.get_input_embeddings()
        input_embeds = embed_layer(input_ids)  # (batch, seq_len, d)
        mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        pooled = (input_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled.float()  # cast to float32 for the mixer MLP

    def get_mixed_prompt(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute P(x) = sum_k alpha_k * p_k. Returns (batch, prompt_len, d)."""
        # alpha: (batch, K), prompt_embeddings: (K, prompt_len, d)
        # Einstein summation: batch over k, keep prompt_len and d
        mixed = torch.einsum("bk,kld->bld", alpha.to(self.prompt_embeddings.dtype), self.prompt_embeddings)
        return mixed

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 alpha: torch.Tensor, max_new_tokens: int = 512) -> list[str]:
        """Generate text with the mixed prompt prepended to input embeddings."""
        embed_layer = self.backbone.get_input_embeddings()
        input_embeds = embed_layer(input_ids)  # (batch, seq_len, d)
        prompt_embeds = self.get_mixed_prompt(alpha)  # (batch, prompt_len, d)

        # Concatenate: [P(x); E(x)]
        combined_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)

        # Build attention mask for the combined sequence
        prompt_mask = torch.ones(input_ids.size(0), self.prompt_len, device=self.device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        outputs = self.backbone.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
