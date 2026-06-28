from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class LayerNorm(nn.Module):
    """Layer normalization with optional bias."""

    def __init__(self, size: int, bias: bool) -> None:
        """Create layer normalization.

        Args:
            size: Feature dimension.
            bias: Whether to include a bias vector.
        """

        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size)) if bias else None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Normalize an input tensor.

        Args:
            value: Tensor to normalize.

        Returns:
            Normalized tensor.
        """

        return F.layer_norm(value, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention block."""

    def __init__(self, config: ModelConfig) -> None:
        """Create attention module.

        Args:
            config: Model architecture configuration.
        """

        super().__init__()
        self.head_count = config.head_count
        self.embedding_size = config.embedding_size
        self.c_attn = nn.Linear(config.embedding_size, 3 * config.embedding_size, bias=config.bias)
        self.c_proj = nn.Linear(config.embedding_size, config.embedding_size, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.context_length, config.context_length)).view(
                1, 1, config.context_length, config.context_length
            ),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply causal self-attention.

        Args:
            value: Input hidden states.

        Returns:
            Attention output tensor.
        """

        batch_size, token_count, channel_count = value.size()
        qkv = self.c_attn(value)
        query, key, val = qkv.split(self.embedding_size, dim=2)
        head_size = channel_count // self.head_count

        key = key.view(batch_size, token_count, self.head_count, head_size).transpose(1, 2)
        query = query.view(batch_size, token_count, self.head_count, head_size).transpose(1, 2)
        val = val.view(batch_size, token_count, self.head_count, head_size).transpose(1, 2)

        attention = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        attention = attention.masked_fill(self.mask[:, :, :token_count, :token_count] == 0, float("-inf"))
        attention = F.softmax(attention, dim=-1)
        attention = self.attn_dropout(attention)

        y = attention @ val
        y = y.transpose(1, 2).contiguous().view(batch_size, token_count, channel_count)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Feed-forward network inside a transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        """Create feed-forward network.

        Args:
            config: Model architecture configuration.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embedding_size, 4 * config.embedding_size, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.embedding_size, config.embedding_size, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformation.

        Args:
            value: Input hidden states.

        Returns:
            Transformed hidden states.
        """

        return self.net(value)


class Block(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(self, config: ModelConfig) -> None:
        """Create a transformer block.

        Args:
            config: Model architecture configuration.
        """

        super().__init__()
        self.ln_1 = LayerNorm(config.embedding_size, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.embedding_size, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply transformer block.

        Args:
            value: Input hidden states.

        Returns:
            Updated hidden states.
        """

        value = value + self.attn(self.ln_1(value))
        value = value + self.mlp(self.ln_2(value))
        return value


class MicroGPT(nn.Module):
    """Small GPT-style causal language model."""

    def __init__(self, config: ModelConfig) -> None:
        """Create the model.

        Args:
            config: Model architecture configuration.
        """

        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_size)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.layer_count)])
        self.ln_f = LayerNorm(config.embedding_size, bias=config.bias)
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize module weights.

        Args:
            module: Module to initialize.
        """

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            idx: Token IDs with shape ``[batch, tokens]``.

        Returns:
            Logits with shape ``[batch, tokens, vocab]``.

        Raises:
            ValueError: If the sequence is longer than context length.
        """

        _, token_count = idx.size()
        if token_count > self.config.context_length:
            raise ValueError("Input sequence is longer than context_length")
        positions = torch.arange(0, token_count, dtype=torch.long, device=idx.device)
        value = self.drop(self.token_embedding(idx) + self.position_embedding(positions))
        value = self.blocks(value)
        value = self.ln_f(value)
        return self.lm_head(value)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
    ) -> torch.Tensor:
        """Autoregressively sample new tokens.

        Args:
            idx: Starting token IDs.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Optional top-k cutoff.

        Returns:
            Token IDs including the original context and generated tokens.
        """

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.context_length :]
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
