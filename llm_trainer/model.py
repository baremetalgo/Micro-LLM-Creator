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


class RMSNorm(nn.Module):
    """Root mean square normalization used by Llama-style models."""

    def __init__(self, size: int, eps: float = 1e-6) -> None:
        """Create RMSNorm.

        Args:
            size: Feature dimension.
            eps: Numerical stability value.
        """

        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Normalize by root mean square.

        Args:
            value: Input tensor.

        Returns:
            Normalized tensor.
        """

        return self.weight * value * torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + self.eps)


def make_norm(config: ModelConfig) -> nn.Module:
    """Create the configured normalization layer.

    Args:
        config: Model configuration.

    Returns:
        Normalization module.
    """

    if config.norm_type == "rmsnorm":
        return RMSNorm(config.embedding_size)
    return LayerNorm(config.embedding_size, bias=config.bias)


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding cache for attention heads."""

    def __init__(self, head_size: int, context_length: int, theta: float) -> None:
        """Create RoPE caches.

        Args:
            head_size: Attention head dimension.
            context_length: Maximum context length.
            theta: Frequency base.
        """

        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_size, 2).float() / head_size))
        positions = torch.arange(context_length, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, start_pos: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors.

        Args:
            query: Query tensor with shape ``[batch, heads, tokens, head_size]``.
            key: Key tensor with shape ``[batch, heads, tokens, head_size]``.
            start_pos: Absolute starting token position.

        Returns:
            Rotated query and key tensors.
        """

        token_count = query.size(-2)
        cos = self.cos[:, :, start_pos : start_pos + token_count, :]
        sin = self.sin[:, :, start_pos : start_pos + token_count, :]
        return (query * cos) + (_rotate_half(query) * sin), (key * cos) + (_rotate_half(key) * sin)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension in RoPE pairs.

    Args:
        value: Tensor to rotate.

    Returns:
        Rotated tensor.
    """

    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


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
        self.position_encoding = config.position_encoding
        self.c_attn = nn.Linear(config.embedding_size, 3 * config.embedding_size, bias=config.bias)
        self.c_proj = nn.Linear(config.embedding_size, config.embedding_size, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        head_size = config.embedding_size // config.head_count
        self.rotary = (
            RotaryEmbedding(head_size, config.context_length, config.rope_theta)
            if config.position_encoding == "rope"
            else None
        )
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.context_length, config.context_length)).view(
                1, 1, config.context_length, config.context_length
            ),
        )

    def forward(
        self,
        value: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Apply causal self-attention.

        Args:
            value: Input hidden states.
            past_kv: Optional cached key/value tensors.
            start_pos: Absolute starting token position.
            use_cache: Whether to return updated key/value cache.

        Returns:
            Attention output tensor, plus cache when requested.
        """

        batch_size, token_count, channel_count = value.size()
        qkv = self.c_attn(value)
        query, key, val = qkv.split(self.embedding_size, dim=2)
        head_size = channel_count // self.head_count

        key = key.view(batch_size, token_count, self.head_count, head_size).transpose(1, 2)
        query = query.view(batch_size, token_count, self.head_count, head_size).transpose(1, 2)
        val = val.view(batch_size, token_count, self.head_count, head_size).transpose(1, 2)
        if self.rotary is not None:
            query, key = self.rotary(query, key, start_pos=start_pos)

        if past_kv is not None:
            past_key, past_val = past_kv
            key = torch.cat((past_key, key), dim=-2)
            val = torch.cat((past_val, val), dim=-2)
            if key.size(-2) > self.mask.size(-1):
                key = key[:, :, -self.mask.size(-1) :, :]
                val = val[:, :, -self.mask.size(-1) :, :]
        present = (key, val)

        attention = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        key_count = key.size(-2)
        if past_kv is None:
            mask = self.mask[:, :, :token_count, :key_count]
        else:
            start = max(0, key_count - token_count)
            mask = self.mask[:, :, start : start + token_count, :key_count]
        attention = attention.masked_fill(mask == 0, float("-inf"))
        attention = F.softmax(attention, dim=-1)
        attention = self.attn_dropout(attention)

        y = attention @ val
        y = y.transpose(1, 2).contiguous().view(batch_size, token_count, channel_count)
        output = self.resid_dropout(self.c_proj(y))
        if use_cache:
            return output, present
        return output


class MLP(nn.Module):
    """Feed-forward network inside a transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        """Create feed-forward network.

        Args:
            config: Model architecture configuration.
        """

        super().__init__()
        self.mlp_type = config.mlp_type
        hidden_size = 4 * config.embedding_size
        if self.mlp_type == "swiglu":
            self.w1 = nn.Linear(config.embedding_size, hidden_size, bias=config.bias)
            self.w2 = nn.Linear(hidden_size, config.embedding_size, bias=config.bias)
            self.w3 = nn.Linear(config.embedding_size, hidden_size, bias=config.bias)
            self.dropout = nn.Dropout(config.dropout)
        else:
            self.net = nn.Sequential(
                nn.Linear(config.embedding_size, hidden_size, bias=config.bias),
                nn.GELU(),
                nn.Linear(hidden_size, config.embedding_size, bias=config.bias),
                nn.Dropout(config.dropout),
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformation.

        Args:
            value: Input hidden states.

        Returns:
            Transformed hidden states.
        """

        if self.mlp_type == "swiglu":
            return self.dropout(self.w2(F.silu(self.w1(value)) * self.w3(value)))
        return self.net(value)


class Block(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(self, config: ModelConfig) -> None:
        """Create a transformer block.

        Args:
            config: Model architecture configuration.
        """

        super().__init__()
        self.ln_1 = make_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = make_norm(config)
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

    def forward_with_cache(
        self,
        value: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Apply transformer block and return updated KV cache.

        Args:
            value: Input hidden states.
            past_kv: Optional cached key/value tensors.
            start_pos: Absolute starting token position.

        Returns:
            Updated hidden states and key/value cache.
        """

        attention_output, present = self.attn(self.ln_1(value), past_kv=past_kv, start_pos=start_pos, use_cache=True)
        value = value + attention_output
        value = value + self.mlp(self.ln_2(value))
        return value, present


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
        self.position_embedding = (
            nn.Embedding(config.context_length, config.embedding_size)
            if config.position_encoding == "learned"
            else None
        )
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.layer_count)])
        self.ln_f = make_norm(config)
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
        value = self.token_embedding(idx)
        if self.position_embedding is not None:
            positions = torch.arange(0, token_count, dtype=torch.long, device=idx.device)
            value = value + self.position_embedding(positions)
        value = self.drop(value)
        value = self.blocks(value)
        value = self.ln_f(value)
        return self.lm_head(value)

    def forward_with_cache(
        self,
        idx: torch.Tensor,
        past_kv: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Run a forward pass and return updated KV cache.

        Args:
            idx: Token IDs with shape ``[batch, tokens]``.
            past_kv: Optional per-layer key/value cache.
            start_pos: Absolute starting token position.

        Returns:
            Logits and updated per-layer KV cache.
        """

        _, token_count = idx.size()
        if token_count > self.config.context_length:
            raise ValueError("Input sequence is longer than context_length")
        value = self.token_embedding(idx)
        if self.position_embedding is not None:
            positions = torch.arange(start_pos, start_pos + token_count, dtype=torch.long, device=idx.device)
            positions = positions.clamp(max=self.config.context_length - 1)
            value = value + self.position_embedding(positions)
        value = self.drop(value)
        next_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, block in enumerate(self.blocks):
            layer_cache = past_kv[index] if past_kv is not None and index < len(past_kv) else None
            value, present = block.forward_with_cache(value, past_kv=layer_cache, start_pos=start_pos)
            next_cache.append(present)
        value = self.ln_f(value)
        return self.lm_head(value), next_cache

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        use_kv_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressively sample new tokens.

        Args:
            idx: Starting token IDs.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Optional top-k cutoff.
            use_kv_cache: Whether to reuse key/value tensors during generation.

        Returns:
            Token IDs including the original context and generated tokens.
        """

        past_kv: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.context_length :]
            if use_kv_cache and past_kv is None:
                logits, past_kv = self.forward_with_cache(idx_cond, start_pos=0)
                logits = logits[:, -1, :] / max(temperature, 1e-5)
            elif use_kv_cache and idx.size(1) < self.config.context_length:
                logits, past_kv = self.forward_with_cache(idx[:, -1:], past_kv=past_kv, start_pos=idx.size(1) - 1)
                logits = logits[:, -1, :] / max(temperature, 1e-5)
            else:
                past_kv = None
                logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
