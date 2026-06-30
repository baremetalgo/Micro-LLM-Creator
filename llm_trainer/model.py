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


class LoRALinear(nn.Module):
    """Linear layer with trainable low-rank LoRA adapters."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        """Create a LoRA wrapper around an existing linear layer.

        Args:
            base: Frozen base linear layer.
            rank: Adapter rank.
            alpha: LoRA scaling alpha.
            dropout: Dropout probability before the adapter.
        """

        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the base projection plus LoRA update.

        Args:
            value: Input tensor.

        Returns:
            Projected tensor.
        """

        update = F.linear(F.linear(self.dropout(value), self.lora_a), self.lora_b) * self.scaling
        return self.base(value) + update

    def merged_linear(self) -> nn.Linear:
        """Return a plain linear layer with LoRA weights merged.

        Returns:
            Linear layer equivalent to base plus LoRA update.
        """

        merged = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        merged.weight.data.copy_(self.base.weight.data + (self.lora_b @ self.lora_a) * self.scaling)
        if self.base.bias is not None and merged.bias is not None:
            merged.bias.data.copy_(self.base.bias.data)
        return merged


def _set_nested_module(root: nn.Module, module_name: str, module: nn.Module) -> None:
    """Replace a nested module by dotted name.

    Args:
        root: Root module.
        module_name: Dotted module name.
        module: Replacement module.
    """

    parent_name, child_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, module)


def _lora_target_names(model: nn.Module, target_modules: str) -> set[str]:
    """Resolve LoRA target module names.

    Args:
        model: Model to inspect.
        target_modules: Comma-separated target groups.

    Returns:
        Set of module names to wrap.
    """

    groups = {part.strip().lower() for part in target_modules.split(",") if part.strip()}
    if "all" in groups:
        groups.update({"attention", "mlp"})
    names: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.endswith("lm_head"):
            continue
        is_attention = ".attn." in name
        is_mlp = ".mlp." in name
        if ("attention" in groups and is_attention) or ("mlp" in groups and is_mlp):
            names.add(name)
    return names


def apply_lora_adapters(model: nn.Module, rank: int, alpha: float, dropout: float, target_modules: str) -> int:
    """Attach LoRA adapters to selected linear layers.

    Args:
        model: Model to modify in place.
        rank: LoRA rank.
        alpha: LoRA alpha.
        dropout: LoRA dropout.
        target_modules: Comma-separated target groups.

    Returns:
        Number of wrapped modules.
    """

    names = _lora_target_names(model, target_modules)
    for name in sorted(names):
        module = model.get_submodule(name)
        if isinstance(module, nn.Linear):
            _set_nested_module(model, name, LoRALinear(module, rank, alpha, dropout))
    return len(names)


def freeze_non_lora_parameters(model: nn.Module) -> None:
    """Freeze all parameters except LoRA adapter parameters.

    Args:
        model: Model to update.
    """

    for name, parameter in model.named_parameters():
        parameter.requires_grad_(("lora_a" in name) or ("lora_b" in name))


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return trainable LoRA adapter tensors.

    Args:
        model: Model containing LoRA adapters.

    Returns:
        LoRA-only state dictionary.
    """

    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if ".lora_a" in name or ".lora_b" in name
    }


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load LoRA adapter tensors into a model.

    Args:
        model: Model containing LoRA adapters.
        state: LoRA-only state dictionary.
    """

    model.load_state_dict(state, strict=False)


def merge_lora_adapters(model: nn.Module) -> int:
    """Merge LoRA adapters into plain linear layers.

    Args:
        model: Model to modify in place.

    Returns:
        Number of merged LoRA modules.
    """

    merged = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            _set_nested_module(model, name, module.merged_linear())
            merged += 1
    return merged


def lora_parameter_count(model: nn.Module) -> int:
    """Count trainable LoRA parameters.

    Args:
        model: Model containing LoRA adapters.

    Returns:
        Number of trainable adapter parameters.
    """

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


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
        self.kv_head_count = config.resolved_kv_head_count()
        self.embedding_size = config.embedding_size
        self.position_encoding = config.position_encoding
        self.attention_backend = config.attention_backend
        self.attention_window = config.attention_window
        self.head_size = config.embedding_size // config.head_count
        self.kv_embedding_size = self.kv_head_count * self.head_size
        self.c_attn = nn.Linear(
            config.embedding_size,
            config.embedding_size + (2 * self.kv_embedding_size),
            bias=config.bias,
        )
        self.c_proj = nn.Linear(config.embedding_size, config.embedding_size, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rotary = (
            RotaryEmbedding(self.head_size, config.context_length, config.rope_theta)
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
        query, key, val = qkv.split((self.embedding_size, self.kv_embedding_size, self.kv_embedding_size), dim=2)

        key = key.view(batch_size, token_count, self.kv_head_count, self.head_size).transpose(1, 2)
        query = query.view(batch_size, token_count, self.head_count, self.head_size).transpose(1, 2)
        val = val.view(batch_size, token_count, self.kv_head_count, self.head_size).transpose(1, 2)
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
        expanded_key = self._expand_kv(key)
        expanded_val = self._expand_kv(val)

        key_count = expanded_key.size(-2)
        if past_kv is None:
            mask = self.mask[:, :, :token_count, :key_count]
        else:
            start = max(0, key_count - token_count)
            mask = self.mask[:, :, start : start + token_count, :key_count]
        if self.attention_window > 0:
            positions = torch.arange(key_count, device=value.device)
            query_positions = torch.arange(key_count - token_count, key_count, device=value.device)
            window_mask = positions[None, :] >= (query_positions[:, None] - self.attention_window + 1)
            mask = mask & window_mask.view(1, 1, token_count, key_count)

        if self.attention_backend == "sdpa" and hasattr(F, "scaled_dot_product_attention"):
            attn_mask = mask[:, :, :, :].bool()
            y = F.scaled_dot_product_attention(
                query,
                expanded_key,
                expanded_val,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
        else:
            attention = (query @ expanded_key.transpose(-2, -1)) * (1.0 / math.sqrt(expanded_key.size(-1)))
            attention = attention.masked_fill(mask == 0, float("-inf"))
            attention = F.softmax(attention, dim=-1)
            attention = self.attn_dropout(attention)
            y = attention @ expanded_val
        y = y.transpose(1, 2).contiguous().view(batch_size, token_count, channel_count)
        output = self.resid_dropout(self.c_proj(y))
        if use_cache:
            return output, present
        return output

    def _expand_kv(self, value: torch.Tensor) -> torch.Tensor:
        """Expand grouped key/value heads to query head count.

        Args:
            value: Key or value tensor with key/value head count.

        Returns:
            Tensor with one key/value head per query head.
        """

        if self.kv_head_count == self.head_count:
            return value
        repeat_count = self.head_count // self.kv_head_count
        return value.repeat_interleave(repeat_count, dim=1)


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
