from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch


@dataclass
class DatasetConfig:
    """Configuration for building a tokenizer-ready dataset.

    Attributes:
        input_dir: Folder containing source PDFs, text, JSONL, or code files.
        output_dir: Folder where prepared dataset artifacts are written.
        vocab_size: Optional manual tokenizer vocabulary size.
        min_frequency: Minimum token frequency for BPE vocabulary entries.
        context_length: Token window length used by downstream training.
        validation_split: Fraction of tokens reserved for validation.
        lowercase: Whether to lowercase text during ingestion.
        max_workers: Number of parallel file readers.
        code_training_mode: Enables code/prose tagging and code preservation.
        include_prose: Keeps prose/explanation samples when code mode is active.
        include_source_code: Includes source-code files when code mode is active.
        extract_code_blocks: Detects code-like blocks in PDFs/text.
        preserve_indentation: Keeps code line breaks and indentation.
        generate_instruction_samples: Wraps code with simple instruction tags.
        reasoning_sample_mode: Instruction/reasoning format: none, scaffold, or detailed.
        prepare_mode: Dataset update mode: incremental, full_rebuild, or force_reprocess.
        tokenizer_strategy: Tokenizer policy: auto, train_new, reuse_dataset, or import_tokenizer.
        tokenizer_path: Optional existing tokenizer JSON used by import_tokenizer.
    """

    input_dir: Path
    output_dir: Path
    vocab_size: Optional[int] = None
    min_frequency: int = 2
    context_length: int = 128
    validation_split: float = 0.1
    lowercase: bool = False
    max_workers: int = 4
    code_training_mode: bool = False
    include_prose: bool = True
    include_source_code: bool = True
    extract_code_blocks: bool = True
    preserve_indentation: bool = True
    generate_instruction_samples: bool = True
    reasoning_sample_mode: str = "scaffold"
    prepare_mode: str = "incremental"
    tokenizer_strategy: str = "auto"
    tokenizer_path: Optional[Path] = None


@dataclass
class ModelConfig:
    """Configuration for the GPT-style model architecture.

    Attributes:
        vocab_size: Tokenizer vocabulary size.
        context_length: Maximum tokens visible to the model at once.
        embedding_size: Width of token embeddings and transformer channels.
        head_count: Number of causal attention heads.
        layer_count: Number of transformer blocks.
        dropout: Dropout probability for regularization.
        bias: Whether linear and normalization layers include bias terms.
        norm_type: Normalization type: layernorm or rmsnorm.
        position_encoding: Position encoding type: learned or rope.
        mlp_type: Feed-forward type: gelu or swiglu.
        rope_theta: RoPE frequency base when position_encoding is rope.
    """

    vocab_size: int
    context_length: int = 128
    embedding_size: int = 256
    head_count: int = 4
    layer_count: int = 4
    dropout: float = 0.1
    bias: bool = True
    norm_type: str = "layernorm"
    position_encoding: str = "learned"
    mlp_type: str = "gelu"
    rope_theta: float = 10000.0

    def validate(self) -> None:
        """Validate architecture constraints.

        Raises:
            ValueError: If dimensions are incompatible or too small.
        """

        if self.embedding_size % self.head_count != 0:
            raise ValueError("embedding_size must be divisible by head_count")
        if self.context_length < 8:
            raise ValueError("context_length must be at least 8")
        if self.vocab_size < 16:
            raise ValueError("vocab_size is too small for language modeling")
        if self.norm_type not in {"layernorm", "rmsnorm"}:
            raise ValueError("norm_type must be layernorm or rmsnorm")
        if self.position_encoding not in {"learned", "rope"}:
            raise ValueError("position_encoding must be learned or rope")
        if self.mlp_type not in {"gelu", "swiglu"}:
            raise ValueError("mlp_type must be gelu or swiglu")
        if self.position_encoding == "rope":
            head_size = self.embedding_size // self.head_count
            if head_size % 2 != 0:
                raise ValueError("RoPE requires an even attention head size")


@dataclass
class TrainingConfig:
    """Configuration for model optimization and checkpointing.

    Attributes:
        output_dir: Folder where checkpoints and summaries are saved.
        epochs: Number of full passes over the training dataset.
        batch_size: Number of token windows per training batch.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay regularization.
        gradient_accumulation: Batches to accumulate before optimizer step.
        warmup_steps: Steps used to ramp up learning rate.
        eval_interval: Steps between validation loss checks.
        save_interval: Steps between checkpoint writes.
        max_grad_norm: Gradient clipping norm.
        use_amp: Enables mixed precision on CUDA.
        device: Training device, usually "cuda" or "cpu".
        seed: Random seed for repeatability.
        resume: Whether to resume from checkpoints.
        resume_from_checkpoint: Optional exact checkpoint path to resume from.
        require_compatible_resume: Validate tokenizer/model compatibility before resuming.
    """

    output_dir: Path
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    gradient_accumulation: int = 1
    warmup_steps: int = 100
    eval_interval: int = 100
    save_interval: int = 500
    max_grad_norm: float = 1.0
    use_amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337
    resume: bool = True
    resume_from_checkpoint: Optional[Path] = None
    require_compatible_resume: bool = True


def dataclass_to_jsonable(value: Any) -> dict[str, Any]:
    """Convert a dataclass into JSON-friendly values.

    Args:
        value: Dataclass instance to convert.

    Returns:
        Dictionary safe to pass to ``json.dumps``.
    """

    data = asdict(value)
    for key, item in list(data.items()):
        if isinstance(item, Path):
            data[key] = str(item)
    return data
