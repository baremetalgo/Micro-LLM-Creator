from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import DatasetConfig, ModelConfig, TrainingConfig, dataclass_to_jsonable
from .data import load_documents, write_training_corpus
from .tokenizer import PAD_TOKEN, encode_text, token_id, train_tokenizer
from .training import TrainingResult, split_tokens, train_model


@dataclass(slots=True)
class DatasetBuildResult:
    """Result returned after dataset preparation.

    Attributes:
        output_dir: Prepared dataset folder.
        tokenizer_path: Path to tokenizer JSON.
        document_count: Number of loaded samples.
        token_count: Total encoded tokens.
        vocab_size: Final tokenizer vocabulary size.
        character_count: Total corpus characters.
        suggested_vocab_size: Automatically estimated vocabulary size.
        warning: Optional dataset quality warning.
        code_sample_count: Number of code samples.
        prose_sample_count: Number of prose samples.
    """

    output_dir: Path
    tokenizer_path: Path
    document_count: int
    token_count: int
    vocab_size: int
    character_count: int
    suggested_vocab_size: int
    warning: str | None = None
    code_sample_count: int = 0
    prose_sample_count: int = 0


def _emit(progress: Callable[[Any], None] | None, message: str, percent: int | None = None) -> None:
    """Emit a progress event if a callback is available.

    Args:
        progress: Optional callback for progress dictionaries.
        message: Human-readable progress message.
        percent: Optional progress percentage.
    """

    if progress:
        progress({"message": message, "percent": percent})


def estimate_vocab_size(character_count: int, unique_word_count: int) -> int:
    """Estimate a reasonable tokenizer vocabulary size.

    Args:
        character_count: Number of corpus characters.
        unique_word_count: Approximate number of unique whitespace words.

    Returns:
        Suggested vocabulary size.
    """

    if character_count < 20_000:
        ceiling = 1_000
    elif character_count < 100_000:
        ceiling = 4_000
    elif character_count < 500_000:
        ceiling = 8_000
    elif character_count < 2_000_000:
        ceiling = 16_000
    else:
        ceiling = 32_000

    desired = max(512, int(unique_word_count * 1.7), int(character_count / 45))
    return max(256, min(ceiling, desired))


def content_warning(character_count: int) -> str | None:
    """Return a corpus-size warning when the dataset is small.

    Args:
        character_count: Number of corpus characters.

    Returns:
        Warning text, or ``None`` when the corpus is large enough.
    """

    if character_count < 10_000:
        return "The corpus is very small. Training can run, but the model will only be useful for smoke tests."
    if character_count < 100_000:
        return "The corpus is modest. Use more text for better generations and reasoning behavior."
    return None


def build_dataset(
    config: DatasetConfig,
    progress: Callable[[Any], None] | None = None,
) -> DatasetBuildResult:
    """Build a tokenizer-ready dataset project.

    Args:
        config: Dataset preparation settings.
        progress: Optional callback receiving progress event dictionaries.

    Returns:
        Dataset build summary.

    Raises:
        ValueError: If no supported documents are found.
    """

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _emit(progress, "Scanning source folder...", 3)
    documents = load_documents(
        config.input_dir,
        lowercase=config.lowercase,
        max_workers=config.max_workers,
        code_training_mode=config.code_training_mode,
        include_prose=config.include_prose,
        include_source_code=config.include_source_code,
        extract_code_blocks=config.extract_code_blocks,
        preserve_indentation=config.preserve_indentation,
        progress=progress,
    )
    if not documents:
        raise ValueError("No supported text, PDF, or JSONL documents were found.")

    all_text = "\n".join(doc.text for doc in documents)
    character_count = len(all_text)
    unique_words = len({word.lower() for word in all_text.split()})
    suggested_vocab_size = estimate_vocab_size(character_count, unique_words)
    selected_vocab_size = config.vocab_size or suggested_vocab_size
    warning = content_warning(character_count)
    code_sample_count = sum(1 for doc in documents if doc.kind == "code")
    prose_sample_count = sum(1 for doc in documents if doc.kind != "code")
    _emit(progress, f"Content size: {character_count:,} characters across {len(documents)} files.", 45)
    if config.code_training_mode:
        _emit(progress, f"Code mode: {code_sample_count:,} code samples, {prose_sample_count:,} prose samples.", 46)
    _emit(progress, f"Unique word estimate: {unique_words:,}.", 48)
    _emit(progress, f"Auto vocabulary size: {selected_vocab_size:,}.", 50)
    if warning:
        _emit(progress, f"Warning: {warning}")

    corpus_path = config.output_dir / "corpus.txt"
    _emit(progress, "Writing normalized corpus...", 56)
    write_training_corpus(
        documents,
        corpus_path,
        code_training_mode=config.code_training_mode,
        generate_instruction_samples=config.generate_instruction_samples,
    )
    tokenizer_path = config.output_dir / "tokenizer.json"
    _emit(progress, "Training tokenizer. This may take a while for large PDF folders...", 62)
    tokenizer = train_tokenizer(
        corpus_path,
        tokenizer_path,
        vocab_size=selected_vocab_size,
        min_frequency=config.min_frequency,
    )

    _emit(progress, "Encoding corpus into token IDs...", 78)
    tokens = encode_text(tokenizer, all_text)
    _emit(progress, f"Encoded {len(tokens):,} tokens.", 86)
    train_tokens, val_tokens = split_tokens(tokens, config.validation_split)
    _emit(progress, f"Training tokens: {len(train_tokens):,}; validation tokens: {len(val_tokens):,}.", 92)
    (config.output_dir / "train_tokens.json").write_text(json.dumps(train_tokens), encoding="utf-8")
    (config.output_dir / "val_tokens.json").write_text(json.dumps(val_tokens), encoding="utf-8")

    summary = {
        "dataset_config": dataclass_to_jsonable(config),
        "document_count": len(documents),
        "character_count": character_count,
        "token_count": len(tokens),
        "code_sample_count": code_sample_count,
        "prose_sample_count": prose_sample_count,
        "suggested_vocab_size": suggested_vocab_size,
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "warning": warning,
        "source_files": [str(doc.path) for doc in documents],
    }
    (config.output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _emit(progress, f"Dataset ready: {config.output_dir}", 100)
    return DatasetBuildResult(
        config.output_dir,
        tokenizer_path,
        len(documents),
        len(tokens),
        tokenizer.get_vocab_size(),
        character_count,
        suggested_vocab_size,
        warning,
        code_sample_count,
        prose_sample_count,
    )


def train_from_dataset(
    data_dir: Path,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    progress: Callable[[Any], None] | None = None,
) -> TrainingResult:
    """Train a model using a prepared dataset folder.

    Args:
        data_dir: Prepared dataset folder.
        model_config: Model architecture settings.
        training_config: Optimizer and checkpoint settings.
        progress: Optional callback receiving progress event dictionaries.

    Returns:
        Training result with final checkpoint and summary paths.

    Raises:
        FileNotFoundError: If the tokenizer is missing.
    """

    tokenizer_path = data_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    from .tokenizer import load_tokenizer

    tokenizer = load_tokenizer(tokenizer_path)
    train_tokens = json.loads((data_dir / "train_tokens.json").read_text(encoding="utf-8"))
    val_tokens = json.loads((data_dir / "val_tokens.json").read_text(encoding="utf-8"))

    if model_config.vocab_size != tokenizer.get_vocab_size():
        model_config.vocab_size = tokenizer.get_vocab_size()

    training_config.output_dir.mkdir(parents=True, exist_ok=True)
    (training_config.output_dir / "tokenizer.json").write_text(
        tokenizer_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return train_model(
        model_config,
        training_config,
        train_tokens,
        val_tokens,
        pad_token_id=token_id(tokenizer, PAD_TOKEN),
        progress=progress,
    )
