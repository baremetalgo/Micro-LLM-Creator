from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .config import DatasetConfig, ModelConfig, TrainingConfig, dataclass_to_jsonable
from .data import (
    document_from_dict,
    document_to_dict,
    expand_code_documents,
    file_sha256,
    read_supported_document,
    supported_source_paths,
    write_training_corpus,
)
from .tokenizer import PAD_TOKEN, encode_text, load_tokenizer, token_id, train_tokenizer
from .training import TrainingResult, split_tokens, train_model


@dataclass
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
        cached_file_count: Number of unchanged source files reused from cache.
        processed_file_count: Number of source files extracted this run.
    """

    output_dir: Path
    tokenizer_path: Path
    document_count: int
    token_count: int
    vocab_size: int
    character_count: int
    suggested_vocab_size: int
    warning: Optional[str] = None
    code_sample_count: int = 0
    prose_sample_count: int = 0
    cached_file_count: int = 0
    processed_file_count: int = 0


def _emit(progress: Optional[Callable[[Any], None]], message: str, percent: Optional[int] = None) -> None:
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


def content_warning(character_count: int) -> Optional[str]:
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


def _resolve_tokenizer_strategy(config: DatasetConfig, tokenizer_path: Path) -> tuple[str, bool]:
    """Resolve tokenizer strategy into an executable mode.

    Args:
        config: Dataset configuration.
        tokenizer_path: Dataset tokenizer output path.

    Returns:
        Strategy name and whether the dataset tokenizer should be reused.
    """

    strategy = config.tokenizer_strategy or "auto"
    if strategy == "auto":
        return strategy, config.prepare_mode == "incremental" and tokenizer_path.exists()
    if strategy == "reuse_dataset":
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Cannot reuse dataset tokenizer because tokenizer.json was not found in {config.output_dir}."
            )
        return strategy, True
    if strategy in {"train_new", "import_tokenizer"}:
        return strategy, False
    raise ValueError(f"Unsupported tokenizer strategy: {strategy}")


def _load_or_create_tokenizer(
    config: DatasetConfig,
    corpus_path: Path,
    tokenizer_path: Path,
    selected_vocab_size: int,
    progress: Optional[Callable[[Any], None]],
) -> tuple[Any, bool, bool, Optional[str]]:
    """Load, import, or train a tokenizer for the prepared corpus.

    Args:
        config: Dataset configuration.
        corpus_path: Normalized training corpus path.
        tokenizer_path: Dataset tokenizer output path.
        selected_vocab_size: Vocabulary size used when training a new tokenizer.
        progress: Optional progress callback.

    Returns:
        Tokenizer, reused flag, imported flag, and optional source path.
    """

    strategy, reuse_tokenizer = _resolve_tokenizer_strategy(config, tokenizer_path)
    imported = False
    source_path: Optional[str] = None

    if reuse_tokenizer:
        _emit(progress, "Reusing existing dataset tokenizer.json...", 62)
        return load_tokenizer(tokenizer_path), True, imported, source_path

    if strategy == "import_tokenizer":
        if config.tokenizer_path is None:
            raise ValueError("Choose a tokenizer.json file when tokenizer strategy is Import tokenizer.json.")
        import_path = Path(config.tokenizer_path)
        if not import_path.exists():
            raise FileNotFoundError(f"Tokenizer import file not found: {import_path}")
        _emit(progress, f"Importing tokenizer from {import_path}...", 62)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        if import_path.resolve() != tokenizer_path.resolve():
            shutil.copy2(import_path, tokenizer_path)
        return load_tokenizer(tokenizer_path), False, True, str(import_path)

    _emit(progress, "Training tokenizer. This may take a while for large PDF folders...", 62)
    tokenizer = train_tokenizer(
        corpus_path,
        tokenizer_path,
        vocab_size=selected_vocab_size,
        min_frequency=config.min_frequency,
    )
    return tokenizer, False, imported, source_path


def _cache_key(config: DatasetConfig) -> str:
    """Return a cache key for extraction-affecting options.

    Args:
        config: Dataset configuration.

    Returns:
        Cache key string.
    """

    return json.dumps(
        {
            "lowercase": config.lowercase,
            "code_training_mode": config.code_training_mode,
            "include_prose": config.include_prose,
            "include_source_code": config.include_source_code,
            "extract_code_blocks": config.extract_code_blocks,
            "preserve_indentation": config.preserve_indentation,
        },
        sort_keys=True,
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    """Read a dataset manifest.

    Args:
        path: Manifest path.

    Returns:
        Manifest dictionary.
    """

    if not path.exists():
        return {"version": 1, "files": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_documents_with_cache(
    config: DatasetConfig,
    progress: Optional[Callable[[Any], None]],
    should_stop: Optional[Callable[[], bool]],
) -> tuple[list[Any], dict[str, Any], int, int]:
    """Load documents using an extraction cache.

    Args:
        config: Dataset configuration.
        progress: Optional progress callback.
        should_stop: Optional cancellation callback.

    Returns:
        Documents, manifest, cached file count, processed file count.
    """

    manifest_path = config.output_dir / "dataset_manifest.json"
    cache_dir = config.output_dir / "cache" / "documents"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(manifest_path)
    manifest.setdefault("files", {})
    key = _cache_key(config)
    force_reprocess = config.prepare_mode == "force_reprocess"

    source_paths = supported_source_paths(
        config.input_dir,
        code_training_mode=config.code_training_mode,
        include_source_code=config.include_source_code,
    )
    _emit(progress, f"Found {len(source_paths)} supported files in {config.input_dir}.", 8)
    documents: list[Any] = []
    cached_count = 0
    processed_count = 0
    new_files: dict[str, Any] = {}

    for index, path in enumerate(source_paths, start=1):
        if should_stop and should_stop():
            raise RuntimeError("Dataset preparation stopped by user.")
        percent = 10 + int(32 * index / max(len(source_paths), 1))
        stat = path.stat()
        digest = file_sha256(path)
        cache_path = cache_dir / f"{digest}.json"
        manifest_key = str(path.resolve())
        previous = manifest.get("files", {}).get(manifest_key, {})
        can_use_cache = (
            not force_reprocess
            and previous.get("sha256") == digest
            and previous.get("cache_key") == key
            and cache_path.exists()
        )
        if can_use_cache:
            cached_documents = [
                document_from_dict(item)
                for item in json.loads(cache_path.read_text(encoding="utf-8"))
            ]
            documents.extend(cached_documents)
            cached_count += 1
            _emit(progress, f"Reused {path.name} from cache ({len(cached_documents)} sample(s)).", percent)
        else:
            source_doc = read_supported_document(
                path,
                lowercase=config.lowercase,
                code_training_mode=config.code_training_mode,
                preserve_indentation=config.preserve_indentation,
            )
            if source_doc is None:
                _emit(progress, f"Skipped {path.name}: no readable text found.", percent)
                continue
            source_documents = [source_doc]
            if config.code_training_mode:
                source_documents = expand_code_documents(
                    source_documents,
                    include_prose=config.include_prose,
                    extract_code_blocks=config.extract_code_blocks,
                    preserve_indentation=config.preserve_indentation,
                    should_stop=should_stop,
                )
            cache_path.write_text(
                json.dumps([document_to_dict(doc) for doc in source_documents], ensure_ascii=False),
                encoding="utf-8",
            )
            documents.extend(source_documents)
            processed_count += 1
            _emit(progress, f"Processed {path.name}: {len(source_documents)} sample(s).", percent)

        new_files[manifest_key] = {
            "path": str(path),
            "sha256": digest,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "cache_key": key,
            "cache_file": str(cache_path.relative_to(config.output_dir)),
        }

    manifest["files"] = new_files
    manifest["dataset_config"] = dataclass_to_jsonable(config)
    manifest["cache_key"] = key
    return sorted(documents, key=lambda document: (str(document.path), document.kind, document.language or "")), manifest, cached_count, processed_count


def build_dataset(
    config: DatasetConfig,
    progress: Optional[Callable[[Any], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> DatasetBuildResult:
    """Build a tokenizer-ready dataset project.

    Args:
        config: Dataset preparation settings.
        progress: Optional callback receiving progress event dictionaries.
        should_stop: Optional callback returning true when the user requested stop.

    Returns:
        Dataset build summary.

    Raises:
        ValueError: If no supported documents are found.
    """

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _emit(progress, "Scanning source folder...", 3)
    documents, manifest, cached_file_count, processed_file_count = _load_documents_with_cache(config, progress, should_stop)
    if should_stop and should_stop():
        raise RuntimeError("Dataset preparation stopped by user.")
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
    if cached_file_count or processed_file_count:
        _emit(progress, f"Cache: reused {cached_file_count:,} file(s), processed {processed_file_count:,} file(s).", 47)
    _emit(progress, f"Unique word estimate: {unique_words:,}.", 48)
    _emit(progress, f"Auto vocabulary size: {selected_vocab_size:,}.", 50)
    if warning:
        _emit(progress, f"Warning: {warning}")

    corpus_path = config.output_dir / "corpus.txt"
    _emit(progress, "Writing normalized corpus...", 56)
    if should_stop and should_stop():
        raise RuntimeError("Dataset preparation stopped by user.")
    write_training_corpus(
        documents,
        corpus_path,
        code_training_mode=config.code_training_mode,
        generate_instruction_samples=config.generate_instruction_samples,
    )
    tokenizer_path = config.output_dir / "tokenizer.json"
    if should_stop and should_stop():
        raise RuntimeError("Dataset preparation stopped by user.")
    tokenizer, reuse_tokenizer, tokenizer_imported, tokenizer_source_path = _load_or_create_tokenizer(
        config,
        corpus_path,
        tokenizer_path,
        selected_vocab_size,
        progress,
    )

    _emit(progress, "Encoding corpus into token IDs...", 78)
    if should_stop and should_stop():
        raise RuntimeError("Dataset preparation stopped by user.")
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
        "cached_file_count": cached_file_count,
        "processed_file_count": processed_file_count,
        "prepare_mode": config.prepare_mode,
        "tokenizer_strategy": config.tokenizer_strategy,
        "tokenizer_reused": reuse_tokenizer,
        "tokenizer_imported": tokenizer_imported,
        "tokenizer_source_path": tokenizer_source_path,
    }
    (config.output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (config.output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
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
        cached_file_count,
        processed_file_count,
    )


def train_from_dataset(
    data_dir: Path,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    progress: Optional[Callable[[Any], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> TrainingResult:
    """Train a model using a prepared dataset folder.

    Args:
        data_dir: Prepared dataset folder.
        model_config: Model architecture settings.
        training_config: Optimizer and checkpoint settings.
        progress: Optional callback receiving progress event dictionaries.
        should_stop: Optional callback returning true when the user requested stop.

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
        should_stop=should_stop,
    )
