from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import PyPDF2

from .config import DatasetConfig, ModelConfig, TrainingConfig, dataclass_to_jsonable
from .conversation_datasets import CONVERSATION_DATASET_PRESETS, dataset_ids_for_stage, load_conversation_documents
from .data import (
    Document,
    SUPPORTED_CODE_SUFFIXES,
    SUPPORTED_TEXT_SUFFIXES,
    document_from_dict,
    document_to_dict,
    expand_code_documents,
    file_sha256,
    read_supported_document,
    write_training_corpus,
    supported_source_paths,
)
from .lineage import read_json, record_dataset_version, stable_json_hash, utc_timestamp, write_json
from .tokenizer import PAD_TOKEN, encode_text, load_tokenizer, token_id, train_tokenizer, validate_training_tokenizer
from .training import TrainingResult, latest_checkpoint, split_tokens, train_model


LOGGER = logging.getLogger(__name__)


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
        conversation_sample_count: Number of conversation/instruction samples.
        cached_file_count: Number of unchanged source files reused from cache.
        processed_file_count: Number of source files extracted this run.
        skipped_file_count: Number of files with no readable text.
        failed_file_count: Number of files that failed extraction.
        dataset_version_id: Unique dataset version identifier.
        dataset_version_number: One-based dataset version number.
        mixture_report: Per-source family sampling report.
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
    conversation_sample_count: int = 0
    cached_file_count: int = 0
    processed_file_count: int = 0
    skipped_file_count: int = 0
    failed_file_count: int = 0
    dataset_version_id: str = ""
    dataset_version_number: int = 0
    mixture_report: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectHealthResult:
    """Result returned by the project health checker.

    Attributes:
        status: Overall status: ok, warning, or error.
        checks: Individual check dictionaries.
        summary: Human-readable summary.
    """

    status: str
    checks: list[dict[str, str]]
    summary: str


@dataclass
class DatasetPreviewResult:
    """Result returned by the dataset preview scanner.

    Attributes:
        source_file_count: Number of supported source files.
        prepared: Whether prepared dataset artifacts exist.
        total_bytes: Total bytes across supported source files.
        suffix_counts: Supported file counts by suffix.
        sample_previews: Small text/code samples for inspection.
        issues: Quality issues or recommendations.
        summary: Existing dataset summary when available.
        duplicate_count: Number of files involved in likely duplicate groups.
        duplicate_groups: Duplicate group summaries.
        bad_extraction_count: Number of files with suspicious extraction quality.
        bad_extraction_files: Suspicious extraction summaries.
        code_preview_count: Code-like samples seen during preview.
        prose_preview_count: Prose samples seen during preview.
        balance_label: Human-readable code/prose balance.
        readiness_score: Training readiness score from 0 to 100.
        readiness_label: Human-readable readiness label.
        readiness_reasons: Reasons behind the readiness score.
    """

    source_file_count: int
    prepared: bool
    total_bytes: int
    suffix_counts: dict[str, int]
    sample_previews: list[dict[str, str]]
    issues: list[str]
    summary: dict[str, Any]
    duplicate_count: int = 0
    duplicate_groups: list[dict[str, Any]] = field(default_factory=list)
    bad_extraction_count: int = 0
    bad_extraction_files: list[dict[str, str]] = field(default_factory=list)
    code_preview_count: int = 0
    prose_preview_count: int = 0
    balance_label: str = "Unknown"
    readiness_score: int = 0
    readiness_label: str = "Unknown"
    readiness_reasons: list[str] = field(default_factory=list)


def _emit(progress: Optional[Callable[[Any], None]], message: str, percent: Optional[int] = None) -> None:
    """Emit a progress event if a callback is available.

    Args:
        progress: Optional callback for progress dictionaries.
        message: Human-readable progress message.
        percent: Optional progress percentage.
    """

    LOGGER.info(message)
    if progress:
        progress({"message": message, "percent": percent})


def _health_check(name: str, status: str, detail: str) -> dict[str, str]:
    """Create a project health check row.

    Args:
        name: Check label.
        status: ok, warning, or error.
        detail: Human-readable detail.

    Returns:
        Check dictionary.
    """

    return {"name": name, "status": status, "detail": detail}


def check_project_health(
    input_dir: Path,
    dataset_dir: Path,
    model_dir: Path,
    export_dir: Path,
    gguf_path: Optional[Path],
    llama_cpp_dir: Optional[Path],
    training_device: str,
    progress: Optional[Callable[[Any], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> ProjectHealthResult:
    """Validate paths, artifacts, and hardware readiness for the current project.

    Args:
        input_dir: Source data folder.
        dataset_dir: Prepared dataset folder.
        model_dir: Training output folder.
        export_dir: Export output folder.
        gguf_path: Optional GGUF model selected for chat.
        llama_cpp_dir: Optional llama.cpp checkout folder.
        training_device: Selected training device.
        progress: Optional callback receiving progress event dictionaries.
        should_stop: Optional callback returning true when the user requested stop.

    Returns:
        Project health result.
    """

    checks: list[dict[str, str]] = []
    _emit(progress, "Checking project paths...", 10)
    if should_stop and should_stop():
        raise RuntimeError("Project health check stopped by user.")

    if input_dir.exists() and input_dir.is_dir():
        try:
            source_count = len(supported_source_paths(input_dir, code_training_mode=True, include_source_code=True))
            if source_count:
                checks.append(_health_check("Source vault", "ok", f"{source_count:,} supported file(s) found."))
            else:
                checks.append(_health_check("Source vault", "warning", "Folder exists, but no supported files were found."))
        except Exception as exc:
            checks.append(_health_check("Source vault", "error", str(exc)))
    else:
        checks.append(_health_check("Source vault", "warning", f"Folder not found: {input_dir}"))

    _emit(progress, "Checking prepared dataset artifacts...", 30)
    required_dataset = ("tokenizer.json", "train_tokens.json", "val_tokens.json", "dataset_summary.json")
    missing_dataset = [name for name in required_dataset if not (dataset_dir / name).exists()]
    if not dataset_dir.exists():
        checks.append(_health_check("Dataset core", "warning", f"Dataset folder not found yet: {dataset_dir}"))
    elif missing_dataset:
        checks.append(_health_check("Dataset core", "warning", f"Missing prepared artifact(s): {', '.join(missing_dataset)}"))
    else:
        summary = read_json(dataset_dir / "dataset_summary.json", default={}) or {}
        tokens = int(summary.get("token_count", 0) or 0)
        vocab = int(summary.get("tokenizer_vocab_size", 0) or 0)
        checks.append(_health_check("Dataset core", "ok", f"Prepared with {tokens:,} token(s), vocab {vocab:,}."))

    _emit(progress, "Checking model artifacts...", 50)
    if not model_dir.exists():
        checks.append(_health_check("Model output", "warning", f"Model folder not found yet: {model_dir}"))
    elif (model_dir / "final_model.pt").exists():
        checks.append(_health_check("Model output", "ok", "final_model.pt found."))
    elif (model_dir / "checkpoints").exists() and any((model_dir / "checkpoints").glob("*.pt")):
        checks.append(_health_check("Model output", "warning", "Checkpoints found, but final_model.pt is not present yet."))
    else:
        checks.append(_health_check("Model output", "warning", "No trained model or checkpoint found yet."))

    _emit(progress, "Checking export and GGUF paths...", 65)
    if export_dir.exists():
        artifact_count = sum(1 for item in export_dir.iterdir())
        checks.append(_health_check("Export bay", "ok" if artifact_count else "warning", f"{artifact_count:,} item(s) in export folder."))
    else:
        checks.append(_health_check("Export bay", "warning", f"Export folder not found yet: {export_dir}"))

    if gguf_path and str(gguf_path).strip():
        if gguf_path.exists():
            checks.append(_health_check("GGUF chat model", "ok", f"Found {gguf_path.name}."))
        else:
            checks.append(_health_check("GGUF chat model", "warning", f"GGUF file not found: {gguf_path}"))
    else:
        checks.append(_health_check("GGUF chat model", "warning", "No GGUF model selected for chat."))

    if llama_cpp_dir and str(llama_cpp_dir).strip():
        converter = llama_cpp_dir / "convert_hf_to_gguf.py"
        checks.append(
            _health_check(
                "llama.cpp",
                "ok" if converter.exists() else "warning",
                "convert_hf_to_gguf.py found." if converter.exists() else f"Converter not found in {llama_cpp_dir}.",
            )
        )

    _emit(progress, "Checking hardware/runtime...", 85)
    if training_device == "cuda":
        if torch.cuda.is_available():
            checks.append(_health_check("Training device", "ok", f"CUDA ready: {torch.cuda.get_device_name(0)}."))
        else:
            checks.append(_health_check("Training device", "error", "CUDA selected but PyTorch cannot use CUDA."))
    else:
        checks.append(_health_check("Training device", "ok", "CPU selected. Training will be slower but compatible."))

    statuses = [check["status"] for check in checks]
    if "error" in statuses:
        status = "error"
    elif "warning" in statuses:
        status = "warning"
    else:
        status = "ok"
    summary = f"{sum(1 for item in statuses if item == 'ok')} ok, {sum(1 for item in statuses if item == 'warning')} warning, {sum(1 for item in statuses if item == 'error')} error."
    _emit(progress, f"Project health check complete: {summary}", 100)
    return ProjectHealthResult(status=status, checks=checks, summary=summary)


def _supported_source_paths_cancellable(
    input_dir: Path,
    code_training_mode: bool,
    include_source_code: bool,
    should_stop: Optional[Callable[[], bool]],
) -> list[Path]:
    """Return supported source paths while honoring cancellation.

    Args:
        input_dir: Folder to scan.
        code_training_mode: Whether source-code files are supported.
        include_source_code: Whether to include source-code files.
        should_stop: Optional cancellation callback.

    Returns:
        Sorted supported source paths.
    """

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    allowed = set(SUPPORTED_TEXT_SUFFIXES) | {".pdf", ".jsonl"}
    if code_training_mode and include_source_code:
        allowed |= set(SUPPORTED_CODE_SUFFIXES)
    paths: list[Path] = []
    for root, dirs, files in os.walk(input_dir):
        if should_stop and should_stop():
            raise RuntimeError("Dataset preview stopped by user.")
        dirs[:] = [
            name
            for name in dirs
            if name not in {".git", "__pycache__", ".venv", "venv", "node_modules"}
        ]
        for filename in files:
            if should_stop and should_stop():
                raise RuntimeError("Dataset preview stopped by user.")
            path = Path(root) / filename
            if path.suffix.lower() in allowed:
                paths.append(path)
    return sorted(paths)


def _preview_supported_document(
    path: Path,
    config: DatasetConfig,
    should_stop: Optional[Callable[[], bool]],
    max_chars: int = 1200,
) -> Optional[dict[str, str]]:
    """Read a small preview from one source document.

    Args:
        path: Source path.
        config: Dataset options.
        should_stop: Optional cancellation callback.
        max_chars: Maximum preview characters.

    Returns:
        Preview dictionary, or ``None`` when the file has no readable text.
    """

    if should_stop and should_stop():
        raise RuntimeError("Dataset preview stopped by user.")
    suffix = path.suffix.lower()
    kind = "prose"
    language = ""
    text = ""
    if config.code_training_mode and suffix in SUPPORTED_CODE_SUFFIXES:
        kind = "code"
        language = SUPPORTED_CODE_SUFFIXES[suffix]
        with path.open("rb") as file:
            text = file.read(128 * 1024).decode("utf-8", errors="ignore")
        text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()
    elif suffix in SUPPORTED_TEXT_SUFFIXES:
        with path.open("rb") as file:
            text = file.read(128 * 1024).decode("utf-8", errors="ignore")
        text = re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()
    elif suffix == ".jsonl":
        chunks: list[str] = []
        with path.open("r", encoding="utf-8", errors="ignore") as file:
            for index, line in enumerate(file):
                if should_stop and should_stop():
                    raise RuntimeError("Dataset preview stopped by user.")
                if index >= 40 or sum(len(chunk) for chunk in chunks) >= max_chars:
                    break
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, str):
                    chunks.append(value)
                elif isinstance(value, dict):
                    for key in ("text", "content", "prompt", "completion"):
                        if value.get(key):
                            chunks.append(str(value[key]))
                            break
        text = "\n".join(chunks).strip()
    elif suffix == ".pdf":
        chunks: list[str] = []
        with path.open("rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages[:3]:
                if should_stop and should_stop():
                    raise RuntimeError("Dataset preview stopped by user.")
                chunks.append(page.extract_text() or "")
                if sum(len(chunk) for chunk in chunks) >= max_chars:
                    break
        text = re.sub(r"\s+", " ", "\n".join(chunks).replace("\x00", " ")).strip()
    if config.lowercase:
        text = text.lower()
    if not text:
        return None
    return {
        "path": str(path),
        "kind": kind,
        "language": language,
        "characters": str(len(text)),
        "preview": text[:max_chars],
    }


def _file_sha256_cancellable(path: Path, should_stop: Optional[Callable[[], bool]]) -> str:
    """Calculate a file digest while honoring cancellation between chunks.

    Args:
        path: File path.
        should_stop: Optional cancellation callback.

    Returns:
        File SHA-256 hex digest.
    """

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            if should_stop and should_stop():
                raise RuntimeError("Dataset preview stopped by user.")
            digest.update(chunk)
    return digest.hexdigest()


def _preview_fingerprint(text: str) -> str:
    """Return a normalized content fingerprint for duplicate detection.

    Args:
        text: Preview text.

    Returns:
        SHA-1 fingerprint.
    """

    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    normalized = re.sub(r"\d+", "0", normalized)
    return hashlib.sha1(normalized[:4000].encode("utf-8", errors="ignore")).hexdigest()


def _bad_extraction_reasons(path: Path, preview: Optional[dict[str, str]], size: int) -> list[str]:
    """Return quality reasons when extracted preview text looks suspicious.

    Args:
        path: Source path.
        preview: Preview dictionary, or ``None``.
        size: Source file size in bytes.

    Returns:
        List of quality concerns.
    """

    suffix = path.suffix.lower()
    if preview is None:
        return ["no readable preview text"]
    text = str(preview.get("preview", ""))
    if not text.strip():
        return ["empty preview text"]
    reasons: list[str] = []
    visible = [char for char in text if not char.isspace()]
    if len(text) < 80 and suffix == ".pdf":
        reasons.append("very little text extracted from PDF preview")
    if size > 250_000 and suffix == ".pdf" and len(text) < 200:
        reasons.append("large PDF produced very little readable text")
    if visible:
        alpha_ratio = sum(char.isalpha() for char in visible) / len(visible)
        symbol_ratio = sum(not char.isalnum() for char in visible) / len(visible)
        if alpha_ratio < 0.25 and str(preview.get("kind")) != "code":
            reasons.append("low alphabetic text ratio")
        if symbol_ratio > 0.45:
            reasons.append("high symbol/noise ratio")
    if re.search(r"(.)\1{18,}", text):
        reasons.append("long repeated character run")
    if text.count("\ufffd") >= 3 or "Ã" in text[:500]:
        reasons.append("encoding artifacts detected")
    words = re.findall(r"[A-Za-z]{2,}", text)
    if suffix in {".pdf", ".txt", ".md", ".text"} and len(set(words)) < 8 and len(text) > 200:
        reasons.append("very low word variety")
    return reasons


def _balance_label(code_count: int, prose_count: int, code_training_mode: bool) -> str:
    """Return a readable code/prose balance label.

    Args:
        code_count: Number of code samples.
        prose_count: Number of prose samples.
        code_training_mode: Whether code-oriented preparation is enabled.

    Returns:
        Balance label.
    """

    total = code_count + prose_count
    if total <= 0:
        return "Unknown"
    code_ratio = code_count / total
    if not code_training_mode:
        return "Prose focused"
    if code_ratio < 0.2:
        return "Prose heavy"
    if code_ratio > 0.8:
        return "Code heavy"
    return "Balanced code/prose"


def _readiness_report(
    source_file_count: int,
    total_bytes: int,
    prepared: bool,
    summary: dict[str, Any],
    duplicate_count: int,
    bad_extraction_count: int,
    code_count: int,
    prose_count: int,
    code_training_mode: bool,
) -> tuple[int, str, list[str]]:
    """Calculate an explainable training readiness score.

    Args:
        source_file_count: Number of supported source files.
        total_bytes: Total supported source bytes.
        prepared: Whether token artifacts exist.
        summary: Prepared dataset summary when available.
        duplicate_count: Files involved in likely duplicates.
        bad_extraction_count: Files with suspicious extraction.
        code_count: Code sample count.
        prose_count: Prose sample count.
        code_training_mode: Whether user is preparing a code-oriented dataset.

    Returns:
        Score, label, and explanatory reasons.
    """

    score = 100
    reasons: list[str] = []
    token_count = int(summary.get("token_count", 0) or 0)
    if prepared and token_count:
        if token_count < 50_000:
            score -= 35
            reasons.append("Prepared token count is very small.")
        elif token_count < 250_000:
            score -= 18
            reasons.append("Prepared token count is modest.")
        else:
            reasons.append("Prepared token count looks usable.")
    else:
        score -= 15
        reasons.append("Dataset is not prepared yet; score is based on source preview.")
        if total_bytes < 250_000:
            score -= 25
            reasons.append("Source size is small for meaningful training.")
        elif total_bytes < 2_000_000:
            score -= 10
            reasons.append("Source size is modest.")

    if source_file_count == 0:
        score -= 60
        reasons.append("No supported source files were found.")
    duplicate_ratio = duplicate_count / max(source_file_count, 1)
    if duplicate_ratio >= 0.25:
        score -= 25
        reasons.append("Duplicate ratio is high.")
    elif duplicate_ratio >= 0.1:
        score -= 12
        reasons.append("Some likely duplicate files were found.")

    bad_ratio = bad_extraction_count / max(source_file_count, 1)
    if bad_ratio >= 0.2:
        score -= 25
        reasons.append("Many files have suspicious extraction quality.")
    elif bad_ratio > 0:
        score -= 10
        reasons.append("Some files have suspicious extraction quality.")

    total_samples = code_count + prose_count
    if code_training_mode and total_samples:
        code_ratio = code_count / total_samples
        if code_ratio < 0.1:
            score -= 12
            reasons.append("Code training mode is enabled but very little code was detected.")
        elif code_ratio > 0.95 and prose_count == 0:
            score -= 5
            reasons.append("Dataset is almost entirely code; explanations may be weak.")

    score = max(0, min(100, score))
    if score >= 80:
        label = "Ready"
    elif score >= 60:
        label = "Usable with warnings"
    elif score >= 35:
        label = "Needs cleanup"
    else:
        label = "Not ready"
    return score, label, reasons


def scan_dataset_preview(
    config: DatasetConfig,
    sample_limit: int = 8,
    progress: Optional[Callable[[Any], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> DatasetPreviewResult:
    """Scan source/prepared dataset quality without writing training artifacts.

    Args:
        config: Dataset preparation settings.
        sample_limit: Maximum readable source samples to preview.
        progress: Optional callback receiving progress event dictionaries.
        should_stop: Optional callback returning true when the user requested stop.

    Returns:
        Dataset preview result.
    """

    _emit(progress, "Scanning supported source files...", 5)
    if config.input_dir.exists():
        paths = _supported_source_paths_cancellable(
            config.input_dir,
            config.code_training_mode,
            config.include_source_code,
            should_stop,
        )
    elif config.conversation_datasets:
        paths = []
    else:
        paths = _supported_source_paths_cancellable(
            config.input_dir,
            config.code_training_mode,
            config.include_source_code,
            should_stop,
        )
    suffix_counts: dict[str, int] = {}
    total_bytes = 0
    for path in paths:
        if should_stop and should_stop():
            raise RuntimeError("Dataset preview stopped by user.")
        suffix_counts[path.suffix.lower() or "<none>"] = suffix_counts.get(path.suffix.lower() or "<none>", 0) + 1
        try:
            total_bytes += path.stat().st_size
        except OSError:
            pass

    summary = read_json(config.output_dir / "dataset_summary.json", default={}) or {}
    prepared = all((config.output_dir / name).exists() for name in ("tokenizer.json", "train_tokens.json", "val_tokens.json"))
    issues: list[str] = []
    if not paths and not config.conversation_datasets:
        issues.append("No supported source files found.")
    if config.conversation_datasets:
        labels = [
            CONVERSATION_DATASET_PRESETS[item].label
            for item in config.conversation_datasets
            if item in CONVERSATION_DATASET_PRESETS
        ]
        issues.append(f"Conversation datasets selected: {', '.join(labels)}.")
    if total_bytes < 100_000:
        issues.append("Source content appears small for meaningful LLM training.")
    if prepared and summary:
        token_count = int(summary.get("token_count", 0) or 0)
        if token_count < 50_000:
            issues.append("Prepared token count is low; expect smoke-test quality only.")
        if summary.get("warning"):
            issues.append(str(summary["warning"]))
    elif config.output_dir.exists():
        issues.append("Dataset folder exists but does not contain a complete prepared dataset.")

    _emit(progress, "Reading a few preview samples...", 30)
    sample_previews: list[dict[str, str]] = []
    all_previews: list[dict[str, str]] = []
    bad_extraction_files: list[dict[str, str]] = []
    content_fingerprints: dict[str, list[str]] = {}
    readable = 0
    scan_limit = min(len(paths), max(sample_limit * 8, 80))
    for index, path in enumerate(paths[:scan_limit], start=1):
        if should_stop and should_stop():
            raise RuntimeError("Dataset preview stopped by user.")
        try:
            preview = _preview_supported_document(path, config, should_stop)
        except RuntimeError:
            raise
        except Exception as exc:
            issues.append(f"Could not preview {path.name}: {exc}")
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        reasons = _bad_extraction_reasons(path, preview, size)
        if reasons:
            bad_extraction_files.append(
                {
                    "path": str(path),
                    "reasons": "; ".join(reasons),
                    "size": str(size),
                }
            )
        if preview is None:
            issues.append(f"{path.name} has no readable text.")
            continue
        readable += 1
        all_previews.append(preview)
        preview_text = preview.get("preview", "")
        if len(preview_text) >= 120:
            content_fingerprints.setdefault(_preview_fingerprint(preview_text), []).append(str(path))
        if len(sample_previews) < sample_limit:
            sample_previews.append(preview)
            percent = 30 + int(45 * len(sample_previews) / max(sample_limit, 1))
            _emit(progress, f"Previewed {path.name}.", percent)

    if readable == 0 and paths:
        issues.append("Supported files were found, but none produced readable preview text.")
    _emit(progress, "Checking duplicate files and repeated extracted text...", 85)
    size_groups: dict[int, list[Path]] = {}
    for path in paths:
        if should_stop and should_stop():
            raise RuntimeError("Dataset preview stopped by user.")
        try:
            size_groups.setdefault(path.stat().st_size, []).append(path)
        except OSError:
            continue
    exact_hashes: dict[str, list[str]] = {}
    for same_size_paths in size_groups.values():
        if len(same_size_paths) < 2:
            continue
        for path in same_size_paths:
            digest = _file_sha256_cancellable(path, should_stop)
            exact_hashes.setdefault(digest, []).append(str(path))
    duplicate_groups: list[dict[str, Any]] = []
    for group in exact_hashes.values():
        if len(group) > 1:
            duplicate_groups.append({"type": "exact file", "count": len(group), "files": group[:8]})
    for group in content_fingerprints.values():
        unique_group = sorted(set(group))
        if len(unique_group) > 1:
            duplicate_groups.append({"type": "similar extracted text", "count": len(unique_group), "files": unique_group[:8]})
    duplicate_files = {
        file_path
        for group in duplicate_groups
        for file_path in group.get("files", [])
    }
    duplicate_count = len(duplicate_files)
    if duplicate_groups:
        issues.append(f"Found {len(duplicate_groups)} likely duplicate group(s) involving {duplicate_count} file entries.")
    if bad_extraction_files:
        issues.append(f"Found {len(bad_extraction_files)} file(s) with suspicious extraction quality.")
    summary_code_count = int(summary.get("code_sample_count", 0) or 0)
    summary_prose_count = int(summary.get("prose_sample_count", 0) or 0)
    code_preview_count = summary_code_count or sum(1 for preview in all_previews if preview.get("kind") == "code")
    prose_preview_count = summary_prose_count or sum(1 for preview in all_previews if preview.get("kind") != "code")
    balance = _balance_label(code_preview_count, prose_preview_count, config.code_training_mode)
    effective_source_count = len(paths) + len(config.conversation_datasets)
    readiness_score, readiness_label, readiness_reasons = _readiness_report(
        source_file_count=effective_source_count,
        total_bytes=total_bytes,
        prepared=prepared,
        summary=summary,
        duplicate_count=duplicate_count,
        bad_extraction_count=len(bad_extraction_files),
        code_count=code_preview_count,
        prose_count=prose_preview_count,
        code_training_mode=config.code_training_mode,
    )
    _emit(progress, "Dataset preview complete.", 100)
    return DatasetPreviewResult(
        source_file_count=effective_source_count,
        prepared=prepared,
        total_bytes=total_bytes,
        suffix_counts=dict(sorted(suffix_counts.items())),
        sample_previews=sample_previews,
        issues=issues,
        summary=summary,
        duplicate_count=duplicate_count,
        duplicate_groups=duplicate_groups,
        bad_extraction_count=len(bad_extraction_files),
        bad_extraction_files=bad_extraction_files[:30],
        code_preview_count=code_preview_count,
        prose_preview_count=prose_preview_count,
        balance_label=balance,
        readiness_score=readiness_score,
        readiness_label=readiness_label,
        readiness_reasons=readiness_reasons,
    )


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
    should_stop: Optional[Callable[[], bool]],
) -> tuple[Any, bool, bool, Optional[str]]:
    """Load, import, or train a tokenizer for the prepared corpus.

    Args:
        config: Dataset configuration.
        corpus_path: Normalized training corpus path.
        tokenizer_path: Dataset tokenizer output path.
        selected_vocab_size: Vocabulary size used when training a new tokenizer.
        progress: Optional progress callback.
        should_stop: Optional cancellation callback.

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
        should_stop=should_stop,
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
            "generate_instruction_samples": config.generate_instruction_samples,
            "reasoning_sample_mode": config.reasoning_sample_mode,
            "dataset_stage": config.dataset_stage,
            "conversation_datasets": config.conversation_datasets,
            "conversation_sample_limit": config.conversation_sample_limit,
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
) -> tuple[list[Any], dict[str, Any], int, int, int, int]:
    """Load documents using an extraction cache.

    Args:
        config: Dataset configuration.
        progress: Optional progress callback.
        should_stop: Optional cancellation callback.

    Returns:
        Documents, manifest, cached, processed, skipped, and failed file counts.
    """

    manifest_path = config.output_dir / "dataset_manifest.json"
    cache_dir = config.output_dir / "cache" / "documents"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(manifest_path)
    manifest.setdefault("files", {})
    key = _cache_key(config)
    force_reprocess = config.prepare_mode == "force_reprocess"

    if config.input_dir.exists():
        source_paths = supported_source_paths(
            config.input_dir,
            code_training_mode=config.code_training_mode,
            include_source_code=config.include_source_code,
        )
    elif config.conversation_datasets:
        source_paths = []
    else:
        source_paths = supported_source_paths(
            config.input_dir,
            code_training_mode=config.code_training_mode,
            include_source_code=config.include_source_code,
        )
    _emit(progress, f"Found {len(source_paths)} supported files in {config.input_dir}.", 8)
    documents: list[Any] = []
    cached_count = 0
    processed_count = 0
    skipped_count = 0
    failed_count = 0
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
            try:
                source_doc = read_supported_document(
                    path,
                    lowercase=config.lowercase,
                    code_training_mode=config.code_training_mode,
                    preserve_indentation=config.preserve_indentation,
                )
            except Exception as exc:
                failed_count += 1
                _emit(progress, f"Failed {path.name}: {exc}", percent)
                new_files[manifest_key] = {
                    "path": str(path),
                    "sha256": digest,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "cache_key": key,
                    "status": "failed",
                    "error": str(exc),
                }
                continue
            if source_doc is None:
                skipped_count += 1
                _emit(progress, f"Skipped {path.name}: no readable text found.", percent)
                new_files[manifest_key] = {
                    "path": str(path),
                    "sha256": digest,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "cache_key": key,
                    "status": "skipped_empty",
                }
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
            "status": "cached" if can_use_cache else "processed",
        }

    if config.conversation_datasets:
        allowed_dataset_ids = set(dataset_ids_for_stage(config.dataset_stage))
        skipped_stage_ids = [dataset_id for dataset_id in config.conversation_datasets if dataset_id not in allowed_dataset_ids]
        selected_dataset_ids = [dataset_id for dataset_id in config.conversation_datasets if dataset_id in allowed_dataset_ids]
        if skipped_stage_ids:
            skipped_labels = [
                CONVERSATION_DATASET_PRESETS[item].label
                for item in skipped_stage_ids
                if item in CONVERSATION_DATASET_PRESETS
            ]
            _emit(progress, f"Skipping dataset(s) not recommended for {config.dataset_stage}: {', '.join(skipped_labels)}.")
        if not selected_dataset_ids:
            _emit(progress, f"No online datasets selected for {config.dataset_stage}; continuing with local sources only.")
            config.conversation_datasets = []
            manifest["files"] = new_files
            manifest["dataset_config"] = dataclass_to_jsonable(config)
            manifest["cache_key"] = key
            return (
                sorted(documents, key=lambda document: (str(document.path), document.kind, document.language or "")),
                manifest,
                cached_count,
                processed_count,
                skipped_count,
                failed_count,
            )
        hf_cache_dir = config.output_dir / "cache" / "huggingface"
        labels = [
            CONVERSATION_DATASET_PRESETS[item].label
            for item in selected_dataset_ids
            if item in CONVERSATION_DATASET_PRESETS
        ]
        _emit(progress, f"Online training datasets enabled: {', '.join(labels)}.", 8)
        _emit(progress, f"Online training datasets will be cached in: {hf_cache_dir}", 8)
        hf_documents = load_conversation_documents(
            selected_dataset_ids,
            config.conversation_sample_limit,
            hf_cache_dir,
            lowercase=config.lowercase,
            progress=progress,
            should_stop=should_stop,
        )
        documents.extend(hf_documents)
        config.conversation_datasets = selected_dataset_ids
        for dataset_id in selected_dataset_ids:
            preset = CONVERSATION_DATASET_PRESETS.get(dataset_id)
            new_files[f"hf://{dataset_id}"] = {
                "path": f"hf://{dataset_id}",
                "dataset": preset.hf_path if preset else dataset_id,
                "config_name": preset.config_name if preset else "",
                "split": preset.split if preset else "",
                "sample_limit": config.conversation_sample_limit,
                "cache_key": key,
                "status": "processed",
            }
        processed_count += len(selected_dataset_ids)

    manifest["files"] = new_files
    manifest["dataset_config"] = dataclass_to_jsonable(config)
    manifest["cache_key"] = key
    return (
        sorted(documents, key=lambda document: (str(document.path), document.kind, document.language or "")),
        manifest,
        cached_count,
        processed_count,
        skipped_count,
        failed_count,
    )


MIXTURE_LABELS = {
    "local_prose": "Local prose",
    "source_code": "Source code",
    "online_base": "Online base",
    "instruction": "Instruction",
    "conversation": "Conversation",
}


def _document_mixture_family(document: Document) -> str:
    """Classify a document into a dataset mixture source family.

    Args:
        document: Loaded source document.

    Returns:
        Mixture source family identifier.
    """

    if document.kind == "code":
        return "source_code"
    if document.kind == "instruction":
        return "instruction"
    if document.kind == "conversation":
        return "conversation"
    dataset_id = str(document.language or "")
    preset = CONVERSATION_DATASET_PRESETS.get(dataset_id)
    if preset and preset.stage == "base":
        return "online_base"
    if "__hf_datasets__" in document.path.parts:
        for part in document.path.parts:
            preset = CONVERSATION_DATASET_PRESETS.get(part)
            if preset and preset.stage == "base":
                return "online_base"
    return "local_prose"


def _stable_document_sort_key(document: Document) -> str:
    """Return a stable pseudo-random sort key for sampling.

    Args:
        document: Loaded source document.

    Returns:
        Hex digest used to order documents deterministically.
    """

    key = f"{document.path}|{document.kind}|{document.language or ''}|{len(document.text)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _empty_mixture_report(weights: dict[str, float], documents: list[Document], applied: bool, reason: str = "") -> dict[str, Any]:
    """Build a mixture report without changing document selection.

    Args:
        weights: Requested mixture weights.
        documents: Available documents.
        applied: Whether weighted sampling was applied.
        reason: Optional reason when sampling was not applied.

    Returns:
        Mixture report dictionary.
    """

    by_family: dict[str, list[Document]] = {key: [] for key in MIXTURE_LABELS}
    for document in documents:
        by_family.setdefault(_document_mixture_family(document), []).append(document)
    total_chars = sum(len(document.text) for document in documents)
    families = {}
    for family, label in MIXTURE_LABELS.items():
        available = by_family.get(family, [])
        available_chars = sum(len(document.text) for document in available)
        families[family] = {
            "label": label,
            "requested_weight": float(weights.get(family, 0.0) or 0.0),
            "available_documents": len(available),
            "available_characters": available_chars,
            "selected_documents": len(available) if not applied else 0,
            "selected_characters": available_chars if not applied else 0,
            "actual_percent": (available_chars * 100.0 / total_chars) if total_chars else 0.0,
            "dropped_documents": 0,
            "dropped_characters": 0,
        }
    return {
        "applied": applied,
        "reason": reason,
        "total_available_documents": len(documents),
        "total_selected_documents": len(documents) if not applied else 0,
        "total_available_characters": total_chars,
        "total_selected_characters": total_chars if not applied else 0,
        "families": families,
    }


def _apply_dataset_mixture(
    documents: list[Document],
    weights: dict[str, float],
    progress: Optional[Callable[[Any], None]],
) -> tuple[list[Document], dict[str, Any]]:
    """Apply weighted sampling by source family.

    Args:
        documents: Loaded documents before mixture sampling.
        weights: Requested mixture percentages.
        progress: Optional progress callback.

    Returns:
        Selected documents and mixture report.
    """

    clean_weights: dict[str, float] = {}
    for family in MIXTURE_LABELS:
        try:
            clean_weights[family] = max(0.0, float(weights.get(family, 0.0) or 0.0))
        except (TypeError, ValueError):
            clean_weights[family] = 0.0
    requested_total = sum(clean_weights.values())
    if requested_total <= 0.0:
        return documents, _empty_mixture_report(clean_weights, documents, applied=False, reason="No positive mixture weights.")

    grouped: dict[str, list[Document]] = {key: [] for key in MIXTURE_LABELS}
    for document in documents:
        grouped.setdefault(_document_mixture_family(document), []).append(document)
    available_families = {
        family: items
        for family, items in grouped.items()
        if items and clean_weights.get(family, 0.0) > 0.0
    }
    if not available_families:
        return documents, _empty_mixture_report(
            clean_weights,
            documents,
            applied=False,
            reason="No documents matched positive mixture weights.",
        )

    total_available_chars = sum(len(document.text) for document in documents)
    total_budget = total_available_chars
    active_weight_total = sum(clean_weights[family] for family in available_families)
    sorted_groups = {
        family: sorted(items, key=_stable_document_sort_key)
        for family, items in available_families.items()
    }
    selected_by_family: dict[str, list[Document]] = {family: [] for family in MIXTURE_LABELS}
    selected_ids: set[int] = set()
    remaining_budget = 0

    for family, items in sorted_groups.items():
        target_chars = int(total_budget * clean_weights[family] / active_weight_total)
        selected_chars = 0
        for document in items:
            if selected_chars >= target_chars and selected_by_family[family]:
                break
            selected_by_family[family].append(document)
            selected_ids.add(id(document))
            selected_chars += len(document.text)
        remaining_budget += max(0, target_chars - selected_chars)

    if remaining_budget > 0:
        weighted_families = sorted(sorted_groups, key=lambda family: clean_weights[family], reverse=True)
        for family in weighted_families:
            for document in sorted_groups[family]:
                if remaining_budget <= 0:
                    break
                if id(document) in selected_ids:
                    continue
                selected_by_family[family].append(document)
                selected_ids.add(id(document))
                remaining_budget -= len(document.text)

    selected_documents = [
        document
        for family in MIXTURE_LABELS
        for document in selected_by_family.get(family, [])
    ]
    selected_documents = sorted(selected_documents, key=lambda document: (str(document.path), document.kind, document.language or ""))
    selected_total_chars = sum(len(document.text) for document in selected_documents)
    report = {
        "applied": True,
        "reason": "",
        "total_available_documents": len(documents),
        "total_selected_documents": len(selected_documents),
        "total_available_characters": total_available_chars,
        "total_selected_characters": selected_total_chars,
        "families": {},
    }
    for family, label in MIXTURE_LABELS.items():
        available = grouped.get(family, [])
        selected = selected_by_family.get(family, [])
        available_chars = sum(len(document.text) for document in available)
        selected_chars = sum(len(document.text) for document in selected)
        report["families"][family] = {
            "label": label,
            "requested_weight": clean_weights.get(family, 0.0),
            "available_documents": len(available),
            "available_characters": available_chars,
            "selected_documents": len(selected),
            "selected_characters": selected_chars,
            "actual_percent": (selected_chars * 100.0 / selected_total_chars) if selected_total_chars else 0.0,
            "dropped_documents": max(0, len(available) - len(selected)),
            "dropped_characters": max(0, available_chars - selected_chars),
        }
    if len(selected_documents) != len(documents):
        _emit(
            progress,
            f"Weighted sampler selected {len(selected_documents):,}/{len(documents):,} samples "
            f"({selected_total_chars:,}/{total_available_chars:,} characters).",
            49,
        )
    return selected_documents or documents, report


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
    (
        documents,
        manifest,
        cached_file_count,
        processed_file_count,
        skipped_file_count,
        failed_file_count,
    ) = _load_documents_with_cache(config, progress, should_stop)
    if should_stop and should_stop():
        raise RuntimeError("Dataset preparation stopped by user.")
    if not documents:
        raise ValueError("No supported text, PDF, or JSONL documents were found.")
    documents, mixture_report = _apply_dataset_mixture(documents, config.mixture_weights, progress)
    if should_stop and should_stop():
        raise RuntimeError("Dataset preparation stopped by user.")
    if not documents:
        raise ValueError("Dataset mixture selected no documents. Adjust mixture weights and try again.")

    all_text = "\n".join(doc.text for doc in documents)
    character_count = len(all_text)
    unique_words = len({word.lower() for word in all_text.split()})
    suggested_vocab_size = estimate_vocab_size(character_count, unique_words)
    selected_vocab_size = config.vocab_size or suggested_vocab_size
    warning = content_warning(character_count)
    code_sample_count = sum(1 for doc in documents if doc.kind == "code")
    conversation_sample_count = sum(1 for doc in documents if doc.kind in {"conversation", "instruction"})
    prose_sample_count = sum(1 for doc in documents if doc.kind not in {"code", "conversation", "instruction"})
    _emit(progress, f"Content size: {character_count:,} characters across {len(documents)} files.", 45)
    if config.code_training_mode:
        _emit(progress, f"Code mode: {code_sample_count:,} code samples, {prose_sample_count:,} prose samples.", 46)
    if conversation_sample_count:
        _emit(progress, f"Conversation data: {conversation_sample_count:,} dialogue/instruction samples.", 46)
    if cached_file_count or processed_file_count:
        _emit(progress, f"Cache: reused {cached_file_count:,} file(s), processed {processed_file_count:,} file(s).", 47)
    if skipped_file_count or failed_file_count:
        _emit(progress, f"Quality: skipped {skipped_file_count:,} empty file(s), failed {failed_file_count:,} file(s).", 48)
    if config.mixture_weights:
        mixture_parts = []
        for key, value in config.mixture_weights.items():
            try:
                numeric_value = float(value or 0.0)
            except (TypeError, ValueError):
                numeric_value = 0.0
            if numeric_value > 0.0:
                mixture_parts.append(f"{key.replace('_', ' ')} {numeric_value:.1f}%")
        mixture_text = ", ".join(mixture_parts)
        if mixture_text:
            _emit(progress, f"Dataset mixture plan: {mixture_text}.", 49)
    if mixture_report.get("applied"):
        for family in MIXTURE_LABELS:
            row = mixture_report.get("families", {}).get(family, {})
            if int(row.get("selected_documents", 0) or 0) > 0 or float(row.get("requested_weight", 0.0) or 0.0) > 0.0:
                _emit(
                    progress,
                    (
                        f"Mixture {row.get('label', family)}: requested {float(row.get('requested_weight', 0.0) or 0.0):.1f}%, "
                        f"selected {int(row.get('selected_documents', 0) or 0):,}/"
                        f"{int(row.get('available_documents', 0) or 0):,} sample(s), "
                        f"actual {float(row.get('actual_percent', 0.0) or 0.0):.1f}%."
                    ),
                    49,
                )
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
        reasoning_sample_mode=config.reasoning_sample_mode,
    )
    corpus_text = corpus_path.read_text(encoding="utf-8")
    tokenizer_path = config.output_dir / "tokenizer.json"
    if should_stop and should_stop():
        raise RuntimeError("Dataset preparation stopped by user.")
    tokenizer, reuse_tokenizer, tokenizer_imported, tokenizer_source_path = _load_or_create_tokenizer(
        config,
        corpus_path,
        tokenizer_path,
        selected_vocab_size,
        progress,
        should_stop,
    )
    validate_training_tokenizer(tokenizer)

    _emit(progress, "Encoding corpus into token IDs...", 78)
    if should_stop and should_stop():
        raise RuntimeError("Dataset preparation stopped by user.")
    tokens = encode_text(tokenizer, corpus_text)
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
        "train_token_count": len(train_tokens),
        "val_token_count": len(val_tokens),
        "code_sample_count": code_sample_count,
        "prose_sample_count": prose_sample_count,
        "conversation_sample_count": conversation_sample_count,
        "dataset_stage": config.dataset_stage,
        "conversation_datasets": config.conversation_datasets,
        "conversation_sample_limit": config.conversation_sample_limit,
        "mixture_weights": config.mixture_weights,
        "mixture_report": mixture_report,
        "suggested_vocab_size": suggested_vocab_size,
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "warning": warning,
        "source_files": [str(doc.path) for doc in documents],
        "cached_file_count": cached_file_count,
        "processed_file_count": processed_file_count,
        "skipped_file_count": skipped_file_count,
        "failed_file_count": failed_file_count,
        "source_file_count": len(manifest.get("files", {})),
        "prepare_mode": config.prepare_mode,
        "tokenizer_strategy": config.tokenizer_strategy,
        "reasoning_sample_mode": config.reasoning_sample_mode,
        "tokenizer_reused": reuse_tokenizer,
        "tokenizer_imported": tokenizer_imported,
        "tokenizer_source_path": tokenizer_source_path,
    }
    dataset_version = record_dataset_version(config.output_dir, summary, manifest)
    write_json(config.output_dir / "dataset_summary.json", summary)
    write_json(config.output_dir / "dataset_manifest.json", manifest)
    _emit(progress, f"Dataset version recorded: {dataset_version['version_id']}.", 98)
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
        conversation_sample_count,
        cached_file_count,
        processed_file_count,
        skipped_file_count,
        failed_file_count,
        str(dataset_version["version_id"]),
        int(dataset_version["version_number"]),
        mixture_report,
    )


def _resume_checkpoint_for(training_config: TrainingConfig) -> Optional[Path]:
    """Return the checkpoint that will be used for resume, if any.

    Args:
        training_config: Training configuration.

    Returns:
        Resume checkpoint path or ``None``.
    """

    if not training_config.resume:
        return None
    if training_config.resume_from_checkpoint:
        return Path(training_config.resume_from_checkpoint)
    return latest_checkpoint(training_config.output_dir / "checkpoints")


def _compatible_model_config(model_config: ModelConfig) -> dict[str, Any]:
    """Return checkpoint compatibility fields for a model config.

    Args:
        model_config: Current model configuration.

    Returns:
        Dictionary of architecture-shape fields.
    """

    data = dataclass_to_jsonable(model_config)
    return {
        key: data.get(key)
        for key in (
            "vocab_size",
            "context_length",
            "embedding_size",
            "head_count",
            "layer_count",
            "bias",
            "norm_type",
            "position_encoding",
            "mlp_type",
            "rope_theta",
            "attention_type",
        )
    }


def _tokenizer_files_match(left: Path, right: Path) -> bool:
    """Return whether tokenizer files are byte-identical or JSON-equivalent.

    Args:
        left: First tokenizer path.
        right: Second tokenizer path.

    Returns:
        True when tokenizers are compatible.
    """

    if file_sha256(left) == file_sha256(right):
        return True
    left_json = read_json(left, default=None)
    right_json = read_json(right, default=None)
    return left_json is not None and left_json == right_json


def _validate_resume_compatibility(
    data_dir: Path,
    tokenizer_path: Path,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> Optional[Path]:
    """Validate tokenizer and architecture before continuing training.

    Args:
        data_dir: Prepared dataset folder.
        tokenizer_path: Dataset tokenizer path.
        model_config: Current model architecture.
        training_config: Training configuration.

    Returns:
        Resume checkpoint path when one exists.

    Raises:
        ValueError: If tokenizer or architecture is incompatible.
    """

    resume_path = _resume_checkpoint_for(training_config)
    if not resume_path or not resume_path.exists() or not training_config.require_compatible_resume:
        return resume_path

    existing_tokenizer = training_config.output_dir / "tokenizer.json"
    if existing_tokenizer.exists():
        if not _tokenizer_files_match(existing_tokenizer, tokenizer_path):
            raise ValueError(
                "Resume safety check failed: the selected dataset tokenizer does not match the tokenizer "
                f"used by the existing model folder.\nExisting tokenizer: {existing_tokenizer}\n"
                f"Dataset tokenizer: {tokenizer_path}\n\nUse the same tokenizer policy for continued training, "
                "or choose a new model output folder to start a new model."
            )

    checkpoint = torch.load(resume_path, map_location="cpu")
    checkpoint_config = checkpoint.get("model_config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError(f"Resume safety check failed: checkpoint has no model_config: {resume_path}")

    current = _compatible_model_config(model_config)
    legacy_defaults = {
        "bias": True,
        "norm_type": "layernorm",
        "position_encoding": "learned",
        "mlp_type": "gelu",
        "rope_theta": 10000.0,
        "attention_type": "mha",
    }
    previous = {key: checkpoint_config.get(key, legacy_defaults.get(key)) for key in current}
    mismatches = {
        key: {"checkpoint": previous.get(key), "current": current.get(key)}
        for key in current
        if previous.get(key) != current.get(key)
    }
    if mismatches:
        mismatch_text = ", ".join(
            f"{key} checkpoint={value['checkpoint']} current={value['current']}"
            for key, value in mismatches.items()
        )
        raise ValueError(
            "Resume safety check failed: model architecture does not match the checkpoint. "
            f"{mismatch_text}. Keep architecture settings identical for continued training, "
            "or use a new model output folder."
        )

    return resume_path


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

    dataset_summary = read_json(data_dir / "dataset_summary.json", default={}) or {}
    dataset_lineage = read_json(data_dir / "dataset_lineage.json", default={}) or {}
    tokenizer = load_tokenizer(tokenizer_path)
    validate_training_tokenizer(tokenizer)
    train_tokens = json.loads((data_dir / "train_tokens.json").read_text(encoding="utf-8"))
    val_tokens = json.loads((data_dir / "val_tokens.json").read_text(encoding="utf-8"))

    if model_config.vocab_size != tokenizer.get_vocab_size():
        model_config.vocab_size = tokenizer.get_vocab_size()

    training_config.output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = _validate_resume_compatibility(data_dir, tokenizer_path, model_config, training_config)
    if resume_path:
        _emit(progress, f"Resume safety check passed: {resume_path}", 3)
    shutil.copy2(tokenizer_path, training_config.output_dir / "tokenizer.json")
    result = train_model(
        model_config,
        training_config,
        train_tokens,
        val_tokens,
        pad_token_id=token_id(tokenizer, PAD_TOKEN),
        progress=progress,
        should_stop=should_stop,
        decode_preview=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
    )
    training_summary = read_json(result.summary_path, default={}) or {}
    run_id = f"run_{utc_timestamp()}_{stable_json_hash({'dataset': dataset_summary.get('dataset_version'), 'model': training_summary.get('model_config'), 'training': training_summary.get('training_config')})}"
    lineage = {
        "schema": "micro_llm_model_lineage",
        "version": 1,
        "training_run_id": run_id,
        "created_at": utc_timestamp(),
        "dataset_dir": str(data_dir),
        "dataset_id": dataset_summary.get("dataset_id") or dataset_lineage.get("dataset_id"),
        "dataset_version": dataset_summary.get("dataset_version"),
        "dataset_fingerprint": (dataset_summary.get("dataset_version") or {}).get("source_fingerprint"),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "training_mode": training_config.training_mode,
        "fine_tune_from_checkpoint": (
            str(training_config.fine_tune_from_checkpoint)
            if training_config.fine_tune_from_checkpoint
            else None
        ),
        "peft_method": training_config.peft_method,
        "lora_rank": training_config.lora_rank if training_config.peft_method == "lora" else None,
        "lora_alpha": training_config.lora_alpha if training_config.peft_method == "lora" else None,
        "lora_target_modules": training_config.lora_target_modules if training_config.peft_method == "lora" else None,
        "resume_checkpoint": str(resume_path) if resume_path else None,
        "resume_safety_required": training_config.require_compatible_resume,
        "checkpoint_path": str(result.checkpoint_path),
        "summary_path": str(result.summary_path),
        "stopped": result.stopped,
    }
    training_summary["training_run_id"] = run_id
    training_summary["model_lineage"] = lineage
    write_json(result.summary_path, training_summary)
    write_json(training_config.output_dir / "model_lineage.json", lineage)
    write_json(training_config.output_dir / "dataset_summary.json", dataset_summary)
    return result
