from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Optional

from llm_trainer.data import Document, clean_text


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConversationDatasetPreset:
    """Built-in Hugging Face dataset recipe for conversation/instruction data.

    Attributes:
        dataset_id: Stable UI/config identifier.
        label: User-facing label with size hint.
        hf_path: Hugging Face dataset path.
        config_name: Optional Hugging Face dataset configuration name.
        split: Dataset split to load.
        stage: Recommended training stage: base, instruction, or conversation.
        description: Short user-facing purpose hint.
    """

    dataset_id: str
    label: str
    hf_path: str
    config_name: Optional[str]
    split: str
    stage: str
    description: str


CONVERSATION_DATASET_PRESETS: dict[str, ConversationDatasetPreset] = {
    "tinystories": ConversationDatasetPreset(
        "tinystories",
        "TinyStories (~2M short stories)",
        "roneneldan/TinyStories",
        None,
        "train",
        "base",
        "Language fluency, simple narrative structure, and basic world knowledge.",
    ),
    "wikitext_103": ConversationDatasetPreset(
        "wikitext_103",
        "WikiText-103 (~100M tokens)",
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        "train",
        "base",
        "Clean Wikipedia-style long-form text for grammar, facts, and language modeling.",
    ),
    "wikipedia_en": ConversationDatasetPreset(
        "wikipedia_en",
        "Wikipedia EN 2023 (large encyclopedia)",
        "wikimedia/wikipedia",
        "20231101.en",
        "train",
        "base",
        "Broad encyclopedia prose. Use a row limit unless you intentionally want a large download.",
    ),
    "fineweb_edu": ConversationDatasetPreset(
        "fineweb_edu",
        "FineWeb-Edu sample (large educational web)",
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        "train",
        "base",
        "High-quality educational web text for base language pretraining. Use a row limit.",
    ),
    "ultrachat_200k": ConversationDatasetPreset(
        "ultrachat_200k",
        "UltraChat 200K (~200K conversations)",
        "HuggingFaceH4/ultrachat_200k",
        None,
        "train_sft",
        "conversation",
        "Multi-turn assistant conversation and helpful response style.",
    ),
    "dailydialog": ConversationDatasetPreset(
        "dailydialog",
        "DailyDialog (~13K dialogues)",
        "pixelsandpointers/better_daily_dialog",
        None,
        "train",
        "conversation",
        "Natural everyday dialogue and short conversational turns.",
    ),
    "alpaca_52k": ConversationDatasetPreset(
        "alpaca_52k",
        "Alpaca 52K (~52K instructions)",
        "tatsu-lab/alpaca",
        None,
        "train",
        "instruction",
        "Instruction following with concise task-answer pairs.",
    ),
    "dolly_15k": ConversationDatasetPreset(
        "dolly_15k",
        "Dolly 15K (~15K instructions)",
        "databricks/databricks-dolly-15k",
        None,
        "train",
        "instruction",
        "Human-written instruction following, brainstorming, QA, and classification.",
    ),
    "oasst1": ConversationDatasetPreset(
        "oasst1",
        "OpenAssistant OASST1 (~88K messages)",
        "OpenAssistant/oasst1",
        None,
        "train",
        "conversation",
        "Assistant-style conversational messages and preference data text.",
    ),
    "slimorca": ConversationDatasetPreset(
        "slimorca",
        "SlimOrca (~517K examples)",
        "Open-Orca/SlimOrca",
        None,
        "train",
        "instruction",
        "Instruction and reasoning-style assistant answers.",
    ),
}

BASE_DATASET_IDS = [dataset_id for dataset_id, preset in CONVERSATION_DATASET_PRESETS.items() if preset.stage == "base"]
INSTRUCTION_DATASET_IDS = [dataset_id for dataset_id, preset in CONVERSATION_DATASET_PRESETS.items() if preset.stage == "instruction"]
CONVERSATION_DATASET_IDS = [dataset_id for dataset_id, preset in CONVERSATION_DATASET_PRESETS.items() if preset.stage == "conversation"]


def dataset_ids_for_stage(stage: str) -> list[str]:
    """Return online dataset IDs recommended for a training stage.

    Args:
        stage: Dataset/training stage.

    Returns:
        Dataset IDs for the selected stage.
    """

    if stage == "base":
        return BASE_DATASET_IDS
    if stage == "instruction":
        return INSTRUCTION_DATASET_IDS
    if stage == "conversation":
        return CONVERSATION_DATASET_IDS
    return []


def dataset_stage_label(stage: str) -> str:
    """Return a user-facing stage label.

    Args:
        stage: Dataset/training stage.

    Returns:
        Human-readable stage name.
    """

    return {
        "base": "Base pretraining",
        "instruction": "Instruction fine-tune",
        "conversation": "Conversation fine-tune",
    }.get(stage, "Custom")


def load_conversation_documents(
    dataset_ids: list[str],
    sample_limit: int,
    cache_dir: Path,
    lowercase: bool = False,
    progress: Optional[Callable[[Any], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> list[Document]:
    """Load selected Hugging Face conversation datasets as training documents.

    Args:
        dataset_ids: Preset IDs to load.
        sample_limit: Maximum rows per dataset. Zero means no limit.
        cache_dir: Hugging Face dataset cache directory.
        lowercase: Whether to lowercase extracted text.
        progress: Optional progress callback.
        should_stop: Optional cancellation callback.

    Returns:
        Conversation/instruction documents.
    """

    if not dataset_ids:
        return []
    documents: list[Document] = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    _emit(progress, f"Hugging Face dataset cache: {cache_dir}")
    LOGGER.info("Hugging Face dataset cache: %s", cache_dir)
    for preset_index, dataset_id in enumerate(dataset_ids, start=1):
        if should_stop and should_stop():
            raise RuntimeError("Dataset preparation stopped by user.")
        preset = CONVERSATION_DATASET_PRESETS.get(dataset_id)
        if preset is None:
            _emit(progress, f"Skipping unknown conversation dataset: {dataset_id}")
            LOGGER.warning("Skipping unknown conversation dataset: %s", dataset_id)
            continue
        _emit(
            progress,
            f"Downloading/loading {preset.label} into {cache_dir}...",
            8 + min(25, preset_index * 3),
        )
        LOGGER.info("Downloading/loading %s into %s", preset.label, cache_dir)
        rows = _load_preset_rows_in_subprocess(preset, sample_limit, cache_dir, lowercase, progress, should_stop)
        total = len(rows)
        loaded = 0
        for row_index, row in enumerate(rows):
            if should_stop and should_stop():
                raise RuntimeError("Dataset preparation stopped by user.")
            text = str(row.get("text") or "")
            kind = str(row.get("kind") or "conversation")
            if not text:
                continue
            documents.append(
                Document(
                    path=Path("__hf_datasets__") / preset.dataset_id / f"{row_index}.txt",
                    text=text,
                    kind=kind,
                    language=preset.dataset_id,
                )
            )
            loaded += 1
            if loaded % 1000 == 0:
                _emit(progress, f"{preset.label}: loaded {loaded:,}/{total:,} sample(s).")
        _emit(progress, f"{preset.label}: added {loaded:,} sample(s).")
        LOGGER.info("%s added %s sample(s)", preset.label, f"{loaded:,}")
    return documents


def _load_preset_rows_in_subprocess(
    preset: ConversationDatasetPreset,
    sample_limit: int,
    cache_dir: Path,
    lowercase: bool,
    progress: Optional[Callable[[Any], None]],
    should_stop: Optional[Callable[[], bool]],
) -> list[dict[str, str]]:
    """Extract a Hugging Face preset in a child process.

    Args:
        preset: Dataset preset to extract.
        sample_limit: Maximum rows to extract.
        cache_dir: Hugging Face cache directory.
        lowercase: Whether to lowercase text.
        progress: Optional progress callback.
        should_stop: Optional cancellation callback.

    Returns:
        Extracted text rows.
    """

    extract_dir = cache_dir / "_micro_llm_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    output_path = extract_dir / f"{preset.dataset_id}_{max(sample_limit, 0)}_{int(lowercase)}.jsonl"
    if output_path.exists():
        output_path.unlink()
    command = [
        sys.executable,
        "-m",
        "llm_trainer.conversation_datasets",
        "extract",
        "--dataset-id",
        preset.dataset_id,
        "--sample-limit",
        str(sample_limit),
        "--cache-dir",
        str(cache_dir),
        "--output-jsonl",
        str(output_path),
    ]
    if preset.config_name:
        command.extend(["--config-name", preset.config_name])
    if lowercase:
        command.append("--lowercase")
    LOGGER.info("Starting Hugging Face extraction subprocess: %s", " ".join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_hf_subprocess_environment(cache_dir),
    )
    assert process.stdout is not None
    for line in process.stdout:
        text = line.strip()
        if text:
            LOGGER.info("[hf:%s] %s", preset.dataset_id, text)
            _emit(progress, text)
        if should_stop and should_stop():
            process.terminate()
            raise RuntimeError("Dataset preparation stopped by user.")
    return_code = process.wait()
    LOGGER.info("Hugging Face extraction subprocess finished for %s with code %s", preset.dataset_id, return_code)
    if return_code != 0:
        raise RuntimeError(
            f"Hugging Face dataset loader exited with code {return_code} while loading {preset.label}. "
            "Check micro_llm_creator.log and micro_llm_creator_faults.log."
        )
    if not output_path.exists():
        raise RuntimeError(f"Hugging Face extraction did not create output: {output_path}")
    rows: list[dict[str, str]] = []
    with output_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _hf_subprocess_environment(cache_dir: Path) -> dict[str, str]:
    """Build a Hugging Face environment that stays inside the project cache.

    Args:
        cache_dir: Project-local Hugging Face cache directory.

    Returns:
        Environment variables for the extraction subprocess.
    """

    env = os.environ.copy()
    env["HF_HOME"] = str(cache_dir / "hf_home")
    env["HF_HUB_CACHE"] = str(cache_dir / "hub")
    env["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    return env


def _emit(progress: Optional[Callable[[Any], None]], message: str, percent: Optional[int] = None) -> None:
    """Emit a progress event if a callback is available."""

    if progress:
        progress({"message": message, "percent": percent})


def _conversation_text_from_row(row: dict[str, Any]) -> tuple[str, str]:
    """Extract tagged conversation/instruction text from a dataset row.

    Args:
        row: Hugging Face row.

    Returns:
        Text and document kind.
    """

    messages = row.get("messages") or row.get("conversation") or row.get("conversations")
    if isinstance(messages, list):
        rendered = _render_message_list(messages)
        if rendered:
            return rendered, "conversation"
    dialogue = row.get("dialogue")
    if isinstance(dialogue, list):
        turns = [f"{'User' if index % 2 == 0 else 'Assistant'}: {value}" for index, value in enumerate(dialogue)]
        return "\n".join(turns), "conversation"
    instruction = str(row.get("instruction") or row.get("prompt") or row.get("question") or "").strip()
    input_text = str(row.get("input") or row.get("context") or "").strip()
    output = str(row.get("output") or row.get("response") or row.get("answer") or row.get("completion") or "").strip()
    if instruction and output:
        user = instruction if not input_text else f"{instruction}\n\n{input_text}"
        return f"User: {user}\nAssistant: {output}", "instruction"
    for key in ("text", "story", "content"):
        value = row.get(key)
        if value:
            return str(value), "prose"
    return "", "prose"


def _render_message_list(messages: list[Any]) -> str:
    """Render common message-list schemas into role-prefixed turns."""

    turns: list[str] = []
    for index, message in enumerate(messages):
        if isinstance(message, dict):
            role = str(message.get("role") or message.get("from") or message.get("speaker") or "").strip()
            content = str(message.get("content") or message.get("value") or message.get("text") or "").strip()
        else:
            role = "user" if index % 2 == 0 else "assistant"
            content = str(message).strip()
        if not content:
            continue
        label = "Assistant" if role.lower() in {"assistant", "gpt", "bot"} else "User"
        if role.lower() in {"system"}:
            label = "System"
        turns.append(f"{label}: {content}")
    return "\n".join(turns)


def _extract_preset_to_jsonl(
    dataset_id: str,
    sample_limit: int,
    cache_dir: Path,
    output_jsonl: Path,
    lowercase: bool,
) -> None:
    """Extract one Hugging Face preset to JSONL for the parent app.

    Args:
        dataset_id: Preset ID.
        sample_limit: Maximum rows to extract.
        cache_dir: Hugging Face cache directory.
        output_jsonl: JSONL output path.
        lowercase: Whether to lowercase extracted text.
    """

    preset = CONVERSATION_DATASET_PRESETS[dataset_id]
    os.environ.update(_hf_subprocess_environment(cache_dir))
    print(f"Importing datasets package for {preset.label}.", flush=True)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the datasets package to use Hugging Face conversation datasets.") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading/loading {preset.label} into {cache_dir}.", flush=True)
    if preset.config_name:
        dataset = load_dataset(preset.hf_path, preset.config_name, split=preset.split, cache_dir=str(cache_dir))
    else:
        dataset = load_dataset(preset.hf_path, split=preset.split, cache_dir=str(cache_dir))
    row_count = len(dataset) if hasattr(dataset, "__len__") else 0
    print(f"Loaded {preset.hf_path} split {preset.split} with {row_count or 'unknown'} row(s).", flush=True)
    limit = row_count if sample_limit <= 0 or row_count <= 0 else min(sample_limit, row_count)
    if limit and hasattr(dataset, "select"):
        dataset = dataset.select(range(limit))
    loaded = 0
    with output_jsonl.open("w", encoding="utf-8") as file:
        if dataset_id == "dailydialog":
            loaded = _write_daily_dialog_rows(dataset, file, lowercase)
        else:
            for row in dataset:
                text, kind = _conversation_text_from_row(dict(row))
                text = clean_text(text, lowercase=lowercase)
                if not text:
                    continue
                file.write(json.dumps({"text": text, "kind": kind}, ensure_ascii=False) + "\n")
                loaded += 1
                if loaded % 1000 == 0:
                    print(f"{preset.label}: extracted {loaded:,}/{limit:,} sample(s).", flush=True)
    print(f"{preset.label}: wrote {loaded:,} sample(s) to {output_jsonl}.", flush=True)


def _write_daily_dialog_rows(dataset: Any, file: Any, lowercase: bool) -> int:
    """Group DailyDialog utterance rows into dialogue samples.

    Args:
        dataset: Hugging Face dataset rows.
        file: Open JSONL file handle.
        lowercase: Whether to lowercase extracted text.

    Returns:
        Number of written dialogue samples.
    """

    dialogues: dict[str, list[str]] = {}
    order: list[str] = []
    for row in dataset:
        value = dict(row)
        dialog_id = str(value.get("dialog_id", len(order)))
        utterance = str(value.get("utterance") or "").strip()
        if not utterance:
            continue
        if dialog_id not in dialogues:
            dialogues[dialog_id] = []
            order.append(dialog_id)
        dialogues[dialog_id].append(utterance)
    written = 0
    for dialog_id in order:
        turns = dialogues[dialog_id]
        if not turns:
            continue
        text = "\n".join(
            f"{'User' if index % 2 == 0 else 'Assistant'}: {utterance}"
            for index, utterance in enumerate(turns)
        )
        text = clean_text(text, lowercase=lowercase)
        if not text:
            continue
        file.write(json.dumps({"text": text, "kind": "conversation"}, ensure_ascii=False) + "\n")
        written += 1
        if written % 1000 == 0:
            print(f"DailyDialog: extracted {written:,} dialogue sample(s).", flush=True)
    return written


def main() -> None:
    """Run conversation dataset helper commands."""

    parser = argparse.ArgumentParser(description="Micro LLM conversation dataset helper")
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--dataset-id", required=True, choices=sorted(CONVERSATION_DATASET_PRESETS))
    extract_parser.add_argument("--sample-limit", type=int, default=20000)
    extract_parser.add_argument("--cache-dir", required=True)
    extract_parser.add_argument("--output-jsonl", required=True)
    extract_parser.add_argument("--config-name", default=None)
    extract_parser.add_argument("--lowercase", action="store_true")
    args = parser.parse_args()
    if args.command == "extract":
        _extract_preset_to_jsonl(
            dataset_id=args.dataset_id,
            sample_limit=args.sample_limit,
            cache_dir=Path(args.cache_dir),
            output_jsonl=Path(args.output_jsonl),
            lowercase=args.lowercase,
        )


if __name__ == "__main__":
    main()
