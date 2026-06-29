from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4


def utc_timestamp() -> str:
    """Return a compact UTC timestamp for artifact identifiers.

    Returns:
        Timestamp string.
    """

    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def stable_json_hash(value: Any) -> str:
    """Return a deterministic short hash for JSON-serializable data.

    Args:
        value: Data to hash.

    Returns:
        Twelve-character SHA-256 prefix.
    """

    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def read_json(path: Path, default: Optional[Any] = None) -> Any:
    """Read JSON from disk.

    Args:
        path: JSON file path.
        default: Value returned when the file does not exist or cannot be read.

    Returns:
        Parsed JSON or default.
    """

    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    """Write JSON to disk with a stable format.

    Args:
        path: Destination path.
        data: JSON-serializable data.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def next_version_number(lineage: dict[str, Any]) -> int:
    """Return the next dataset version number.

    Args:
        lineage: Existing lineage dictionary.

    Returns:
        Next one-based version number.
    """

    versions = lineage.get("versions", [])
    return len(versions) + 1


def ensure_dataset_lineage(output_dir: Path) -> dict[str, Any]:
    """Load or create dataset lineage metadata.

    Args:
        output_dir: Dataset output folder.

    Returns:
        Dataset lineage dictionary.
    """

    lineage_path = output_dir / "dataset_lineage.json"
    lineage = read_json(lineage_path, default=None)
    if isinstance(lineage, dict) and lineage.get("dataset_id"):
        lineage.setdefault("versions", [])
        return lineage
    return {
        "schema": "micro_llm_dataset_lineage",
        "version": 1,
        "dataset_id": f"ds_{uuid4().hex[:12]}",
        "created_at": utc_timestamp(),
        "versions": [],
    }


def record_dataset_version(output_dir: Path, summary: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Record a new dataset version and snapshot its metadata.

    Args:
        output_dir: Dataset output folder.
        summary: Dataset summary dictionary.
        manifest: Dataset manifest dictionary.

    Returns:
        Version metadata appended to lineage.
    """

    lineage = ensure_dataset_lineage(output_dir)
    version_number = next_version_number(lineage)
    source_fingerprint = stable_json_hash(
        {
            "files": manifest.get("files", {}),
            "dataset_config": summary.get("dataset_config", {}),
            "tokenizer_vocab_size": summary.get("tokenizer_vocab_size"),
            "tokenizer_sha256": summary.get("tokenizer_sha256"),
            "tokenizer_strategy": summary.get("tokenizer_strategy"),
        }
    )
    version_id = f"v{version_number:03d}_{utc_timestamp()}_{source_fingerprint}"
    version = {
        "version_number": version_number,
        "version_id": version_id,
        "created_at": utc_timestamp(),
        "source_fingerprint": source_fingerprint,
        "document_count": summary.get("document_count"),
        "character_count": summary.get("character_count"),
        "token_count": summary.get("token_count"),
        "tokenizer_vocab_size": summary.get("tokenizer_vocab_size"),
        "tokenizer_sha256": summary.get("tokenizer_sha256"),
        "code_sample_count": summary.get("code_sample_count"),
        "prose_sample_count": summary.get("prose_sample_count"),
        "prepare_mode": summary.get("prepare_mode"),
        "tokenizer_strategy": summary.get("tokenizer_strategy"),
        "summary_path": "dataset_summary.json",
        "manifest_path": "dataset_manifest.json",
        "snapshot_dir": f"versions/{version_id}",
    }
    lineage["updated_at"] = utc_timestamp()
    lineage.setdefault("versions", []).append(version)
    summary["dataset_id"] = lineage["dataset_id"]
    summary["dataset_version"] = version
    manifest["dataset_id"] = lineage["dataset_id"]
    manifest["dataset_version"] = version

    snapshot_dir = output_dir / "versions" / version_id
    write_json(snapshot_dir / "dataset_summary.json", summary)
    write_json(snapshot_dir / "dataset_manifest.json", manifest)
    write_json(output_dir / "dataset_lineage.json", lineage)
    return version
