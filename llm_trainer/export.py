from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import torch


class ExportError(RuntimeError):
    """Raised when model export or quantization cannot be completed."""

    pass


def export_project_bundle(project_dir: Path, output_dir: Path) -> Path:
    """Create a portable model bundle.

    Args:
        project_dir: Trained model project folder.
        output_dir: Destination export folder.

    Returns:
        Export folder path.

    Raises:
        ExportError: If required model artifacts are missing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    required = ["final_model.pt", "tokenizer.json", "training_summary.json"]
    for name in required:
        source = project_dir / name
        if not source.exists():
            raise ExportError(f"Missing required file for export: {source}")
        shutil.copy2(source, output_dir / name)
    return output_dir


def quantize_checkpoint(checkpoint_path: Path, output_path: Path, mode: str = "fp16") -> Path:
    """Create a smaller inference checkpoint.

    Args:
        checkpoint_path: Source PyTorch checkpoint path.
        output_path: Destination quantized checkpoint path.
        mode: Quantization mode. Currently only ``fp16`` is supported.

    Returns:
        Quantized checkpoint path.

    Raises:
        ExportError: If the checkpoint is missing or mode is unsupported.
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    if not checkpoint_path.exists():
        raise ExportError(f"Checkpoint not found: {checkpoint_path}")

    mode = mode.lower()
    if mode not in {"fp16", "float16"}:
        raise ExportError("Only FP16 checkpoint quantization is currently supported for MicroGPT.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict")
    if not state_dict:
        raise ExportError("Checkpoint does not contain model_state_dict.")

    checkpoint["model_state_dict"] = {
        key: value.half() if torch.is_floating_point(value) else value
        for key, value in state_dict.items()
    }
    checkpoint["quantization"] = {
        "mode": "fp16",
        "source": str(checkpoint_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return output_path


def export_gguf_with_llama_cpp(project_dir: Path, llama_cpp_dir: Path, output_path: Path) -> Path:
    """Export a Hugging Face-compatible model through llama.cpp.

    Args:
        project_dir: Model project containing an ``hf_model`` folder.
        llama_cpp_dir: Local llama.cpp checkout folder.
        output_path: Destination GGUF file path.

    Returns:
        GGUF output path.

    Raises:
        ExportError: If converter or HF model folder is missing.
    """
    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not converter.exists():
        raise ExportError(f"Could not find llama.cpp converter: {converter}")

    hf_dir = project_dir / "hf_model"
    if not hf_dir.exists():
        raise ExportError(
            "GGUF export needs an HF-compatible model folder. "
            "That conversion layer is the next backend step."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["python", str(converter), str(hf_dir), "--outfile", str(output_path)],
        check=True,
    )
    return output_path
