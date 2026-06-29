from __future__ import annotations

import shutil
import subprocess
import json
from pathlib import Path
from typing import Optional

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
    optional_files = ["model_lineage.json", "dataset_summary.json"]
    for name in optional_files:
        source = project_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)
    benchmarks = project_dir / "benchmarks"
    if benchmarks.exists():
        shutil.copytree(benchmarks, output_dir / "benchmarks", dirs_exist_ok=True)
    manifest = {
        "schema": "micro_llm_export_bundle",
        "project_dir": str(project_dir),
        "output_dir": str(output_dir),
        "files": sorted(path.name for path in output_dir.iterdir()),
    }
    (output_dir / "export_summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
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


def export_hf_microgpt_package(project_dir: Path, output_dir: Optional[Path] = None) -> Path:
    """Export a MicroGPT checkpoint as an HF-style local model package.

    Args:
        project_dir: Trained model project folder.
        output_dir: Optional destination folder. Defaults to ``project_dir/hf_model``.

    Returns:
        HF-style package folder.

    Raises:
        ExportError: If required files are missing.
    """

    project_dir = Path(project_dir)
    output_dir = Path(output_dir) if output_dir else project_dir / "hf_model"
    checkpoint_path = project_dir / "final_model.pt"
    tokenizer_path = project_dir / "tokenizer.json"
    summary_path = project_dir / "training_summary.json"
    for path in (checkpoint_path, tokenizer_path, summary_path):
        if not path.exists():
            raise ExportError(f"Missing required file for HF package: {path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint.get("model_config")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_config, dict) or not state_dict:
        raise ExportError("Checkpoint must contain model_config and model_state_dict.")

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output_dir / "pytorch_model.bin")
    shutil.copy2(tokenizer_path, output_dir / "tokenizer.json")
    shutil.copy2(summary_path, output_dir / "training_summary.json")
    for optional in ("model_lineage.json", "dataset_summary.json"):
        source = project_dir / optional
        if source.exists():
            shutil.copy2(source, output_dir / optional)

    config = {
        "model_type": "microgpt",
        "architectures": ["MicroGPTForCausalLM"],
        "library_name": "micro-llm-creator",
        "llama_cpp_convertible": False,
        "vocab_size": model_config.get("vocab_size"),
        "n_positions": model_config.get("context_length"),
        "n_ctx": model_config.get("context_length"),
        "n_embd": model_config.get("embedding_size"),
        "n_head": model_config.get("head_count"),
        "n_layer": model_config.get("layer_count"),
        "dropout": model_config.get("dropout"),
        "bias": model_config.get("bias"),
        "norm_type": model_config.get("norm_type", "layernorm"),
        "position_encoding": model_config.get("position_encoding", "learned"),
        "mlp_type": model_config.get("mlp_type", "gelu"),
        "rope_theta": model_config.get("rope_theta", 10000.0),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "generation_config.json").write_text(
        json.dumps(
            {
                "max_new_tokens": 128,
                "temperature": 0.7,
                "top_k": 50,
                "do_sample": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_file": "tokenizer.json",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "model_max_length": model_config.get("context_length"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(_hf_readme(config), encoding="utf-8")
    return output_dir


def _hf_readme(config: dict[str, object]) -> str:
    """Create README text for a MicroGPT HF-style package.

    Args:
        config: Exported config dictionary.

    Returns:
        README Markdown.
    """

    return (
        "# MicroGPT HF-Style Package\n\n"
        "This folder was exported by Micro LLM Creator. It uses a Hugging Face-style "
        "layout (`config.json`, `pytorch_model.bin`, `tokenizer.json`) for portability, "
        "but `model_type` is `microgpt` and it is not directly convertible by llama.cpp as "
        "a Llama/Mistral/Gemma model.\n\n"
        "Load it with Micro LLM Creator's MicroGPT code, or build a custom Transformers "
        "model class that understands this config and tensor naming.\n\n"
        f"- Block style: {config.get('norm_type')} / {config.get('position_encoding')} / {config.get('mlp_type')}\n"
        f"- Context length: {config.get('n_ctx')}\n"
        f"- Embedding size: {config.get('n_embd')}\n"
        f"- Layers: {config.get('n_layer')}\n"
        f"- Heads: {config.get('n_head')}\n"
    )


def export_gguf_with_llama_cpp(project_dir: Path, llama_cpp_dir: Path, output_path: Path, outtype: str = "f16") -> Path:
    """Export a Hugging Face-compatible model through llama.cpp.

    Args:
        project_dir: Model project containing an ``hf_model`` folder.
        llama_cpp_dir: Local llama.cpp checkout folder.
        output_path: Destination GGUF file path.
        outtype: llama.cpp converter output type, usually f16 or f32.

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
            "GGUF export needs model_core/hf_model. Use Export HF Package first, "
            "but note MicroGPT packages are not llama.cpp-convertible unless llama.cpp "
            "has a matching MicroGPT converter/model implementation."
        )
    config_path = hf_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("model_type") == "microgpt":
            raise ExportError(
                "This hf_model folder is a MicroGPT package, not a llama.cpp-supported "
                "Llama/Mistral/Gemma model. Real GGUF export needs a supported HF model "
                "architecture or a custom llama.cpp MicroGPT converter."
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if outtype not in {"f16", "f32", "bf16", "q8_0"}:
        raise ExportError(f"Unsupported GGUF outtype for converter: {outtype}")

    subprocess.run(
        ["python", str(converter), str(hf_dir), "--outfile", str(output_path), "--outtype", outtype],
        check=True,
    )
    return output_path
