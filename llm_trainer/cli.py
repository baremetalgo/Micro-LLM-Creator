from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import DatasetConfig, ModelConfig, TrainingConfig
from .services import build_dataset
from .tokenizer import PAD_TOKEN, load_tokenizer, token_id
from .training import train_model


def prepare(args: argparse.Namespace) -> None:
    """Prepare a dataset from command-line arguments.

    Args:
        args: Parsed command-line arguments for the prepare command.
    """

    def print_progress(event: object) -> None:
        """Print a progress event in CLI-friendly form.

        Args:
            event: Progress dictionary or message.
        """

        if isinstance(event, dict):
            message = event.get("message")
            percent = event.get("percent")
            prefix = f"[{percent:>3}%] " if percent is not None else ""
            if message:
                print(prefix + str(message))
        else:
            print(event)

    config = DatasetConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        context_length=args.context_length,
        validation_split=args.validation_split,
        lowercase=args.lowercase,
        max_workers=args.max_workers,
        code_training_mode=args.code_training_mode,
        include_prose=not args.exclude_prose,
        include_source_code=not args.exclude_source_code,
        extract_code_blocks=not args.no_extract_code_blocks,
        preserve_indentation=not args.no_preserve_indentation,
        generate_instruction_samples=not args.no_instruction_samples,
    )
    result = build_dataset(config, progress=print_progress)
    print(
        f"Documents: {result.document_count} | Characters: {result.character_count} | "
        f"Tokens: {result.token_count} | Vocab: {result.vocab_size}"
    )


def train(args: argparse.Namespace) -> None:
    """Train a model from command-line arguments.

    Args:
        args: Parsed command-line arguments for the train command.
    """

    data_dir = Path(args.data_dir)
    tokenizer = load_tokenizer(data_dir / "tokenizer.json")
    train_tokens = json.loads((data_dir / "train_tokens.json").read_text(encoding="utf-8"))
    val_tokens = json.loads((data_dir / "val_tokens.json").read_text(encoding="utf-8"))

    model_config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        context_length=args.context_length,
        embedding_size=args.embedding_size,
        head_count=args.head_count,
        layer_count=args.layer_count,
        dropout=args.dropout,
    )
    training_config = TrainingConfig(
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation=args.gradient_accumulation,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        use_amp=args.use_amp,
        device=args.device,
        resume=not args.no_resume,
        resume_from_checkpoint=Path(args.resume_checkpoint) if args.resume_checkpoint else None,
    )
    training_config.output_dir.mkdir(parents=True, exist_ok=True)
    (training_config.output_dir / "tokenizer.json").write_text(
        (data_dir / "tokenizer.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    result = train_model(
        model_config,
        training_config,
        train_tokens,
        val_tokens,
        pad_token_id=token_id(tokenizer, PAD_TOKEN),
    )
    print(f"Saved model: {result.checkpoint_path}")
    print(f"Saved summary: {result.summary_path}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(description="Small LLM trainer backend")
    subparsers = parser.add_subparsers(required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Load documents and train tokenizer")
    prepare_parser.add_argument("--input_dir", required=True)
    prepare_parser.add_argument("--output_dir", required=True)
    prepare_parser.add_argument("--vocab_size", type=int, default=None)
    prepare_parser.add_argument("--min_frequency", type=int, default=2)
    prepare_parser.add_argument("--context_length", type=int, default=128)
    prepare_parser.add_argument("--validation_split", type=float, default=0.1)
    prepare_parser.add_argument("--lowercase", action="store_true")
    prepare_parser.add_argument("--max_workers", type=int, default=4)
    prepare_parser.add_argument("--code_training_mode", action="store_true")
    prepare_parser.add_argument("--exclude_prose", action="store_true")
    prepare_parser.add_argument("--exclude_source_code", action="store_true")
    prepare_parser.add_argument("--no_extract_code_blocks", action="store_true")
    prepare_parser.add_argument("--no_preserve_indentation", action="store_true")
    prepare_parser.add_argument("--no_instruction_samples", action="store_true")
    prepare_parser.set_defaults(func=prepare)

    train_parser = subparsers.add_parser("train", help="Train a MicroGPT model")
    train_parser.add_argument("--data_dir", required=True)
    train_parser.add_argument("--output_dir", required=True)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--batch_size", type=int, default=16)
    train_parser.add_argument("--context_length", type=int, default=128)
    train_parser.add_argument("--embedding_size", type=int, default=256)
    train_parser.add_argument("--head_count", type=int, default=4)
    train_parser.add_argument("--layer_count", type=int, default=4)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--learning_rate", type=float, default=3e-4)
    train_parser.add_argument("--gradient_accumulation", type=int, default=1)
    train_parser.add_argument("--eval_interval", type=int, default=100)
    train_parser.add_argument("--save_interval", type=int, default=500)
    train_parser.add_argument("--use_amp", action="store_true")
    train_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train_parser.add_argument("--no_resume", action="store_true")
    train_parser.add_argument("--resume_checkpoint", default=None)
    train_parser.set_defaults(func=train)
    return parser


def main() -> None:
    """Run the command-line interface."""

    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
