from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator, Optional

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


def train_tokenizer(
    corpus_path: Path,
    output_path: Path,
    vocab_size: int = 8000,
    min_frequency: int = 2,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Tokenizer:
    """Train a byte-level BPE tokenizer.

    Args:
        corpus_path: Text corpus used for tokenizer training.
        output_path: Destination tokenizer JSON path.
        vocab_size: Target vocabulary size.
        min_frequency: Minimum token frequency for BPE merges.
        should_stop: Optional callback returning true when training should stop.

    Returns:
        Trained tokenizer instance.
    """

    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(_iter_corpus_lines(corpus_path, should_stop), trainer=trainer)
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        special_tokens=[
            (BOS_TOKEN, tokenizer.token_to_id(BOS_TOKEN)),
            (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
        ],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return tokenizer


def _iter_corpus_lines(corpus_path: Path, should_stop: Optional[Callable[[], bool]]) -> Iterator[str]:
    """Yield corpus lines and check for cancellation between chunks.

    Args:
        corpus_path: Text corpus path.
        should_stop: Optional cancellation callback.

    Yields:
        Corpus text lines.

    Raises:
        RuntimeError: If cancellation is requested.
    """

    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if should_stop and should_stop():
                raise RuntimeError("Dataset preparation stopped by user.")
            yield line


def load_tokenizer(path: Path) -> Tokenizer:
    """Load a tokenizer from disk.

    Args:
        path: Tokenizer JSON path.

    Returns:
        Loaded tokenizer.
    """

    return Tokenizer.from_file(str(path))


def token_id(tokenizer: Tokenizer, token: str) -> int:
    """Return the integer ID for a required special token.

    Args:
        tokenizer: Tokenizer to query.
        token: Token string to find.

    Returns:
        Token ID.

    Raises:
        ValueError: If the tokenizer does not contain the token.
    """

    value = tokenizer.token_to_id(token)
    if value is None:
        raise ValueError(f"Tokenizer is missing required token: {token}")
    return value


def missing_training_special_tokens(tokenizer: Tokenizer) -> list[str]:
    """Return special tokens missing from a tokenizer.

    Args:
        tokenizer: Tokenizer to inspect.

    Returns:
        Missing special token strings.
    """

    return [token for token in SPECIAL_TOKENS if tokenizer.token_to_id(token) is None]


def validate_training_tokenizer(tokenizer: Tokenizer) -> None:
    """Validate that a tokenizer can be used by the MicroGPT trainer.

    Args:
        tokenizer: Tokenizer to validate.

    Raises:
        ValueError: If required special tokens are missing.
    """

    missing = missing_training_special_tokens(tokenizer)
    if missing:
        raise ValueError(
            "Tokenizer is not compatible with Micro LLM Creator training. "
            f"Missing required special token(s): {', '.join(missing)}. "
            "Use a tokenizer created by this app, or import a tokenizer containing "
            "<pad>, <unk>, <bos>, and <eos>."
        )


def encode_text(tokenizer: Tokenizer, text: str) -> list[int]:
    """Encode text into token IDs.

    Args:
        tokenizer: Tokenizer used for encoding.
        text: Text to encode.

    Returns:
        List of token IDs.
    """

    return tokenizer.encode(text).ids
