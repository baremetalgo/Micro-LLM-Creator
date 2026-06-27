from __future__ import annotations

from pathlib import Path

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
) -> Tokenizer:
    """Train a byte-level BPE tokenizer.

    Args:
        corpus_path: Text corpus used for tokenizer training.
        output_path: Destination tokenizer JSON path.
        vocab_size: Target vocabulary size.
        min_frequency: Minimum token frequency for BPE merges.

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
    tokenizer.train([str(corpus_path)], trainer)
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


def encode_text(tokenizer: Tokenizer, text: str) -> list[int]:
    """Encode text into token IDs.

    Args:
        tokenizer: Tokenizer used for encoding.
        text: Text to encode.

    Returns:
        List of token IDs.
    """

    return tokenizer.encode(text).ids
