"""Thai text encoder — char-level tokenization via input vocab.

No diacritics to strip (Thai has none); the cleaner just normalizes
whitespace and folds common variants (e.g. fullwidth punct → ASCII).
"""

from __future__ import annotations

import re
import unicodedata

from .constants import (
    INPUT_VOCAB,
    PAD_ID,
    UNK_ID,
    input_char_to_id,
)

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_thai(text: str) -> str:
    """NFKC normalize + collapse whitespace.

    NFKC folds compatibility chars (fullwidth ASCII, ligatures) into
    canonical form so the encoder sees a single representation.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


class ThaiEncoder:
    """Maps Thai text → integer token IDs using the input vocab."""

    def __init__(self, cleaner: str = "thai") -> None:
        if cleaner not in ("basic", "thai"):
            raise ValueError(f"unknown cleaner: {cleaner}")
        self.cleaner = cleaner
        self.input_symbol_to_id: dict[str, int] = {s: i for i, s in enumerate(INPUT_VOCAB)}
        self.input_id_to_symbol: list[str] = INPUT_VOCAB
        self.input_pad_id = PAD_ID

    def clean(self, text: str) -> str:
        if self.cleaner == "basic":
            return _WHITESPACE_RE.sub(" ", text).strip()
        return normalize_thai(text)

    def encode(self, text: str) -> list[int]:
        return [input_char_to_id(c) for c in text]

    def decode_input(self, ids: list[int]) -> str:
        return "".join(
            self.input_id_to_symbol[i] for i in ids
            if i != self.input_pad_id and i != UNK_ID
        )
