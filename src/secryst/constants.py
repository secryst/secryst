"""Thai + IPA alphabet constants for secryst.

Input vocabulary (Thai chars):
  - 44 consonants (U+0E01..U+0E2E minus a few archaic)
  - 15 vowel symbols (U+0E34..U+0E4D as monospacing chars)
  - 4 tone marks (ไม้เอก ่, ไม้โท ้, ไม้ตรี ๊, ไม้จัตวา ๋)
  - 10 Thai digits (U+0E50..U+0E59) — optional, fold to ASCII digits in cleaner
  - Leading consonant marker: Maihan-akat (U+0E31), Sara aa (U+0E32), Nikhahit
  - Common punctuation, whitespace

Output vocabulary (IPA):
  - Standard IPA consonants/vowels used in Thai transcriptions
  - Tone marks: ◌̋ (high), ◌̌ (rising), ◌̀ (low), ◌̄ (mid), ◌̂ (falling)
  - Length mark: ː
  - Stress: ˈ (primary), ˌ (secondary)
  - Syllable boundary: .

The vocab is intentionally generous — unknown chars at inference time
get mapped to a special "O" token. The model is robust to OCR / spelling
variation; we don't drop chars at encode time.
"""

from __future__ import annotations

# Special tokens — first 4 IDs reserved.
PAD_SYMBOL = "P"
BOS_SYMBOL = "B"   # beginning of sequence (decoder input)
EOS_SYMBOL = "E"   # end of sequence (decoder target)
UNK_SYMBOL = "U"   # unknown char

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

# ---- Thai input vocabulary -------------------------------------------

# Core Thai consonants (modern usage). U+0E01 .. U+0E2E minus deprecated.
THAI_CONSONANTS: tuple[str, ...] = (
    "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ",
)

# Vowels and combining marks used in Thai orthography.
# Includes leading vowels (เ แ โ ใ ไ) which precede consonants and are
# extremely common in modern Thai (เ she, แ aek, โ boat, ใ use, ไ go).
THAI_VOWELS: tuple[str, ...] = (
    "ะ ั า ำ ิ ี ึ ื ุ ู ๅ ็ ่ ้ ๊ ๋ ์ ํ ๎ เ แ โ ใ ไ ไๆ ๐ ๑ ๒ ๓ ๔ ๕ ๖ ๗ ๘ ๙",
)

# Pair separators / punctuation in modern Thai text.
THAI_PUNCT: tuple[str, ...] = ("ฯ", "ๅ", ".", ",", "!", "?", "-", "—", "…", "\"", "'", "(", ")", " ")

# Assemble the full input char list. Deduplicate, preserve order.
_input_set: list[str] = [PAD_SYMBOL, BOS_SYMBOL, EOS_SYMBOL, UNK_SYMBOL]
for group in (THAI_CONSONANTS, THAI_VOWELS, THAI_PUNCT):
    for ch in group:
        for c in ch:
            if c == " ":
                continue
            if c not in _input_set:
                _input_set.append(c)

INPUT_VOCAB: list[str] = _input_set
INPUT_VOCAB_SIZE = len(INPUT_VOCAB)

# Lookup: char → input ID.
_INPUT_CHAR_TO_ID: dict[str, int] = {c: i for i, c in enumerate(INPUT_VOCAB)}


def input_char_to_id(c: str) -> int:
    """Return vocab ID for char, or UNK_ID if not in vocab."""
    return _INPUT_CHAR_TO_ID.get(c, UNK_ID)


def input_id_to_char(i: int) -> str:
    """Inverse lookup. Out-of-range → UNK."""
    if 0 <= i < len(INPUT_VOCAB):
        return INPUT_VOCAB[i]
    return UNK_SYMBOL


# ---- IPA output vocabulary -------------------------------------------

# IPA symbols used in Thai transcriptions (Yamasaki 2022 / standard
# phonology refs). Tone diacritics per IPA 2015 convention.
IPA_SYMBOLS: tuple[str, ...] = (
    # Vowels
    "a", "aː", "i", "iː", "ɯ", "ɯː", "u", "uː",
    "e", "eː", "ɛ", "ɛː", "o", "oː", "ɔ", "ɔː",
    "ɤ", "ɤː", "ə", "ɨ",
    # Consonants (initial + final positions use same symbol)
    "p", "pʰ", "b", "t", "tʰ", "d", "k", "kʰ", "g",
    "c", "cʰ", "ʔ", "m", "n", "ŋ", "ɲ", "j", "w",
    "l", "r", "s", "h", "f", "v", "z", "ʂ", "ɖ", "ʈ",
    # Aspiration / affricate clusters
    "tɕ", "tɕʰ",
    # Tone diacritics (IPA 2015)
    "˥", "˧˥", "˨˩˧", "˨˩", "˧", "˥˩",
    # Length, stress, syllable boundary
    "ː", "ˈ", "ˌ", ".",
    # Space (between syllables/words in our tokenization)
    " ",
)

# Build output vocab. Treat multi-char tokens (e.g. "aː", "pʰ", "tɕ") as
# single IDs — the dataset layer tokenizes IPA strings using this lexicon.
_output_set: list[str] = [PAD_SYMBOL, BOS_SYMBOL, EOS_SYMBOL, UNK_SYMBOL]
for tok in IPA_SYMBOLS:
    if tok not in _output_set:
        _output_set.append(tok)

OUTPUT_VOCAB: list[str] = _output_set
OUTPUT_VOCAB_SIZE = len(OUTPUT_VOCAB)

# Longest-token-first tokenizer for IPA. Sorted by length desc so "aː"
# matches before "a", "pʰ" before "p", etc.
_IPA_TOKENS_BY_LEN: list[str] = sorted(
    [tok for tok in OUTPUT_VOCAB if len(tok) > 0 and tok != PAD_SYMBOL],
    key=lambda t: -len(t),
)

_OUTPUT_TOKEN_TO_ID: dict[str, int] = {t: i for i, t in enumerate(OUTPUT_VOCAB)}


def tokenize_ipa(ipa: str) -> list[int]:
    """Greedy longest-match tokenizer for IPA strings.

    Splits an IPA string (e.g. "pʰaː˥") into a list of vocab IDs.
    Unknown chars (e.g. editor annotations) map to UNK_ID.
    """
    out: list[int] = []
    i = 0
    n = len(ipa)
    while i < n:
        matched = False
        for tok in _IPA_TOKENS_BY_LEN:
            if tok and ipa.startswith(tok, i):
                out.append(_OUTPUT_TOKEN_TO_ID.get(tok, UNK_ID))
                i += len(tok)
                matched = True
                break
        if not matched:
            # Single unknown char — skip one position so we make progress.
            out.append(UNK_ID)
            i += 1
    return out


def detokenize_ipa(ids: list[int]) -> str:
    """Inverse of tokenize_ipa. Skips PAD/BOS/EOS/UNK silently."""
    out: list[str] = []
    for i in ids:
        if i in (PAD_ID, BOS_ID, EOS_ID, UNK_ID):
            continue
        if 0 <= i < len(OUTPUT_VOCAB):
            out.append(OUTPUT_VOCAB[i])
    return "".join(out)
