"""Phonological features for Thai input.

Computes per-character feature IDs that encode Thai phonological properties:
  - `consonant_class`: 0=low, 1=mid, 2=high, 3=other (determines tone).
  - `has_tone_mark`: 1 if this position is a tone mark, 0 otherwise.
  - `syllable_initial`: 1 if this position likely starts a new syllable.

These features encode Thai's 5-tone system constraints, which the model
would otherwise have to learn implicitly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


# Thai consonant classes (determine tone rules).
# Source: standard Thai phonology references.
LOW_CONSONANTS = set("กจดฎฏดตบปอ")  # Actually mid — see below for correct mapping
MID_CONSONANTS = set("กจดฎฏบปอ")
HIGH_CONSONANTS = set("ขฉถผฝสห")
# Low consonants are the rest of the consonant set.

# Corrected classification (per Thai phonology):
MID_CONSONANTS = set("กจดฎฏบปอ")  # 9 mid-class consonants
HIGH_CONSONANTS = set("ขฉถผฝสห")  # 11 high-class consonants
LOW_CONSONANTS = set("คฆงชซฌญฑฒณทธนพฟภมยรลวฅฆ")  # ~24 low-class consonants

TONE_MARKS = set("่้๊๋")  # mai ek, tho, tri, jattawa
VOWEL_MARKS = set("ะัาำิีึืุู")


@dataclass(frozen=True)
class ThaiCharFeatures:
    consonant_class: int   # 0=low, 1=mid, 2=high, 3=other
    has_tone_mark: int     # 0 or 1
    syllable_initial: int  # 0 or 1 (heuristic)


def compute_thai_features(text: str) -> list[ThaiCharFeatures]:
    """Compute per-char features for a Thai string."""
    out: list[ThaiCharFeatures] = []
    prev_was_vowel = False
    for i, ch in enumerate(text):
        if ch in LOW_CONSONANTS:
            cc = 0
        elif ch in MID_CONSONANTS:
            cc = 1
        elif ch in HIGH_CONSONANTS:
            cc = 2
        else:
            cc = 3
        has_tone = 1 if ch in TONE_MARKS else 0
        # Syllable-initial heuristic: a consonant following a vowel mark
        # likely starts a new syllable.
        is_initial = 1 if (ch in MID_CONSONANTS | HIGH_CONSONANTS | LOW_CONSONANTS and prev_was_vowel) else 0
        if i == 0:
            is_initial = 1
        out.append(ThaiCharFeatures(
            consonant_class=cc,
            has_tone_mark=has_tone,
            syllable_initial=is_initial,
        ))
        prev_was_vowel = ch in VOWEL_MARKS
    return out


CONSONANT_CLASS_VOCAB_SIZE = 4   # 0-3
TONE_MARK_VOCAB_SIZE = 2         # 0, 1
SYLLABLE_INITIAL_VOCAB_SIZE = 2  # 0, 1


def features_to_ids(features: Sequence[ThaiCharFeatures]) -> dict[str, list[int]]:
    return {
        "consonant_class": [f.consonant_class for f in features],
        "has_tone_mark": [f.has_tone_mark for f in features],
        "syllable_initial": [f.syllable_initial for f in features],
    }


FEATURE_VOCAB_SIZES = {
    "consonant_class": CONSONANT_CLASS_VOCAB_SIZE,
    "has_tone_mark": TONE_MARK_VOCAB_SIZE,
    "syllable_initial": SYLLABLE_INITIAL_VOCAB_SIZE,
}
