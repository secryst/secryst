"""Datasets — Wiktionary Thai-IPA pairs + MLM pretrain corpus.

Two datasets:
  1. `ThaiIPADataset` — supervised pairs (Thai word → IPA token IDs).
     Source: `kaikki.org/dictionary/Thai/kaikki.org-dictionary-Thai.jsonl`
     (downloaded by `modal_app.py::fetch_data`). One JSON entry per
     line; we filter to entries with exactly one IPA transcription.

  2. `ThaiMLMDataset` — undiacritized Thai text for MLM pretraining.
     Same source (Thai headwords + example sentences), used to teach
     the encoder Thai orthography before supervised fine-tune.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset

from .constants import BOS_ID, EOS_ID, PAD_ID, UNK_ID
from .encoder import ThaiEncoder
from .constants import tokenize_ipa


@dataclass(frozen=True)
class IPAPair:
    """One parallel (Thai, IPA) pair after encoding."""
    src_ids: list[int]   # input token IDs
    tgt_ids: list[int]   # output token IDs (no BOS/EOS — added at collate time)
    raw_src: str
    raw_tgt: str


@dataclass(frozen=True)
class MLMExample:
    """One MLM training example: masked input + per-position targets."""
    input_ids: list[int]
    target_ids: list[int]
    raw: str


def _valid_ipa(ipa: str) -> bool:
    """Quick filter — drop entries with editor annotations / non-IPA chars."""
    if not ipa or "/" not in ipa and not any(c in ipa for c in "ptkbmŋnaeiouˈˌː"):
        return False
    # Reject entries with bracketed editor notes like [/r], [n.], etc.
    for ch in ("[", "]", "{", "}", "<", ">"):
        if ch in ipa:
            return False
    return True


def _clean_ipa_string(raw: str) -> str | None:
    """Extract a single IPA transcription from a Wiktionary `ipa` field.

    The Kaikki schema stores IPA as a list of strings (often with
    surrounding slashes, editor annotations, and qualifier suffixes
    like "/pʰaː˥/"). We:
      - take the first valid entry
      - strip surrounding slashes
      - reject if it contains editor annotations
      - reject if length is unreasonable (too short or suspiciously long)
    """
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("/") and s.endswith("/"):
        s = s[1:-1].strip()
    elif s.startswith("/") and "/" in s[1:]:
        s = s[1:s.index("/", 1)].strip()
    if not _valid_ipa(s):
        return None
    if len(s) < 1 or len(s) > 64:
        return None
    return s


def load_kaikki_pairs(
    path: Path,
    encoder: ThaiEncoder,
    max_pairs: int | None = None,
) -> list[IPAPair]:
    """Parse a Kaikki Thai JSONL dump into IPA pairs.

    Each JSONL line has the structure (simplified):
        {"word": "ภาษา", "sounds": [{"ipa": "pʰaː˥"}, ...], ...}

    We pick the first valid IPA per entry; one (word, ipa) pair per word.
    """
    pairs: list[IPAPair] = []
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                entry = json.loads(ln)
            except json.JSONDecodeError:
                continue
            word = (entry.get("word") or "").strip()
            if not word:
                continue
            sounds = entry.get("sounds") or []
            ipa_str: str | None = None
            for s in sounds:
                if not isinstance(s, dict):
                    continue
                raw_ipa = s.get("ipa")
                if isinstance(raw_ipa, list):
                    for ip in raw_ipa:
                        cleaned = _clean_ipa_string(str(ip))
                        if cleaned:
                            ipa_str = cleaned
                            break
                elif isinstance(raw_ipa, str):
                    cleaned = _clean_ipa_string(raw_ipa)
                    if cleaned:
                        ipa_str = cleaned
                if ipa_str:
                    break
            if not ipa_str:
                continue
            key = (word, ipa_str)
            if key in seen:
                continue
            seen.add(key)
            cleaned_word = encoder.clean(word)
            if not cleaned_word:
                continue
            src_ids = encoder.encode(cleaned_word)
            tgt_ids = tokenize_ipa(ipa_str)
            if not src_ids or not tgt_ids:
                continue
            # Filter out pairs with too many UNKs in target (corrupt IPA).
            unk_ratio = sum(1 for t in tgt_ids if t == UNK_ID) / max(1, len(tgt_ids))
            if unk_ratio > 0.2:
                continue
            pairs.append(IPAPair(src_ids=src_ids, tgt_ids=tgt_ids, raw_src=word, raw_tgt=ipa_str))
            if max_pairs is not None and len(pairs) >= max_pairs:
                break
    return pairs


# ---- Supervised dataset -----------------------------------------------


class ThaiIPADataset(Dataset):
    """Supervised (Thai → IPA) dataset loaded from a pre-split file.

    File format: JSONL of {"src": "ภาษา", "tgt": "pʰaː˥"} pairs. The
    `modal_app.fetch_data` stage produces this from the raw Kaikki dump.
    """

    def __init__(self, split: str, root: Path | None = None, cleaner: str = "thai", max_len: int = 128) -> None:
        self.root = Path(root) if root else DEFAULT_IPA_ROOT
        path = self.root / f"{split}.jsonl"
        if not path.is_file():
            # Fall back to TXT format: "src<TAB>tgt" per line.
            txt_path = self.root / f"{split}.txt"
            if txt_path.is_file():
                self.examples = self._load_txt(txt_path, cleaner, max_len)
            else:
                raise FileNotFoundError(f"Neither {path} nor {txt_path} found")
        else:
            self.examples = self._load_jsonl(path, cleaner, max_len)
        self.split = split

    @staticmethod
    def _load_jsonl(path: Path, cleaner: str, max_len: int) -> list[IPAPair]:
        enc = ThaiEncoder(cleaner=cleaner)
        out: list[IPAPair] = []
        for ln in path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                continue
            src = (row.get("src") or "").strip()
            tgt = (row.get("tgt") or "").strip()
            if not src or not tgt:
                continue
            cleaned = enc.clean(src)
            if not cleaned:
                continue
            src_ids = enc.encode(cleaned)[:max_len]
            tgt_ids = tokenize_ipa(tgt)
            if not src_ids or not tgt_ids:
                continue
            out.append(IPAPair(src_ids=src_ids, tgt_ids=tgt_ids, raw_src=src, raw_tgt=tgt))
        return out

    @staticmethod
    def _load_txt(path: Path, cleaner: str, max_len: int) -> list[IPAPair]:
        enc = ThaiEncoder(cleaner=cleaner)
        out: list[IPAPair] = []
        for ln in path.read_text(encoding="utf-8").splitlines():
            ln = ln.rstrip("\n")
            if "\t" in ln:
                src, tgt = ln.split("\t", 1)
            elif "," in ln:
                src, tgt = ln.split(",", 1)
            else:
                continue
            src = src.strip()
            tgt = tgt.strip()
            if not src or not tgt:
                continue
            cleaned = enc.clean(src)
            if not cleaned:
                continue
            src_ids = enc.encode(cleaned)[:max_len]
            tgt_ids = tokenize_ipa(tgt)
            if not src_ids or not tgt_ids:
                continue
            out.append(IPAPair(src_ids=src_ids, tgt_ids=tgt_ids, raw_src=src, raw_tgt=tgt))
        return out

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> IPAPair:
        return self.examples[idx]


def load_thai_ipa(
    split: str,
    root: Path | None = None,
    cleaner: str = "thai",
    max_len: int = 128,
) -> ThaiIPADataset:
    return ThaiIPADataset(split=split, root=root, cleaner=cleaner, max_len=max_len)


# ---- MLM pretraining --------------------------------------------------


def _apply_bert_mask(
    input_ids: list[int],
    mask_prob: float,
    rng: random.Random,
    vocab_size: int,
    mask_id: int,
) -> tuple[list[int], list[int]]:
    """BERT-style masking. 80/10/10 split on selected positions."""
    n = len(input_ids)
    masked = list(input_ids)
    target = [PAD_ID] * n
    for i, original in enumerate(input_ids):
        if original == PAD_ID:
            continue
        if rng.random() >= mask_prob:
            continue
        target[i] = original
        r = rng.random()
        if r < 0.8:
            masked[i] = mask_id
        elif r < 0.9:
            masked[i] = rng.randint(1, vocab_size - 1)
    return masked, target


# MASK_ID = UNK_ID for Thai (the input vocab has no dedicated MASK).
# BERT's [MASK] is rare in real Thai text so using UNK as the mask token
# avoids growing the vocab. The model learns "predict the masked slot"
# from the surrounding context regardless of which symbol fills it.
MASK_ID = UNK_ID


# Default corpus locations — set by fetch_data on Modal; falls back to
# local dev path.
def _find_ipa_root() -> Path:
    candidates = [
        Path("/opt/secryst/data/thai-ipa"),     # Modal image build-time
        Path("/datasets/thai-ipa"),              # Modal volume mount
        Path(__file__).resolve().parent.parent.parent / "data" / "thai-ipa",
    ]
    for c in candidates:
        if (c / "train.jsonl").is_file() or (c / "train.txt").is_file():
            return c
    return candidates[-1]


def _find_mlm_root() -> Path:
    candidates = [
        Path("/opt/secryst/data/thai-text"),    # Modal image build-time
        Path("/datasets/thai-text"),             # Modal volume mount
        Path(__file__).resolve().parent.parent.parent / "data" / "thai-text",
    ]
    for c in candidates:
        if (c / "train.txt").is_file():
            return c
    return candidates[-1]


DEFAULT_IPA_ROOT = _find_ipa_root()
DEFAULT_MLM_ROOT = _find_mlm_root()


class ThaiMLMDataset(Dataset):
    """Undiacritized Thai text for masked-LM pretraining."""

    def __init__(
        self,
        split: str = "train",
        root: Path | None = None,
        cleaner: str = "thai",
        mask_prob: float = 0.15,
        max_len: int = 128,
        seed: int = 42,
    ) -> None:
        self.root = Path(root) if root else DEFAULT_MLM_ROOT
        self.mask_prob = mask_prob
        self.max_len = max_len
        self.base_seed = seed
        self.vocab_size = len(ThaiEncoder(cleaner=cleaner).input_id_to_symbol)
        self.mask_id = MASK_ID
        enc = ThaiEncoder(cleaner=cleaner)
        path = self.root / f"{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Thai MLM corpus {split} not found at {path}")
        self.sequences: list[tuple[list[int], str]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cleaned = enc.clean(line)
            if not cleaned:
                continue
            ids = enc.encode(cleaned)[:max_len]
            if len(ids) < 4:
                continue
            self.sequences.append((ids, line))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> MLMExample:
        ids, raw = self.sequences[idx]
        seed = hash((self.base_seed, idx)) & 0xFFFFFFFF
        rng = random.Random(seed)
        masked, target = _apply_bert_mask(ids, self.mask_prob, rng, self.vocab_size, self.mask_id)
        return MLMExample(input_ids=masked, target_ids=target, raw=raw)


def load_thai_mlm(
    split: str = "train",
    root: Path | None = None,
    cleaner: str = "thai",
    mask_prob: float = 0.15,
    max_len: int = 128,
    seed: int = 42,
) -> ThaiMLMDataset:
    return ThaiMLMDataset(
        split=split, root=root, cleaner=cleaner,
        mask_prob=mask_prob, max_len=max_len, seed=seed,
    )
