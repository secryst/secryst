"""ByT5-based Thai→IPA G2P — pretrained seq2seq fine-tuning.

ByT5-small (300M) is pretrained on mC4 at the UTF-8 byte level, giving it
knowledge of Thai character patterns. Fine-tuning on the Kaikki Thai-IPA
corpus should reach competitive PER, and ByT5's autoregressive decoder
avoids the mode collapse that plagues from-scratch seq2seq models.

This is the same approach that worked for Hebrew diacritization (ByT5-small
17% DER, ByT5-Base in progress). Thai G2P has the same fundamental problem:
from-scratch models can't compete with pretrained byte-level models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset


class ByT5ThaiDataset(Dataset):
    """Kaikki Thai-IPA pairs tokenized with ByT5: Thai text → IPA."""

    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        max_len: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        for line in Path(data_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = (row.get("src") or "").strip()
            tgt = (row.get("tgt") or "").strip()
            if not src or not tgt:
                continue
            if len(src.encode("utf-8")) > max_len or len(tgt.encode("utf-8")) > max_len:
                continue
            self.examples.append((src, tgt))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        src, tgt = self.examples[idx]
        model_inputs = self.tokenizer(src, truncation=True, max_length=self.max_len)
        labels = self.tokenizer(tgt, truncation=True, max_length=self.max_len)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def build_byt5_model(model_name: str = "google/byt5-small"):
    """Load pretrained seq2seq model. Works with ByT5, mT5, umt5."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer


def train_byt5(
    cfg: dict[str, Any],
    train_path: Path,
    val_path: Path,
    ckpt_root: Path,
    metrics_path: Path | None = None,
) -> str:
    """Fine-tune ByT5 on Thai→IPA G2P using Seq2SeqTrainer."""
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
    )

    m_cfg = cfg.get("model", {})
    t_cfg = cfg.get("train", {})
    model_name = m_cfg.get("model_name", "google/byt5-small")
    max_len = int(m_cfg.get("max_len", 256))

    model, tokenizer = build_byt5_model(model_name)

    train_ds = ByT5ThaiDataset(train_path, tokenizer, max_len=max_len)
    val_ds = ByT5ThaiDataset(val_path, tokenizer, max_len=max_len)
    print(f"[byt5] train={len(train_ds)}, val={len(val_ds)}", flush=True)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    epochs = int(t_cfg.get("epochs", 20))
    batch_size = int(t_cfg.get("batch_size", 16))
    lr = float(t_cfg.get("learning_rate", 3e-4))
    warmup = int(t_cfg.get("warmup_steps", 200))
    weight_decay = float(t_cfg.get("weight_decay", 0.01))
    grad_clip = float(t_cfg.get("grad_clip", 1.0))
    label_smoothing = float(t_cfg.get("label_smoothing", 0.1))
    seed = int(t_cfg.get("seed", 42))

    args = Seq2SeqTrainingArguments(
        output_dir=str(ckpt_root),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=warmup,
        weight_decay=weight_decay,
        max_grad_norm=grad_clip,
        label_smoothing_factor=label_smoothing,
        seed=seed,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        predict_with_generate=False,
        logging_steps=50,
        report_to=[],
        dataloader_num_workers=4,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    best_path = ckpt_root / "best"
    trainer.save_model(str(best_path))
    tokenizer.save_pretrained(str(best_path))

    if metrics_path:
        log_history = trainer.state.log_history
        with open(metrics_path, "w", encoding="utf-8") as f:
            for entry in log_history:
                f.write(json.dumps(entry) + "\n")

    return str(best_path)


def _edit_distance(a: list, b: list) -> int:
    """Levenshtein distance for PER computation."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


def evaluate_byt5(
    model,
    tokenizer,
    test_path: Path,
    device: torch.device,
    max_new_tokens: int = 256,
    num_beams: int = 4,
) -> dict[str, float]:
    """Generate IPA and compute PER."""
    from transformers import DataCollatorForSeq2Seq

    model.eval()
    test_ds = ByT5ThaiDataset(test_path, tokenizer, max_len=256)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)
    total_ed = 0
    total_gold_len = 0
    exact_match = 0
    total_n = 0

    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(test_ds), batch_size):
            batch_items = [test_ds[j] for j in range(i, min(i + batch_size, len(test_ds)))]
            batch = collator(batch_items)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )

            for j, gen_ids in enumerate(generated):
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                gold = test_ds.examples[i + j][1]
                pred_tokens = gen_text.strip().split()
                gold_tokens = gold.strip().split()
                ed = _edit_distance(pred_tokens, gold_tokens)
                total_ed += ed
                total_gold_len += max(1, len(gold_tokens))
                if ed == 0:
                    exact_match += 1
                total_n += 1

    per = total_ed / max(1, total_gold_len)
    return {
        "per": per,
        "wer": 1.0 - (exact_match / max(1, total_n)),
        "exact_match": exact_match / max(1, total_n),
        "n_examples": total_n,
    }
