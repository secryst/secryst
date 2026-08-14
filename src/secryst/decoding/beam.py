"""Beam search decoder for autoregressive IPA generation.

Greedy + beam search; both call the seq2seq model step-by-step. The
model's `forward(src, tgt_in)` produces logits at every decoder
position; for inference we re-run with the growing prefix.

Implementation notes:
  - We use KV-cache-free decoding (simpler; the encoder is computed
    once and re-used). For long outputs a KV cache would speed things
    up but for Thai words (avg 5–15 IPA tokens) the overhead is small.
  - Length penalty: standard `((5 + len) / (5 + 1)) ** alpha` to
    discourage very short sequences.
  - Early stop: a beam is "finished" once it emits EOS. We keep
    finished beams in the candidate pool until their score is below
    the worst live beam's extended score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..constants import BOS_ID, EOS_ID, PAD_ID
from ..constants import detokenize_ipa


@dataclass
class BeamHypothesis:
    tokens: list[int]
    score: float
    finished: bool


def _length_penalty(length: int, alpha: float = 0.6) -> float:
    return ((5.0 + length) / 6.0) ** alpha


@torch.no_grad()
def beam_search(
    model: torch.nn.Module,
    src: torch.Tensor,
    src_lengths: torch.Tensor,
    max_len: int = 64,
    beam_width: int = 4,
    length_penalty_alpha: float = 0.6,
    eos_bias: float = 0.0,
    device: torch.device | None = None,
) -> list[list[int]]:
    """Run beam search for a batch of source sequences.

    Returns a list (len == batch_size) of token ID lists (the best beam
    per source, excluding BOS/EOS).

    Args:
        model: a ModernSeq2Seq.
        src: (B, T_src) input token IDs.
        src_lengths: (B,) true source lengths.
        max_len: maximum output length (excluding BOS).
        beam_width: beam size.
        length_penalty_alpha: 0 disables penalty; 0.6–1.0 typical.
        eos_bias: optional additive boost to the EOS logit at every step.
            Useful when the model under-predicts EOS (a common seq2seq
            failure mode that yields PER > 1.0 from runaway generation).
            Typical: 2.0–5.0. Default 0.0 (no boost).
    """
    if device is None:
        device = src.device
    model.eval()
    B = src.size(0)

    # Encode source ONCE.
    memory, src_kpm = model.encode(src)
    # Replicate memory per beam for batched decoding.
    # memory: (B, T_src, D) → (B*W, T_src, D)
    memory_exp = memory.unsqueeze(1).expand(-1, beam_width, -1, -1).reshape(B * beam_width, *memory.shape[1:])
    src_kpm_exp = src_kpm.unsqueeze(1).expand(-1, beam_width, -1).reshape(B * beam_width, -1)

    # Initialize beam: every beam starts with BOS, score 0.
    beams = [[BeamHypothesis(tokens=[BOS_ID], score=0.0, finished=False)] for _ in range(B)]
    # We keep the top-W beams per example. Initial: all are [BOS] with 0.
    for b in range(B):
        beams[b] = [BeamHypothesis(tokens=[BOS_ID], score=0.0, finished=False)] * beam_width

    for step in range(max_len):
        # Build batched decoder input: (B*W, step+1) — current beam tokens.
        all_beams: list[BeamHypothesis] = []
        # Flatten for one big batched call.
        flat_tgt = []
        for b in range(B):
            for w in range(beam_width):
                if w < len(beams[b]):
                    flat_tgt.append(beams[b][w].tokens)
                else:
                    flat_tgt.append([BOS_ID] + [PAD_ID] * step)
        tgt_in = torch.full((B * beam_width, step + 1), PAD_ID, dtype=torch.long, device=device)
        for i, toks in enumerate(flat_tgt):
            tgt_in[i, : len(toks)] = torch.tensor(toks, dtype=torch.long, device=device)

        logits = model.decoder(tgt_in, memory_exp, src_kpm_exp)  # (B*W, step+1, V)
        last_logits = logits[:, -1, :]  # (B*W, V)

        # Optional EOS boost: bump EOS logit to make stopping more likely.
        if eos_bias != 0.0:
            last_logits = last_logits.clone()
            last_logits[:, EOS_ID] = last_logits[:, EOS_ID] + eos_bias

        log_probs = F.log_softmax(last_logits, dim=-1)

        # Per example: expand each beam, pick top-W.
        new_beams: list[list[BeamHypothesis]] = [[] for _ in range(B)]
        for b in range(B):
            candidates: list[BeamHypothesis] = []
            for w in range(beam_width):
                beam_idx = b * beam_width + w
                cur_beam = beams[b][w]
                if cur_beam.finished:
                    # Carry forward finished beams unchanged.
                    candidates.append(cur_beam)
                    continue
                topk_lp, topk_ids = log_probs[beam_idx].topk(beam_width)
                for k in range(beam_width):
                    tok = int(topk_ids[k].item())
                    score = cur_beam.score + float(topk_lp[k].item())
                    finished = tok == EOS_ID
                    new_tokens = cur_beam.tokens + ([] if finished else [tok])
                    candidates.append(BeamHypothesis(
                        tokens=new_tokens,
                        score=score,
                        finished=finished,
                    ))
            # Pick top-W by length-penalized score; finished beams use their
            # final score with the actual length.
            candidates.sort(
                key=lambda h: h.score / _length_penalty(max(1, len(h.tokens) - 1), length_penalty_alpha),
                reverse=True,
            )
            new_beams[b] = candidates[:beam_width]
        beams = new_beams

        # Early stop if every beam in every example is finished.
        if all(all(beam.finished for beam in beams[b]) for b in range(B)):
            break

    # Return best beam per example (excluding BOS).
    out: list[list[int]] = []
    for b in range(B):
        best = max(beams[b], key=lambda h: h.score / _length_penalty(max(1, len(h.tokens) - 1), length_penalty_alpha))
        out.append([t for t in best.tokens if t not in (BOS_ID, EOS_ID, PAD_ID)])
    return out


@torch.no_grad()
def greedy_decode(
    model: torch.nn.Module,
    src: torch.Tensor,
    src_lengths: torch.Tensor,
    max_len: int = 64,
    device: torch.device | None = None,
) -> list[list[int]]:
    """Greedy decoding — equivalent to beam_search with beam_width=1."""
    return beam_search(model, src, src_lengths, max_len=max_len, beam_width=1, device=device)


def decode_to_string(model: torch.nn.Module, src_text: str, encoder, max_len: int = 64, beam_width: int = 4, device: torch.device | None = None) -> str:
    """Convenience: encode a single Thai string → IPA string.

    Args:
        model: trained ModernSeq2Seq.
        src_text: Thai input string.
        encoder: a ThaiEncoder instance.
        max_len: max output length.
        beam_width: 1 for greedy, >1 for beam search.
        device: where to run inference.
    """
    if device is None:
        device = next(model.parameters()).device
    cleaned = encoder.clean(src_text)
    ids = encoder.encode(cleaned)
    src = torch.tensor([ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(ids)], dtype=torch.long, device=device)
    tokens = beam_search(model, src, lengths, max_len=max_len, beam_width=beam_width, device=device)
    return detokenize_ipa(tokens[0])
