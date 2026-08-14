"""Engram — DeepSeek V4 episodic memory.

Reference: DeepSeek V4 Engram (arXiv:2601.07372). Maintains a bounded
FIFO+importance buffer of past (hidden_state, label) pairs. At forward
time, the current hidden state queries the buffer via cosine similarity
and retrieves top-K similar past examples. The retrieved hidden states
are concatenated to the current hidden and projected back to dim.

Why this helps: for rare classes (rare haraqat combinations in Arabic,
rare niqqud patterns in Hebrew, rare tone patterns in Thai), the model
sees few examples per epoch. Engram gives those rare examples extra
gradient signal by surfacing similar past contexts.

Memory footprint: capacity=10K examples × dim=256 floats = 10MB on GPU.
Negligible compared to the model itself.

Open/closed: standalone module. Models that want Engram add it as an
optional layer between encoder and head.
"""

from __future__ import annotations

import torch
from torch import nn


class Engram(nn.Module):
    """Episodic memory with cosine-similarity retrieval.

    Args:
        dim: hidden dim of stored vectors.
        capacity: max number of (hidden, label) pairs stored.
        top_k: number of past examples retrieved per query.
        update_every: how often to write to the buffer (in steps). 1 = every step.
    """

    def __init__(
        self,
        dim: int,
        capacity: int = 10000,
        top_k: int = 4,
        update_every: int = 1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.capacity = capacity
        self.top_k = top_k
        self.update_every = update_every
        # Hidden states buffer: (capacity, dim). Initialized lazily on first write.
        self.register_buffer("hidden_buf", torch.zeros(capacity, dim), persistent=False)
        self.register_buffer("label_buf", torch.zeros(capacity, dtype=torch.long), persistent=False)
        self.register_buffer("write_idx", torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer("size", torch.zeros(1, dtype=torch.long), persistent=False)
        # Projection: concat(current_hidden, retrieved_hiddens) → dim.
        self.proj = nn.Linear(dim * (1 + top_k), dim)
        # Gate: learnable scalar for how much to trust retrieved examples.
        self.gate = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

    def write(self, hidden: torch.Tensor, labels: torch.Tensor) -> None:
        """Write (hidden, labels) to the buffer with FIFO replacement.

        Args:
            hidden: (B, T, D) — flatten to (B*T, D) and write the first
                `capacity` available slots.
            labels: (B, T) — per-position labels.
        """
        B, T, D = hidden.shape
        flat_h = hidden.reshape(B * T, D).detach()
        flat_l = labels.reshape(B * T).detach()
        # Drop PAD positions.
        non_pad = flat_l != 0
        flat_h = flat_h[non_pad]
        flat_l = flat_l[non_pad]
        n = flat_h.size(0)
        if n == 0:
            return
        # Write to buffer (FIFO wraparound).
        idx = int(self.write_idx.item())
        for i in range(n):
            slot = (idx + i) % self.capacity
            self.hidden_buf[slot] = flat_h[i]
            self.label_buf[slot] = flat_l[i]
        self.write_idx[0] = (idx + n) % self.capacity
        self.size[0] = min(int(self.size.item()) + n, self.capacity)

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve top-K similar past examples for each query position.

        Args:
            query: (B, T, D)

        Returns: (B, T, K, D) — top-K retrieved hidden states per query.
        """
        if int(self.size.item()) == 0:
            # Buffer empty — return zeros.
            B, T, _ = query.shape
            return torch.zeros(B, T, self.top_k, self.dim, device=query.device, dtype=query.dtype)
        n = int(self.size.item())
        buf = self.hidden_buf[:n]  # (n, D)
        buf_norm = buf / (buf.norm(dim=-1, keepdim=True) + 1e-8)
        # Cosine similarity per query position.
        B, T, D = query.shape
        flat_q = query.reshape(B * T, D)
        q_norm = flat_q / (flat_q.norm(dim=-1, keepdim=True) + 1e-8)
        sims = q_norm @ buf_norm.T  # (B*T, n)
        # Top-K.
        k = min(self.top_k, n)
        _, topk_idx = sims.topk(k, dim=-1)
        retrieved = buf[topk_idx]  # (B*T, K, D)
        # Pad to top_k if k < top_k.
        if k < self.top_k:
            pad = torch.zeros(B * T, self.top_k - k, D, device=query.device, dtype=query.dtype)
            retrieved = torch.cat([retrieved, pad], dim=1)
        return retrieved.reshape(B, T, self.top_k, D)

    def forward(self, hidden: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        """Read (retrieve) + optional write, then project.

        Args:
            hidden: (B, T, D)
            labels: optional (B, T) for write. If None, only retrieve.

        Returns: (B, T, D) — gated mix of original + retrieved.
        """
        retrieved = self.retrieve(hidden)  # (B, T, K, D)
        # Concat along last dim: (B, T, (1+K)*D)
        concat = torch.cat([hidden, retrieved.reshape(*hidden.shape[:2], -1)], dim=-1)
        projected = self.proj(concat)
        # Gate: mix between original hidden and retrieved projection.
        gate = torch.sigmoid(self.gate)
        out = (1 - gate) * hidden + gate * projected
        # Write to buffer if labels provided.
        if labels is not None and self.training:
            self.write(hidden, labels)
        return out
