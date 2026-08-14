"""Modern Transformer blocks — K3/DS4 stack for Thai→IPA G2P.

Architecture choices drawn from current frontier papers (mirrors
rababa's modern.py):

  - **RoPE** (Su et al., 2021) — DS4 + K3 positional encoding
  - **SDPA** — PyTorch 2.x scaled_dot_product_attention (Flash)
  - **mHC** (Manifold-Constrained Hyper-Connections) — DeepSeek V4 (arXiv:2512.24880)
  - **AttnRes** (Attention Residuals) — Kimi K3 (arXiv:2607.24653)
  - **RMSNorm** — DS4 + K3 default
  - **SwiGLU FFN** — Llama / DS / Kimi default

Decoder adds a **cross-attention** block (no RoPE on K/V from the
encoder — RoPE is position-aware and the encoder already encoded
positions). The cross-attn also gets an mHC residual mix and an
AttnRes-style pass-through (cross-attn output flows to the next
layer).

The optimizer side (Muon + QK-Clip) lives in `training/optim.py`,
wired in via the supervised loop.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


# ---- Rotary positional embedding --------------------------------------


class RotaryEmbedding(nn.Module):
    """Pre-computes cos/sin tables for RoPE."""

    def __init__(self, head_dim: int, max_len: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = freqs.repeat_interleave(2, dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q_r = q * cos + _rotate_half(q) * sin
    k_r = k * cos + _rotate_half(k) * sin
    return q_r, k_r


# ---- RMSNorm (DS4 + K3) -----------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# ---- Sinkhorn-Knopp projection for mHC --------------------------------


def sinkhorn_knopp(mat: torch.Tensor, iters: int = 20) -> torch.Tensor:
    """Project mat onto the Birkhoff polytope (doubly-stochastic).

    Numerically safe: uses log-domain Sinkhorn to avoid division by
    near-zero row/column sums. The log-domain version sums log-probabilities
    instead of dividing, which is stable for any input scale.

    For a 2×2 matrix this is overkill in theory, but the iterated division
    version produces -inf/NaN when raw entries happen to have small column
    sums (which CAN occur with our `eye(2) + 0.01 * randn` init). Log-domain
    avoids that failure mode entirely.
    """
    # Shift to non-negative (Sinkhorn needs non-negative input).
    # We use abs() — for our near-identity init this is a no-op on diagonal.
    log_m = torch.log(mat.abs().clamp_min(1e-8))
    for _ in range(iters):
        # Row normalize in log domain: subtract log-sum-exp along rows.
        log_m = log_m - torch.logsumexp(log_m, dim=-1, keepdim=True)
        # Column normalize in log domain.
        log_m = log_m - torch.logsumexp(log_m, dim=-2, keepdim=True)
    return torch.exp(log_m)


# ---- Manifold-Constrained Hyper-Connections --------------------------


class MHC(nn.Module):
    """mHC residual: M @ [x, sublayer_out], where M is SK-normalized."""

    def __init__(self, sk_iters: int = 20) -> None:
        super().__init__()
        self.sk_iters = sk_iters
        raw = torch.eye(2) + 0.01 * torch.randn(2, 2)
        self.mix_raw = nn.Parameter(raw)

    def forward(self, x: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        streams = torch.stack((x, sublayer_out), dim=2)
        m = sinkhorn_knopp(self.mix_raw, self.sk_iters)
        mixed = torch.einsum("ij,btid->btjd", m, streams)
        return mixed[:, :, 0, :]


class MHCN(nn.Module):
    """N-stream Manifold-Constrained Hyper-Connections (DS4-style).

    See rababa/models/modern.py for the full docstring. Drop-in
    generalization of `MHC` from 2 streams to N streams.
    """

    def __init__(self, n_streams: int, sk_iters: int = 20) -> None:
        super().__init__()
        assert n_streams >= 2, f"n_streams must be ≥ 2, got {n_streams}"
        self.n_streams = n_streams
        self.sk_iters = sk_iters
        raw = torch.eye(n_streams) + 0.01 * torch.randn(n_streams, n_streams)
        self.mix_raw = nn.Parameter(raw)

    def forward(self, *streams: torch.Tensor) -> torch.Tensor:
        assert len(streams) == self.n_streams, (
            f"MHCN expected {self.n_streams} streams, got {len(streams)}"
        )
        stacked = torch.stack(streams, dim=2)
        m = sinkhorn_knopp(self.mix_raw, self.sk_iters)
        mixed = torch.einsum("ij,btid->btjd", m, stacked)
        return mixed[:, :, 0, :]
        return mixed[:, :, 0, :]


# ---- Encoder layer ----------------------------------------------------


def _init_linear(ln: nn.Linear, init_scale: float = 0.02) -> None:
    """Small normal init for linear weights — standard Transformer recipe.

    Default PyTorch init is U(-sqrt(k), sqrt(k)) where k=1/fan_in. That
    works for ReLU MLPs but produces logits that are too large for
    Transformer training from scratch (loss = 247 at init, etc.).
    """
    nn.init.normal_(ln.weight, mean=0.0, std=init_scale)
    if ln.bias is not None:
        nn.init.zeros_(ln.bias)


def _init_embedding(emb: nn.Embedding, init_scale: float = 0.02) -> None:
    nn.init.normal_(emb.weight, mean=0.0, std=init_scale)
    if emb.padding_idx is not None:
        nn.init.zeros_(emb.weight[emb.padding_idx])


class ModernEncoderLayer(nn.Module):
    """Pre-norm encoder layer: SwiGLU FFN + mHC residuals + AttnRes.

    Optional ResFormer (arXiv:2410.17897, ACL 2025) value residual:
    V_n = λ_1·V_1 + λ_2·V_n. The first layer's V is cached on the module
    via `_v_first` and threaded to subsequent layers by ModernEncoder.forward.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        sk_iters: int = 20,
        init_scale: float = 0.02,
        resformer_lambda1: float | None = None,
        resformer_lambda2: float | None = None,
    ) -> None:
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.dim = dim

        self.norm1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.norm2 = RMSNorm(dim)
        self.w_gate = nn.Linear(dim, ff_dim, bias=False)
        self.w_up = nn.Linear(dim, ff_dim, bias=False)
        self.w_down = nn.Linear(ff_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.mhc_attn = MHC(sk_iters=sk_iters)
        self.mhc_ff = MHC(sk_iters=sk_iters)

        _init_linear(self.qkv, init_scale)
        _init_linear(self.out_proj, init_scale)
        _init_linear(self.w_gate, init_scale)
        _init_linear(self.w_up, init_scale)
        _init_linear(self.w_down, init_scale)

        # ResFormer value residual (arXiv:2410.17897).
        self.use_resformer = resformer_lambda1 is not None
        if self.use_resformer:
            lam1 = float(resformer_lambda1)  # type: ignore[arg-type]
            lam2 = float(resformer_lambda2) if resformer_lambda2 is not None else 0.5
            self.resformer_lambda1 = nn.Parameter(torch.tensor(lam1))
            self.resformer_lambda2 = nn.Parameter(torch.tensor(lam2))

    def _attention(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                   key_padding_mask: torch.Tensor | None,
                   v1: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Cache V_1 from layer 0 for downstream ResFormer layers.
        v_pre_residual = v
        if self.use_resformer and v1 is not None:
            v = self.resformer_lambda1 * v1 + self.resformer_lambda2 * v
        if v1 is None:
            self._v_first = v_pre_residual
        q, k = apply_rope(q, k, cos, sin)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :].to(torch.bool)
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0 if not self.training else self.dropout.p
        )
        attn = attn.transpose(1, 2).reshape(B, T, self.dim)
        return self.out_proj(attn)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        prev_attn: torch.Tensor | None = None,
        v1: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out = self._attention(self.norm1(x), cos, sin, key_padding_mask, v1=v1)
        if prev_attn is not None:
            attn_out = attn_out + prev_attn
        x = self.mhc_attn(x, attn_out)
        ff_out = self._ffn(self.norm2(x))
        x = self.mhc_ff(x, ff_out)
        return x, attn_out


# ---- Decoder layer (with cross-attention) ----------------------------


class ModernDecoderLayer(nn.Module):
    """Decoder block: causal self-attn + cross-attn + SwiGLU FFN.

    All three sublayers get mHC residuals. AttnRes flows within the
    self-attn stream; cross-attn output is added directly (no AttnRes
    flow because the cross-attn source is the encoder, not the previous
    decoder layer).
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        sk_iters: int = 20,
        init_scale: float = 0.02,
        use_mhc_cross: bool = True,
    ) -> None:
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.dim = dim
        self.use_mhc_cross = use_mhc_cross

        # Self-attention (causal)
        self.norm1 = RMSNorm(dim)
        self.qkv_self = nn.Linear(dim, 3 * dim, bias=False)
        self.out_self = nn.Linear(dim, dim, bias=False)

        # Cross-attention (no RoPE on K/V — they come from the encoder)
        self.norm2 = RMSNorm(dim)
        self.q_cross = nn.Linear(dim, dim, bias=False)
        self.kv_cross = nn.Linear(dim, 2 * dim, bias=False)
        self.out_cross = nn.Linear(dim, dim, bias=False)

        # FFN
        self.norm3 = RMSNorm(dim)
        self.w_gate = nn.Linear(dim, ff_dim, bias=False)
        self.w_up = nn.Linear(dim, ff_dim, bias=False)
        self.w_down = nn.Linear(ff_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.mhc_self = MHC(sk_iters=sk_iters)
        # Cross-attn: standard residual when use_mhc_cross=False. This forces
        # the cross-attn output to contribute to x directly (`x = x + cross_attn(x)`),
        # bypassing the SK-normalized mix that can collapse to identity in
        # seq2seq mode-collapse scenarios.
        self.mhc_cross = MHC(sk_iters=sk_iters) if use_mhc_cross else None
        self.mhc_ff = MHC(sk_iters=sk_iters)

        for ln in (self.qkv_self, self.out_self, self.q_cross, self.kv_cross,
                   self.out_cross, self.w_gate, self.w_up, self.w_down):
            _init_linear(ln, init_scale)

    def _self_attention(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv_self(x).reshape(B, T, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        # Build combined causal + padding mask. SDPA rejects combining
        # is_causal=True with explicit attn_mask, so we construct a full
        # (T, T) mask: lower-triangular (causal) ANDed with inverted padding.
        device = x.device
        causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        attn_mask = causal[None, None, :, :]  # (1, 1, T, T) — broadcastable
        if key_padding_mask is not None:
            kpm = key_padding_mask.to(torch.bool)[:, None, None, :]  # (B, 1, 1, T)
            attn_mask = attn_mask & ~kpm
        # SDPA expects float mask with -inf for masked positions (or bool).
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=0.0 if not self.training else self.dropout.p,
        )
        attn = attn.transpose(1, 2).reshape(B, T, self.dim)
        return self.out_self(attn)

    def _cross_attention(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        T_mem = memory.size(1)
        q = self.q_cross(x).reshape(B, T, self.heads, self.head_dim).transpose(1, 2)
        kv = self.kv_cross(memory).reshape(B, T_mem, 2, self.heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # No RoPE on K — encoder already encoded its positions.
        attn_mask = None
        if memory_key_padding_mask is not None:
            attn_mask = memory_key_padding_mask[:, None, None, :].to(torch.bool)
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0 if not self.training else self.dropout.p,
        )
        attn = attn.transpose(1, 2).reshape(B, T, self.dim)
        return self.out_cross(attn)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor | None,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None,
        prev_self_attn: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self-attn (causal) with AttnRes pass-through.
        self_attn_out = self._self_attention(self.norm1(x), cos, sin, tgt_key_padding_mask)
        if prev_self_attn is not None:
            self_attn_out = self_attn_out + prev_self_attn
        x = self.mhc_self(x, self_attn_out)
        # Cross-attn to encoder memory.
        cross_out = self._cross_attention(self.norm2(x), memory, memory_key_padding_mask)
        if self.mhc_cross is not None:
            x = self.mhc_cross(x, cross_out)
        else:
            x = x + cross_out
        # FFN.
        ff_out = self._ffn(self.norm3(x))
        x = self.mhc_ff(x, ff_out)
        return x, self_attn_out


# ---- Encoder body ------------------------------------------------------


class ModernEncoder(nn.Module):
    """Stack of ModernEncoderLayer with embedding + final norm.

    Optional ResFormer (arXiv:2410.17897) value residual via `resformer` dict:
      - {"mode": "all"}: every layer ≥1 gets V_1 with init λ_1=λ_2=0.5.
      - {"mode": "sparse", "n_last_layers": K, "lambda1": λ}: only last K
        layers receive V_1 with init λ_1=λ (paper recipe: λ=5.0).
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        layers: int,
        heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        max_len: int = 128,
        pad_id: int = 0,
        rope_base: float = 10000.0,
        sk_iters: int = 20,
        resformer: dict | None = None,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.dim = dim
        self.max_len = max_len
        self.head_dim = dim // heads

        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.rotary = RotaryEmbedding(self.head_dim, max_len=max_len, base=rope_base)
        resformer_mode = (resformer or {}).get("mode", "off")
        resformer_n = int((resformer or {}).get("n_last_layers", max(1, layers // 3)))
        resformer_lambda1 = (resformer or {}).get("lambda1", 0.5)
        resformer_lambda2 = (resformer or {}).get("lambda2", 0.5)
        self.layers = nn.ModuleList()
        for i in range(layers):
            if resformer_mode == "all":
                lam1 = resformer_lambda1 if i >= 1 else None
                lam2 = resformer_lambda2
            elif resformer_mode == "sparse":
                is_sparse_layer = i >= (layers - resformer_n) and i >= 1
                lam1 = resformer_lambda1 if is_sparse_layer else None
                lam2 = resformer_lambda2
            else:
                lam1 = None
                lam2 = None
            self.layers.append(
                ModernEncoderLayer(
                    dim, heads, ff_dim,
                    dropout=dropout, sk_iters=sk_iters,
                    resformer_lambda1=lam1,
                    resformer_lambda2=lam2,
                )
            )
        self.final_norm = RMSNorm(dim)
        _init_embedding(self.embedding)

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (hidden_states, key_padding_mask)."""
        B, T = src.shape
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} exceeds max_len {self.max_len}")
        key_padding_mask = src == self.pad_id
        x = self.embedding(src)
        cos, sin = self.rotary(T)
        prev_attn: torch.Tensor | None = None
        v1: torch.Tensor | None = None
        for i, layer in enumerate(self.layers):
            x, prev_attn = layer(x, cos, sin, key_padding_mask, prev_attn, v1=v1)
            if i == 0 and hasattr(layer, "_v_first"):
                v1 = layer._v_first
        return self.final_norm(x), key_padding_mask


# ---- Decoder body ------------------------------------------------------


class ModernDecoder(nn.Module):
    """Stack of ModernDecoderLayer with embedding + final norm + LM head."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        layers: int,
        heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        max_len: int = 128,
        pad_id: int = 0,
        rope_base: float = 10000.0,
        sk_iters: int = 20,
        use_mhc_cross: bool = True,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.dim = dim
        self.max_len = max_len
        self.head_dim = dim // heads

        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.rotary = RotaryEmbedding(self.head_dim, max_len=max_len, base=rope_base)
        self.layers = nn.ModuleList([
            ModernDecoderLayer(
                dim, heads, ff_dim,
                dropout=dropout, sk_iters=sk_iters,
                use_mhc_cross=use_mhc_cross,
            )
            for _ in range(layers)
        ])
        self.final_norm = RMSNorm(dim)
        # Tied output projection — same weights as input embedding.
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        _init_embedding(self.embedding)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return logits over output vocab at each position."""
        B, T = tgt.shape
        if T > self.max_len:
            raise ValueError(f"Decoder seq length {T} exceeds max_len {self.max_len}")
        tgt_key_padding_mask = tgt == self.pad_id
        x = self.embedding(tgt)
        cos, sin = self.rotary(T)
        prev_self_attn: torch.Tensor | None = None
        for layer in self.layers:
            x, prev_self_attn = layer(
                x, cos, sin, tgt_key_padding_mask, memory, memory_key_padding_mask, prev_self_attn,
            )
        x = self.final_norm(x)
        return self.lm_head(x)


# ---- Encoder-only model (for MLM pretraining) -------------------------


class ModernEncoderOnly(nn.Module):
    """Encoder body with no decoder — used for MLM pretraining.

    Pretraining objective: masked-LM on Thai text. The encoder learns
    Thai orthography + syllable structure without needing IPA pairs.
    Encoder weights transfer to the seq2seq encoder for fine-tune.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        layers: int,
        heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        max_len: int = 128,
        pad_id: int = 0,
        rope_base: float = 10000.0,
        sk_iters: int = 20,
        resformer: dict | None = None,
    ) -> None:
        super().__init__()
        self.encoder = ModernEncoder(
            vocab_size, dim, layers, heads, ff_dim,
            dropout=dropout, max_len=max_len, pad_id=pad_id,
            rope_base=rope_base, sk_iters=sk_iters,
            resformer=resformer,
        )
        dim_actual = self.encoder.dim
        # Small MLM head: dense + GELU + RMSNorm + tied decoder + bias.
        self.head_dense = nn.Linear(dim_actual, dim_actual)
        self.head_act = nn.GELU()
        self.head_norm = RMSNorm(dim_actual)
        self.head_decoder = nn.Linear(dim_actual, vocab_size, bias=False)
        self.head_decoder.weight = self.encoder.embedding.weight
        self.head_bias = nn.Parameter(torch.zeros(vocab_size))
        _init_linear(self.head_dense)

    def forward_encoder(self, src: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.encoder(src)
        return hidden

    def forward(self, src: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden = self.forward_encoder(src)
        x = self.head_norm(self.head_act(self.head_dense(hidden)))
        return self.head_decoder(x) + self.head_bias
