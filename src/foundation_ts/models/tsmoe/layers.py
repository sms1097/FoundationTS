from __future__ import annotations

import math
import torch

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func
except Exception:
    flash_attn = None

from torch import nn
from torch.nn import functional as F

from foundation_ts.models.tsmoe.stats import MoEStats

torch.set_float32_matmul_precision("high")
try:
    import torch._dynamo as dynamo

    dynamo.config.capture_scalar_outputs = True
except Exception:
    pass


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # Use the padded sequence length as an upper bound to avoid scalar extraction.
    max_seqlen_in_batch = attention_mask.shape[1]
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _get_segment_unpad_data(attention_mask: torch.Tensor, segment_ids: torch.Tensor):
    batch_size, seq_len = attention_mask.shape
    device = attention_mask.device
    indices_list = []
    cu_seqlens = [0]
    for b in range(batch_size):
        mask = attention_mask[b].to(torch.bool)
        if not mask.any():
            continue
        valid_pos = torch.nonzero(mask, as_tuple=False).flatten()
        seg = segment_ids[b][valid_pos]
        if seg.numel() == 0:
            continue
        change = torch.ones(seg.shape, device=device, dtype=torch.bool)
        if seg.numel() > 1:
            change[1:] = seg[1:] != seg[:-1]
        starts = torch.nonzero(change, as_tuple=False).flatten()
        ends = torch.cat([starts[1:], torch.tensor([seg.numel()], device=device)])
        lengths = ends - starts
        for length in lengths.tolist():
            cu_seqlens.append(cu_seqlens[-1] + int(length))
        indices_list.append(valid_pos + b * seq_len)
    if not indices_list:
        indices = torch.empty((0,), device=device, dtype=torch.long)
    else:
        indices = torch.cat(indices_list)
    cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.int32)
    # Use the padded sequence length as an upper bound to avoid scalar extraction.
    return indices, cu_seqlens, seq_len


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.normalized_shape = (hidden_size,)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # weight = self.weight
        # if weight.dtype != hidden_state.dtype:
        #     weight = weight.to(hidden_state.dtype)
        return F.rms_norm(hidden_state, self.normalized_shape, self.weight.to(hidden_state.dtype), self.eps)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            # seq_len=max_position_embeddings,
            seq_len=4096,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len is None:
            seq_len = x.shape[-2]

        cos = self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device)
        return cos, sin


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, backend: str = "flash"):
        super().__init__()
        assert num_heads >= 1, f"Number of attention heads must be >= 1, got {num_heads}"
        assert hidden_size % num_heads == 0, (
            f"hidden size must be divisible by n_head, hidden_size={hidden_size}, n_head={num_heads}"
        )

        if backend not in ("flash", "sdpa"):
            raise ValueError(f"Unsupported attention backend: {backend}")

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.backend = backend
        self.rotary_emb = RotaryEmbedding(self.head_dim)

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # hidden_state: (B, T, D)
        batch_size, seq_len, _ = hidden_state.shape

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=hidden_state.device, dtype=torch.int32)

        q = self.q_proj(hidden_state)
        k = self.k_proj(hidden_state)
        v = self.v_proj(hidden_state)

        q = q.contiguous().view(batch_size, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        k = k.contiguous().view(batch_size, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        v = v.contiguous().view(batch_size, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

        cos, sin = self.rotary_emb(q, seq_len=seq_len)

        # batch, seq_len, n_head, head_dim
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.backend == "flash":
            # total_tokens, 3, n_head, head_dim
            qkv = (
                torch.stack((q, k, v), dim=2)
                .permute(0, 3, 2, 1, 4)
                .reshape(batch_size * seq_len, 3, self.num_heads, self.head_dim)
            )

            if segment_ids is None:
                indices, cu_seqlens, max_seqlen = _get_unpad_data(attention_mask)
            else:
                indices, cu_seqlens, max_seqlen = _get_segment_unpad_data(attention_mask, segment_ids)

            qkv = qkv.index_select(0, indices)

            attn_out = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, causal=True)
            padded = torch.zeros(
                (batch_size * seq_len, self.num_heads, self.head_dim),
                device=attn_out.device,
                dtype=attn_out.dtype,
            )
            padded.index_copy_(0, indices, attn_out)
            attn_out = padded.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        else:
            if segment_ids is not None:
                raise ValueError("segment_ids are not supported with SDPA attention backend.")
            attn_mask = None
            if attention_mask is not None:
                key_padding = attention_mask == 0
                if key_padding.any():
                    attn_mask = key_padding[:, None, None, :]
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=True,
            )
            attn_out = attn_out.swapaxes(1, 2)

        out = self.out_proj(attn_out.flatten(-2, -1))
        return out


class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_hidden, bias=False)
        self.up_proj = nn.Linear(d_model, d_hidden, bias=False)
        self.down_proj = nn.Linear(d_hidden, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class Router(nn.Module):
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.shared_gate = nn.Linear(d_model, 1, bias=False)  # W_{N+1} in R^{1 x D}

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.router(x)  # (B, T, N)
        g_shared = torch.sigmoid(self.shared_gate(x))  # (B, T, 1)
        return logits, g_shared


class MOELayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        k: int,
        moe_m_tile: int = 1,
        d_ff: int | None = None,
        d_expert: int | None = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.moe_m_tile = max(1, int(moe_m_tile))
        d_ff = hidden_size * 4 if d_ff is None else d_ff
        d_expert = d_ff // k if d_expert is None else d_expert

        self.router = Router(hidden_size, num_experts)

        self.expert_layers = nn.ModuleList([ExpertFFN(hidden_size, d_expert) for _ in range(num_experts)])
        self.shared_expert = ExpertFFN(hidden_size, d_ff)

    def forward(
        self, hidden_state: torch.Tensor, stats: MoEStats, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, ...]:
        # flatten input
        B, T, D = hidden_state.shape
        N = B * T

        # apply router and normalize weights
        # router_logits: (B,T,N), shared_expert_score: (B,T,1)
        router_logits, shared_expert_score = self.router(hidden_state)
        router_scores = torch.softmax(router_logits, dim=-1, dtype=torch.float32).to(hidden_state.dtype)
        topk_vals, topk_idx = torch.topk(router_scores, k=self.k)

        # TODO: Consider adding the renomalization back in
        # topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-8)

        # Flatten routing for dispatch
        x = hidden_state.reshape(N, D)

        # info tensors to group by expert
        M = N * self.k
        expert_for_route = topk_idx.reshape(M)
        gate_for_route = topk_vals.reshape(M)

        token_ids = torch.arange(N, device=hidden_state.device)
        token_for_route = token_ids.repeat_interleave(self.k)

        if attention_mask is not None:
            flat_mask = attention_mask.reshape(N).to(device=hidden_state.device)
            route_valid = flat_mask.repeat_interleave(self.k) > 0
            expert_for_route = expert_for_route[route_valid]
            gate_for_route = gate_for_route[route_valid]
            token_for_route = token_for_route[route_valid]

        # actually group by expert
        expert_sorted, compute_order = torch.sort(expert_for_route)
        gate_sorted = gate_for_route[compute_order]
        token_sorted = token_for_route[compute_order]
        x_sorted = x[token_sorted]

        # Apply the experts on grouped data, with per-expert token rounding.
        counts = torch.bincount(expert_sorted, minlength=self.num_experts)
        offsets = torch.cumsum(counts, dim=0)
        starts = offsets - counts

        if self.moe_m_tile > 1:
            keep_limit = counts - (counts % self.moe_m_tile)
        else:
            keep_limit = counts

        y_sorted = torch.zeros_like(x_sorted)

        for i, exp in enumerate(self.expert_layers):
            s_i, t = starts[i], starts[i] + keep_limit[i]
            if s_i == t:
                continue

            y_sorted[s_i:t] = exp(x_sorted[s_i:t])

        # weight the outputs
        y_sorted = y_sorted * gate_sorted.unsqueeze(-1)

        # finalize output by adding
        y_out = torch.zeros(N, D, device=y_sorted.device, dtype=y_sorted.dtype)
        y_out.scatter_add_(0, index=token_sorted.unsqueeze(-1).expand(-1, D), src=y_sorted)
        y_out = y_out.reshape(B, T, D)

        shared_in = hidden_state.reshape(N, D)
        shared_out = self.shared_expert(shared_in).reshape(B, T, D)

        y_out = y_out + shared_expert_score * shared_out
        if attention_mask is not None:
            y_out = y_out * attention_mask.unsqueeze(-1).to(y_out.dtype)

        # aux loss specifics
        if keep_limit.sum() > 0:
            load = keep_limit / (keep_limit.sum() + 1e-12)
        else:
            load = torch.zeros(self.num_experts, device=counts.device, dtype=router_scores.dtype)
        if attention_mask is None:
            importance = router_scores.mean(dim=(0, 1))
        else:
            flat_mask = attention_mask.reshape(N).to(device=router_scores.device)
            denom = flat_mask.sum() + 1e-12
            importance = (router_scores * flat_mask.view(B, T, 1)).sum(dim=(0, 1)) / denom

        stats.add_values_(importance, load)

        return y_out, stats
