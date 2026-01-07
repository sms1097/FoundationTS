from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from foundation_ts.models.tsmoe.stats import MoEStats

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.set_float32_matmul_precision("high")


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
            seq_len=max_position_embeddings,
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

        if (
            seq_len > self.max_seq_len_cached
            or self.cos_cached.device != x.device
            or self.cos_cached.dtype != x.dtype
        ):
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert num_heads >= 1, f"Number of attention heads must be >= 1, got {num_heads}"
        assert hidden_size % num_heads == 0, (
            f"hidden size must be divisible by n_head, hidden_size={hidden_size}, n_head={num_heads}"
        )

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rotary_emb = RotaryEmbedding(self.head_dim)

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # hidden_state: (B, T, D)
        B, T, _ = hidden_state.shape

        qkv = self.qkv_proj(hidden_state)
        qkv = qkv.contiguous().view(B, T, self.num_heads, 3 * self.head_dim)
        qkv = qkv.swapaxes(1, 2)

        q = qkv[..., : self.head_dim]
        k = qkv[..., self.head_dim : 2 * self.head_dim]
        v = qkv[..., 2 * self.head_dim :]

        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn_mask = None
        if attention_mask is not None:
            key_padding = attention_mask == 0
            if key_padding.any():
                attn_mask = key_padding[:, None, None, :]

        causal_mask = torch.triu(torch.ones((T, T), device=q.device, dtype=torch.bool), diagonal=1)
        if attn_mask is None:
            combined_mask = causal_mask
        else:
            combined_mask = attn_mask | causal_mask[None, None, :, :]

        # SDPA path with explicit combined mask (padding + causal)
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=combined_mask,
            is_causal=False,
            # , scale=0.5 / math.sqrt(self.head_dim)
        )
        # Manual path (kept for reference/testing):
        # scale = 1.0 / (self.head_dim**0.5)
        # scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        # if attn_mask is not None:
        #     scores = scores.masked_fill(attn_mask, torch.finfo(scores.dtype).min)
        # scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        # attn = torch.softmax(scores, dim=-1)
        # _ensure_finite("attention.softmax", attn)
        # out = torch.matmul(attn, v)
        # _ensure_finite("attention.sdpa", out)

        out = out.swapaxes(1, 2)
        out = self.out_proj(out.flatten(-2, -1))
        return out


class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_hidden, bias=False)
        self.w2 = nn.Linear(d_hidden, d_model, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))


class Router(nn.Module):
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.shared_gate = nn.Linear(d_model, 1, bias=False)  # W_{N+1} in R^{1 x D}

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(x)  # (B, T, N)
        s = self.softmax(logits)  # (B, T, N)
        g_shared = torch.sigmoid(self.shared_gate(x))  # (B, T, 1)
        return s, g_shared


class MOELayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        k: int,
        d_ff: int | None = None,
        d_expert: int | None = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        d_expert = hidden_size // 2 if d_expert is None else d_expert
        d_ff = hidden_size // 2 if d_ff is None else d_ff

        self.router = Router(hidden_size, num_experts)

        self.expert_layers = nn.ModuleList(
            [ExpertFFN(hidden_size, d_expert) for _ in range(num_experts)]
        )
        self.shared_expert = ExpertFFN(hidden_size, d_ff)

    def forward(
        self, hidden_state: torch.Tensor, stats: MoEStats, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, ...]:
        # flatten input
        B, T, D = hidden_state.shape
        N = B * T

        # apply router and normalize weights
        # router_scores: (B,T,N), shared_expert_score: (B,T,1)
        router_scores, shared_expert_score = self.router(hidden_state)
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

        # actually group by expert
        expert_sorted, compute_order = torch.sort(expert_for_route)
        gate_sorted = gate_for_route[compute_order]
        token_sorted = token_for_route[compute_order]
        x_sorted = x[token_sorted]

        # Apply the experts on grouped data
        counts = torch.bincount(expert_sorted, minlength=self.num_experts)
        offsets = torch.cumsum(counts, dim=0)
        starts = offsets - counts

        y_sorted = torch.empty_like(x_sorted)

        for i, exp in enumerate(self.expert_layers):
            s_i, t = starts[i], offsets[i]
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

        # aux loss specifics
        if attention_mask is None:
            load = counts / (counts.sum() + 1e-12)  # (N,)
            importance = router_scores.mean(dim=(0, 1))
            # importance = router_scores.mean(dim=(0, 1))
        else:
            flat_mask = attention_mask.reshape(N).to(device=router_scores.device)
            denom = flat_mask.sum() + 1e-12
            importance = (router_scores * flat_mask.view(B, T, 1)).sum(dim=(0, 1)) / denom
            route_valid = flat_mask.repeat_interleave(self.k) > 0
            if route_valid.any():
                masked_counts = torch.bincount(expert_for_route[route_valid], minlength=self.num_experts)
                load = masked_counts / (masked_counts.sum() + 1e-12)
            else:
                load = torch.zeros(self.num_experts, device=counts.device, dtype=router_scores.dtype)

        stats.add_values_(importance, load)

        return y_out, stats
