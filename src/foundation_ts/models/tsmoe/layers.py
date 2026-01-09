from __future__ import annotations

import math
import torch
from flash_attn import flash_attn_varlen_qkvpacked_func
from torch import nn
from torch.nn import functional as F

from foundation_ts.models.tsmoe.stats import MoEStats

torch.set_float32_matmul_precision("high")


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


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

        # if (
        #     seq_len > self.max_seq_len_cached
        #     or self.cos_cached.device != x.device
        #     or self.cos_cached.dtype != x.dtype
        # ):
        #     self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        cos = self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device)
        return cos, sin


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
        batch_size, seq_len, _ = hidden_state.shape

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=hidden_state.device, dtype=torch.int32)

        qkv = self.qkv_proj(hidden_state)
        qkv = qkv.contiguous().view(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.swapaxes(1, 2)

        q = qkv[..., : self.head_dim]
        k = qkv[..., self.head_dim : 2 * self.head_dim]
        v = qkv[..., 2 * self.head_dim :]

        cos, sin = self.rotary_emb(q, seq_len=seq_len)

        # batch, seq_len, n_head, head_dim
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # total_tokens, 3, n_head, head_dim
        qkv = (
            torch.stack((q, k, v), dim=2)
            .permute(0, 3, 2, 1, 4)
            .reshape(batch_size * seq_len, 3, self.num_heads, self.head_dim)
        )

        indices, cu_seqlens, max_seqlen = _get_unpad_data(attention_mask)

        qkv = qkv.index_select(0, indices)

        attn_out = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, causal=True)
        padded = torch.zeros(
            (batch_size * seq_len, self.num_heads, self.head_dim),
            device=attn_out.device,
            dtype=attn_out.dtype,
        )
        padded.index_copy_(0, indices, attn_out)
        attn_out = padded.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # attn_mask = None
        # if attention_mask is not None:
        #     key_padding = attention_mask == 0
        #     if key_padding.any():
        #         attn_mask = key_padding[:, None, None, :]

        # causal_mask = torch.triu(torch.ones((T, T), device=q.device, dtype=torch.bool), diagonal=1)
        # if attn_mask is None:
        #     combined_mask = causal_mask
        # else:
        #     combined_mask = attn_mask | causal_mask[None, None, :, :]

        # # SDPA path with explicit combined mask (padding + causal)
        # out = torch.nn.functional.scaled_dot_product_attention(
        #     q, k, v, attn_mask=combined_mask, is_causal=False
        # )

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

        out = self.out_proj(attn_out.flatten(-2, -1))
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

        self.expert_layers = nn.ModuleList([ExpertFFN(hidden_size, d_expert) for _ in range(num_experts)])
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


class BatchedExperts(torch.nn.Module):
    def __init__(self, num_experts, hidden, ff):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.randn(num_experts, hidden, ff))
        self.w2 = torch.nn.Parameter(torch.randn(num_experts, ff, hidden))

    def forward(self, x):
        # x: [E, C, H]
        x = torch.einsum("ech,ehf->ecf", x, self.w1)
        x = torch.nn.functional.gelu(x)
        x = torch.einsum("ecf,efh->ech", x, self.w2)
        return x


# --- Example interfaces you said you have ---
# Router(hidden_size, num_experts) -> (router_scores[B,T,E], shared_score[B,T,1])
# BatchedExperts(num_experts, hidden_size, d_expert) accepts [E,C,D] -> [E,C,D]
# ExpertFFN(hidden_size, d_ff) accepts [N,D] -> [N,D]
# MoEStats has add_values_(importance, load)


class EfficientMOELayer(nn.Module):
    """
    Compile-friendly MoE:
      - fixed expert capacity (static shape)
      - vectorized dispatch (no Python loops)
      - data-dependent masking only (no shape changes)
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        k: int,
        max_batch_tokens: int,  # <-- REQUIRED for fixed capacity
        d_ff: int | None = None,
        d_expert: int | None = None,
        capacity_factor: float = 1.25,  # typical 1.1-1.3; tune
        drop_policy: str = "drop",  # "drop" (recommended) or "zero"
    ):
        super().__init__()
        assert k >= 1
        assert drop_policy in ("drop", "zero")

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = float(capacity_factor)
        self.drop_policy = drop_policy

        # Fixed capacity based on maximum tokens this layer will ever see in one forward.
        # This is the *shape contract* that enables compilation.
        cap = math.ceil(max_batch_tokens * k / num_experts * self.capacity_factor)
        self.capacity = max(1, int(cap))

        d_expert = hidden_size // 2 if d_expert is None else d_expert
        d_ff = hidden_size // 2 if d_ff is None else d_ff

        self.router = Router(hidden_size, num_experts)
        self.experts = BatchedExperts(num_experts, hidden_size, d_expert)
        self.shared_expert = ExpertFFN(hidden_size, d_ff)

    @torch.no_grad()
    def _check_static(self, B: int, T: int):
        # optional sanity check for debugging; comment out in production
        pass

    def forward(
        self,
        hidden_state: torch.Tensor,  # [B,T,D]
        stats,
        attention_mask: torch.Tensor | None = None,  # [B,T] with 1 for valid
    ):
        B, T, D = hidden_state.shape
        N = B * T
        E = self.num_experts
        K = self.k
        C = self.capacity

        # Router
        router_scores, shared_expert_score = self.router(hidden_state)  # [B,T,E], [B,T,1]
        # topk along experts
        topk_vals, topk_idx = torch.topk(router_scores, k=K, dim=-1)  # [B,T,K], [B,T,K]

        # (Optional) renormalize gates per token for top-k
        # This is usually helpful for stability; keep if you want.
        # topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)

        # Flatten tokens
        x = hidden_state.reshape(N, D)  # [N,D]

        # Build routing lists length R = N*K (static length given B,T,K)
        # token_for_route: [R]
        token_ids = torch.arange(N, device=hidden_state.device)
        token_for_route = token_ids.repeat_interleave(K)  # [N*K]
        expert_for_route = topk_idx.reshape(-1)  # [N*K]
        gate_for_route = topk_vals.reshape(-1)  # [N*K]

        # Mask out padded tokens (data-only, no shape change)
        if attention_mask is not None:
            # [N] bool
            flat_mask = attention_mask.reshape(N).to(device=hidden_state.device)
            valid_token = flat_mask > 0
            # expand to routes [N*K]
            valid_route = valid_token.repeat_interleave(K)
            # zero invalid routes
            gate_for_route = gate_for_route * valid_route.to(gate_for_route.dtype)
            # send invalid routes to expert 0; gate=0 ensures no effect
            expert_for_route = torch.where(valid_route, expert_for_route, torch.zeros_like(expert_for_route))

        # Sort routes by expert id (still static length)
        expert_sorted, order = torch.sort(expert_for_route)  # [R]
        token_sorted = token_for_route[order]  # [R]
        gate_sorted = gate_for_route[order]  # [R]
        x_sorted = x[token_sorted]  # [R,D]

        # Count per expert (static E)
        counts = torch.bincount(expert_sorted, minlength=E)  # [E]
        offsets = torch.cumsum(counts, dim=0)  # [E]
        starts = offsets - counts  # [E]

        # Position within each expert segment: [R]
        # positions[r] = r - starts[expert_sorted[r]]
        r = torch.arange(expert_sorted.numel(), device=hidden_state.device)
        positions = r - starts[expert_sorted]

        # Keep within capacity; IMPORTANT: do NOT change shapes.
        in_cap = positions < C  # [R] bool

        # Also drop routes with zero gate (e.g., padded tokens). Keep shape static.
        nonzero = gate_sorted != 0
        keep = in_cap & nonzero

        # Prepare expert input buffer: [E,C,D] static
        expert_inputs = x_sorted.new_zeros((E, C, D))

        # Scatter kept routes into expert buffers (vectorized, no Python loop)
        # Advanced indexing writes (E_idx, pos_idx) rows.
        e_idx = expert_sorted
        p_idx = positions.clamp(min=0, max=C - 1)  # safe for dropped items; they'll be masked by `keep`
        # only write kept items
        if self.drop_policy == "drop":
            # For dropped items, do nothing (they remain zeros)
            expert_inputs[e_idx[keep], p_idx[keep]] = x_sorted[keep]
        else:
            # "zero" is same here because buffer is zeros; left for completeness
            expert_inputs[e_idx[keep], p_idx[keep]] = x_sorted[keep]

        # Run all experts in one call (static)
        expert_outputs = self.experts(expert_inputs)  # [E,C,D]

        # Gather outputs for each route (static R)
        y_sorted = expert_outputs[e_idx, p_idx]  # [R,D]

        # Zero out dropped/invalid routes, then apply gates
        y_sorted = y_sorted * keep.to(y_sorted.dtype).unsqueeze(-1)
        y_sorted = y_sorted * gate_sorted.to(y_sorted.dtype).unsqueeze(-1)

        # Scatter-add back to tokens: [N,D] static
        y_out = x.new_zeros((N, D))
        y_out.scatter_add_(0, token_sorted.unsqueeze(-1).expand(-1, D), y_sorted.to(y_out.dtype))
        y_out = y_out.reshape(B, T, D)

        # Shared expert (dense) path
        shared_out = self.shared_expert(x).reshape(B, T, D)
        y_out = y_out + shared_expert_score.to(y_out.dtype) * shared_out

        # --- stats / aux (kept close to your code) ---
        # Important: use original (unmasked) routing for importance,
        # and masked routing for load if attention_mask exists.
        if attention_mask is None:
            # load based on routed topk indices (all tokens)
            load_counts = torch.bincount(expert_for_route, minlength=E)
            load = load_counts / (load_counts.sum() + 1e-12)
            importance = router_scores.mean(dim=(0, 1))
        else:
            flat_mask = attention_mask.reshape(N).to(device=router_scores.device)
            denom = flat_mask.sum() + 1e-12
            importance = (router_scores * flat_mask.view(B, T, 1)).sum(dim=(0, 1)) / denom

            # load should reflect only valid tokens (gate nonzero)
            # expert_for_route here already has invalid routes redirected to 0 but gate=0.
            # So count only nonzero-gated routes:
            valid_routes = gate_for_route != 0
            load_counts = (
                torch.bincount(expert_for_route[valid_routes], minlength=E)
                if valid_routes.any()
                else torch.zeros(E, device=router_scores.device, dtype=torch.long)
            )
            load = load_counts / (load_counts.sum() + 1e-12)

        stats.add_values_(importance, load)

        return y_out, stats


class AdaptiveMOELayer(nn.Module):
    """
    Dynamic-capacity MoE (non-compile-friendly):
      - capacity is derived from the current batch
      - uses batched experts but includes per-batch routing compaction
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        k: int,
        d_ff: int | None = None,
        d_expert: int | None = None,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = float(capacity_factor)

        d_expert = hidden_size // 2 if d_expert is None else d_expert
        d_ff = hidden_size // 2 if d_ff is None else d_ff

        self.router = Router(hidden_size, num_experts)
        self.experts = BatchedExperts(num_experts, hidden_size, d_expert)
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
        expert_for_route = topk_idx.reshape(-1)
        gate_for_route = topk_vals.reshape(-1)
        token_ids = torch.arange(N, device=hidden_state.device)
        token_for_route = token_ids.repeat_interleave(self.k)

        if attention_mask is None:
            tokens_per_batch = N
        else:
            flat_mask = attention_mask.reshape(N).to(device=hidden_state.device)
            route_valid = flat_mask.repeat_interleave(self.k) > 0
            expert_for_route = expert_for_route[route_valid]
            gate_for_route = gate_for_route[route_valid]
            token_for_route = token_for_route[route_valid]
            tokens_per_batch = int(flat_mask.sum().item())

        capacity = int(math.ceil(tokens_per_batch * self.k / self.num_experts * self.capacity_factor))
        capacity = max(1, capacity)

        # actually group by expert
        expert_sorted, compute_order = torch.sort(expert_for_route)
        gate_sorted = gate_for_route[compute_order]
        token_sorted = token_for_route[compute_order]
        x_sorted = x[token_sorted]

        # Apply the experts on grouped data
        if expert_sorted.numel():
            counts = torch.bincount(expert_sorted, minlength=self.num_experts)
            offsets = torch.cumsum(counts, dim=0)
            starts = offsets - counts
            positions = (
                torch.arange(expert_sorted.numel(), device=expert_sorted.device) - starts[expert_sorted]
            )
            keep = positions < capacity
            if keep.sum() < keep.numel():
                expert_sorted = expert_sorted[keep]
                gate_sorted = gate_sorted[keep]
                token_sorted = token_sorted[keep]
                x_sorted = x_sorted[keep]
                counts = torch.bincount(expert_sorted, minlength=self.num_experts)
                offsets = torch.cumsum(counts, dim=0)
                starts = offsets - counts

            y_sorted = torch.empty_like(x_sorted)
            expert_inputs = torch.zeros(
                (self.num_experts, capacity, D),
                device=x_sorted.device,
                dtype=x_sorted.dtype,
            )
            for i in range(self.num_experts):
                s_i, t = starts[i], offsets[i]
                if s_i == t:
                    continue
                expert_inputs[i, : t - s_i] = x_sorted[s_i:t]

            expert_outputs = self.experts(expert_inputs)
            for i in range(self.num_experts):
                s_i, t = starts[i], offsets[i]
                if s_i == t:
                    continue
                y_sorted[s_i:t] = expert_outputs[i, : t - s_i]
        else:
            y_sorted = torch.empty_like(x_sorted)

        # weight the outputs
        y_sorted = y_sorted * gate_sorted.unsqueeze(-1)

        # finalize output by adding
        y_out = torch.zeros(N, D, device=y_sorted.device, dtype=y_sorted.dtype)
        if token_sorted.numel():
            y_out.scatter_add_(0, index=token_sorted.unsqueeze(-1).expand(-1, D), src=y_sorted)
        y_out = y_out.reshape(B, T, D)

        shared_in = hidden_state.reshape(N, D)
        shared_out = self.shared_expert(shared_in).reshape(B, T, D)
        y_out = y_out + shared_expert_score * shared_out

        # aux loss specifics
        if attention_mask is None:
            counts = torch.bincount(expert_for_route, minlength=self.num_experts)
            load = counts / (counts.sum() + 1e-12)  # (N,)
            importance = router_scores.mean(dim=(0, 1))
        else:
            flat_mask = attention_mask.reshape(N).to(device=router_scores.device)
            denom = flat_mask.sum() + 1e-12
            importance = (router_scores * flat_mask.view(B, T, 1)).sum(dim=(0, 1)) / denom
            if expert_for_route.numel():
                masked_counts = torch.bincount(expert_for_route, minlength=self.num_experts)
                load = masked_counts / (masked_counts.sum() + 1e-12)
            else:
                load = torch.zeros(self.num_experts, device=router_scores.device, dtype=router_scores.dtype)

        stats.add_values_(importance, load)

        return y_out, stats
