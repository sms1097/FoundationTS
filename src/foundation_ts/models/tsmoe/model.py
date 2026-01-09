from __future__ import annotations

import torch
from torch import nn

from foundation_ts.models.tsmoe.layers import Attention, EfficientMOELayer, RMSNorm
from foundation_ts.models.tsmoe.stats import MoEStats


class TimeEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch: bool = False,
        patch_len: int = 32,
        patch_stride: int = 32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch = patch
        self.patch_len = patch_len
        self.stride = patch_stride
        self.act_fn = nn.SiLU()

        _in_size = 1 if not self.patch else patch_len

        self.emb_layer = nn.Linear(_in_size, self.hidden_size, bias=False)
        self.gate_layer = nn.Linear(_in_size, self.hidden_size, bias=False)

    def _patch_input(self, x):
        B, T, C = x.shape
        if T < self.patch_len:
            raise ValueError(f"T={T} < patch_len={self.patch_len}")

        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  # (B, N, P, C)
        patches = patches.contiguous().view(B, patches.size(1), self.patch_len * C)  # (B, N, P*C)
        return patches

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, S, token_dim)
        return self.act_fn(self.gate_layer(tokens)) * self.emb_layer(tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch:
            x = self._patch_input(x)  # (B, num_patches, patch_len*C)
        # else x is already (B, T, C) and token_dim=C
        return self._embed(x)  # (B, seq_len, hidden_size)


class MultiHorizonOutputLayer(nn.Module):
    def __init__(self, hidden_size: int, horizons: list[int] = (1, 8, 32, 64)):
        super().__init__()
        self.horizons = sorted(horizons)
        self.heads = nn.ModuleDict({str(h): nn.Linear(hidden_size, h) for h in self.horizons})

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return {h: self.heads[str(h)](hidden_state) for h in self.horizons}


class MOEDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_expert_layers: int,
        n_head: int,
        k: int,
        d_ff: int | None = None,
        d_expert: int | None = None,
        max_batch_tokens: int | None = None,
        capacity_factor: float = 1.25,
        drop_policy: str = "drop",
    ):
        super().__init__()
        if max_batch_tokens is None:
            raise ValueError("max_batch_tokens is required for EfficientMOELayer.")
        self.num_experts = num_experts
        self.rms_norm1 = RMSNorm(hidden_size)

        self.attention = Attention(hidden_size, n_head)
        self.rms_norm2 = RMSNorm(hidden_size)
        self.expert_layers = nn.ModuleList(
            [
                EfficientMOELayer(
                    hidden_size,
                    num_experts,
                    k,
                    max_batch_tokens=max_batch_tokens,
                    d_ff=d_ff,
                    d_expert=d_expert,
                    capacity_factor=capacity_factor,
                    drop_policy=drop_policy,
                )
                for _ in range(num_expert_layers)
            ]
        )

    def forward(
        self, tokens: torch.Tensor, stats: MoEStats, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, ...]:
        norm_input_state = self.rms_norm1(tokens)

        hidden_state = self.attention(norm_input_state, attention_mask=attention_mask)
        hidden_state = self.rms_norm2(hidden_state + norm_input_state)

        norm_hidden_state = hidden_state

        for exp in self.expert_layers:
            hidden_state, stats = exp(hidden_state, stats, attention_mask=attention_mask)

        return hidden_state + norm_hidden_state, stats


class TSMOE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_decoder_layers: int,
        num_experts: int,
        num_expert_layers: int,
        k: int,
        n_head: int,
        horizons: list[int],
        patch: bool,
        patch_len: int,
        patch_stride: int,
        d_ff: int | None = None,
        d_expert: int | None = None,
        max_batch_tokens: int | None = None,
        capacity_factor: float = 1.25,
        drop_policy: str = "drop",
    ):
        super().__init__()
        if max_batch_tokens is None:
            raise ValueError("max_batch_tokens is required for EfficientMOELayer.")

        self.num_experts = num_experts
        self.embed_layer = TimeEmbedding(hidden_size, patch, patch_len, patch_stride)
        self.decoder_layers = nn.ModuleList(
            MOEDecoderLayer(
                hidden_size,
                num_experts,
                num_expert_layers,
                n_head,
                k,
                d_ff=d_ff,
                d_expert=d_expert,
                max_batch_tokens=max_batch_tokens,
                capacity_factor=capacity_factor,
                drop_policy=drop_policy,
            )
            for _ in range(n_decoder_layers)
        )

        self.output_layer = MultiHorizonOutputLayer(hidden_size, horizons)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden_state = self.embed_layer(x)

        stats = MoEStats.zeros(self.num_experts, x.device)
        for idx, dl in enumerate(self.decoder_layers):
            hidden_state, stats = dl(hidden_state, stats, attention_mask=attention_mask)

        out = self.output_layer(hidden_state)

        return out, stats
