import pytest
import torch

from foundation_ts.models.tsmoe import TSMOE
from foundation_ts.models.tsmoe.layers import Attention


def test_moe():
    if not torch.cuda.is_available():
        pytest.skip("TSMOE MoE test requires CUDA")
    hidden_size = 32
    n_decoder_layers = 5
    patch = False
    patch_len = 32
    patch_stride = 32
    num_experts = 3
    num_expert_layers = 3
    n_head = 8
    horizons = [1, 8, 32, 64]
    batch_size = 6
    time_size = 8
    k = 2
    capacity_factor = 0.5
    max_batch_tokens = batch_size * time_size

    device = torch.device("cuda")
    model = TSMOE(
        hidden_size,
        n_decoder_layers,
        num_experts,
        num_expert_layers,
        k,
        n_head,
        horizons,
        patch,
        patch_len,
        patch_stride,
        max_batch_tokens=max_batch_tokens,
        capacity_factor=capacity_factor,
    ).to(device).to(torch.bfloat16)

    inputs = torch.rand(batch_size, time_size, 1, dtype=torch.bfloat16, device=device)
    attention_mask = torch.ones(batch_size, time_size, device=device, dtype=torch.int32)
    attention_mask[0, -2:] = 0

    outputs, stats = model(inputs, attention_mask=attention_mask)
    assert set(outputs.keys()) == set(horizons)
    for h in horizons:
        assert outputs[h].shape == (batch_size, time_size, h)
    assert stats.importance.shape == (num_experts,)
    assert stats.load.shape == (num_experts,)


def test_attention():
    if not torch.cuda.is_available():
        pytest.skip("flash-attn Attention requires CUDA")
    torch.manual_seed(0)
    device = torch.device("cuda")

    batch_size = 2
    seq_len = 5
    hidden_size = 32
    num_heads = 4
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.bfloat16)
    attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.int32)
    attention_mask[1, -2:] = 0

    attn = Attention(hidden_size=hidden_size, num_heads=num_heads).to(device).to(torch.bfloat16)
    out = attn(x, attention_mask=attention_mask)

    assert out.shape == x.shape
