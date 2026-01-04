import torch

from foundation_ts.models.tsmoe import TSMOE


def test_moe():
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
    )

    inputs = torch.rand(batch_size, time_size, 1)

    model(inputs)
