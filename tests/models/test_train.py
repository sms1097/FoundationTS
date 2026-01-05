import json

import torch

from foundation_ts.models.training import (
    DatasetConfig,
    ModelConfig,
    RunnerConfig,
    TrainingConfig,
    _build_attention_mask,
    _forecast_loss,
    train,
)
from foundation_ts.models.tsmoe import TSMOE


def test_forecast_loss_masks_padding():
    loss_fn = torch.nn.L1Loss(reduction="none")
    labels = torch.tensor([[1.0, 2.0, 3.0]])
    loss_masks = torch.tensor([[1.0, 0.0, 1.0]])
    preds = torch.tensor([[[1.0], [9.0], [4.0]]])
    outputs = {1: preds}

    loss = _forecast_loss(outputs, labels, loss_masks, loss_fn)

    # Only positions 0 and 2 contribute: |1-1| + |4-3| = 1 over 2 tokens.
    assert torch.isclose(loss, torch.tensor(0.5))


def test_forecast_loss_patch_reduces_length():
    loss_fn = torch.nn.L1Loss(reduction="none")
    labels = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    loss_masks = torch.ones_like(labels)
    preds = torch.tensor([[[4.0], [10.0]]])
    outputs = {1: preds}

    loss = _forecast_loss(
        outputs,
        labels,
        loss_masks,
        loss_fn,
        patch=True,
        patch_len=4,
        patch_stride=4,
    )

    # Patched labels are [4, 8], so |4-4| + |10-8| = 2 over 2 tokens.
    assert torch.isclose(loss, torch.tensor(1.0))


def test_build_attention_mask_patch():
    loss_masks = torch.tensor([[1, 1, 0, 0, 1, 0, 0, 0]], dtype=torch.float32)
    mask = _build_attention_mask(loss_masks, patch=True, patch_len=4, patch_stride=4)
    assert mask.shape == (1, 2)
    assert torch.equal(mask, torch.tensor([[1.0, 1.0]]))


def test_train_runs_on_dummy_dataset(tmp_path):
    sequences = [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 11.0, 12.0, 13.0]]
    data_path = tmp_path / "sequences.json"
    data_path.write_text(json.dumps(sequences), encoding="utf-8")

    model_config = ModelConfig(
        hidden_size=8,
        n_decoder_layers=1,
        num_experts=2,
        num_expert_layers=1,
        k=1,
        n_head=2,
        horizons=[1, 4],
    )
    train_config = TrainingConfig(
        model_config=model_config,
        epochs=1,
        steps_per_epoch=2,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        use_bf16=False,
        use_amp=False,
        val_split=0.0,
        log_every=0,
        val_every=0,
        checkpoint_every=0,
        tensorboard=False,
        num_workers=0,
    )
    dataset_config = DatasetConfig(
        dataset_path=str(data_path),
        seq_max_len=6,
        seq_stride=6,
        normalization_func="zero",
    )

    model = train(RunnerConfig(dataset_config, train_config))
    assert isinstance(model, TSMOE)
