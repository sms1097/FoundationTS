import torch

from foundation_ts.models.tsmoe.stats import MoEStats


def aux_loss(stats: MoEStats):
    N = stats.importance.numel()
    return N * torch.sum(stats.importance * stats.load)


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_batch(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, ...]:
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    loss_masks = batch["loss_masks"].to(device, non_blocking=True)

    if input_ids.dim() == 2:
        input_ids = input_ids.unsqueeze(-1)
    if labels.dim() == 3 and labels.size(-1) == 1:
        labels = labels.squeeze(-1)
    if loss_masks.dim() == 3 and loss_masks.size(-1) == 1:
        loss_masks = loss_masks.squeeze(-1)

    return input_ids, labels, loss_masks


def _build_attention_mask(
    loss_masks: torch.Tensor, patch: bool, patch_len: int, patch_stride: int
) -> torch.Tensor:
    if not patch:
        return loss_masks
    if loss_masks.size(1) < patch_len:
        raise ValueError(f"T={loss_masks.size(1)} < patch_len={patch_len}")
    patches = loss_masks.unfold(dimension=1, size=patch_len, step=patch_stride)
    return (patches.sum(dim=-1) > 0).to(loss_masks.dtype)


def _patch_labels_and_masks(
    labels: torch.Tensor, loss_masks: torch.Tensor, patch_len: int, patch_stride: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if labels.size(1) < patch_len:
        raise ValueError(f"T={labels.size(1)} < patch_len={patch_len}")
    label_patches = labels.unfold(dimension=1, size=patch_len, step=patch_stride)
    mask_patches = loss_masks.unfold(dimension=1, size=patch_len, step=patch_stride)
    patched_labels = label_patches[..., -1]
    patched_masks = (mask_patches.sum(dim=-1) > 0).to(loss_masks.dtype)
    return patched_labels, patched_masks


def _build_horizon_targets(
    labels: torch.Tensor, loss_masks: torch.Tensor, horizon: int
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T = labels.shape
    targets = torch.zeros((B, T, horizon), device=labels.device, dtype=labels.dtype)
    masks = torch.zeros((B, T, horizon), device=labels.device, dtype=labels.dtype)
    for offset in range(horizon):
        valid_len = T - offset
        if valid_len <= 0:
            break
        targets[:, :valid_len, offset] = labels[:, offset:]
        masks[:, :valid_len, offset] = loss_masks[:, offset:].to(labels.dtype)
    return targets, masks


def _forecast_loss(
    outputs: dict[int, torch.Tensor],
    labels: torch.Tensor,
    loss_masks: torch.Tensor,
    loss_fn: torch.nn.Module,
    patch: bool = False,
    patch_len: int = 32,
    patch_stride: int = 32,
) -> torch.Tensor:
    if not outputs:
        raise ValueError("Model returned empty outputs.")
    if patch:
        labels, loss_masks = _patch_labels_and_masks(labels, loss_masks, patch_len, patch_stride)
    total = torch.zeros((), device=labels.device)
    for horizon, pred in outputs.items():
        targets, masks = _build_horizon_targets(labels, loss_masks, horizon)
        per_item = loss_fn(pred, targets) * masks
        denom = masks.sum().clamp(min=1.0)
        total = total + per_item.sum() / denom
    return total / len(outputs)
