import argparse
from pathlib import Path

import torch


def _summarize_tensor(name: str, tensor: torch.Tensor) -> None:
    flat = tensor.flatten()
    finite = torch.isfinite(flat)
    finite_count = int(finite.sum().item())
    total = flat.numel()
    print(f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}")
    print(f"  finite={finite_count}/{total} nan={torch.isnan(flat).sum().item()} inf={torch.isinf(flat).sum().item()}")
    if finite_count == 0:
        return
    finite_vals = flat[finite]
    if not torch.is_floating_point(finite_vals):
        finite_vals = finite_vals.float()
    print(
        "  stats min/max/mean/abs_mean/std"
        f"={finite_vals.min().item():.4g}/"
        f"{finite_vals.max().item():.4g}/"
        f"{finite_vals.mean().item():.4g}/"
        f"{finite_vals.abs().mean().item():.4g}/"
        f"{finite_vals.std(unbiased=False).item():.4g}"
    )


def _summarize_masks(loss_masks: torch.Tensor) -> None:
    if loss_masks.dim() == 3 and loss_masks.size(-1) == 1:
        loss_masks = loss_masks.squeeze(-1)
    per_seq = loss_masks.sum(dim=1)
    print(
        "  loss_masks per-seq min/max/mean="
        f"{per_seq.min().item():.4g}/"
        f"{per_seq.max().item():.4g}/"
        f"{per_seq.float().mean().item():.4g}"
    )
    print(f"  fully-masked sequences={(per_seq == 0).sum().item()}")


def _summarize_sequences(input_ids: torch.Tensor) -> None:
    if input_ids.dim() == 2:
        input_ids = input_ids.unsqueeze(-1)
    seq_abs_max = input_ids.abs().amax(dim=(1, 2))
    print(
        "  input_ids abs max per-seq min/max/mean="
        f"{seq_abs_max.min().item():.4g}/"
        f"{seq_abs_max.max().item():.4g}/"
        f"{seq_abs_max.mean().item():.4g}"
    )


def _load_batch(batch_path: Path) -> dict:
    payload = torch.load(batch_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected payload type: {type(payload)}")
    return payload


def _resolve_batch_path(root: Path) -> Path:
    if root.is_file():
        return root
    candidates = sorted(root.glob("failed_batch_*.pt"))
    if not candidates:
        raise RuntimeError(f"No failed_batch_*.pt found in {root}")
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple failed batches found in {root}; choose one explicitly.")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a failed batch dump.")
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("checkpoints/failed_batches"),
        help="Path to failed_batch_*.pt or its directory.",
    )
    args = parser.parse_args()

    batch_path = _resolve_batch_path(args.path)
    payload = _load_batch(batch_path)
    print(f"loaded: {batch_path}")
    print(f"error: {payload.get('error')}")

    input_ids = payload.get("input_ids")
    labels = payload.get("labels")
    loss_masks = payload.get("loss_masks")
    attention_mask = payload.get("attention_mask")

    if input_ids is None or labels is None or loss_masks is None:
        raise RuntimeError("Missing required tensors in payload.")

    _summarize_tensor("input_ids", input_ids)
    _summarize_sequences(input_ids)
    _summarize_tensor("labels", labels)
    _summarize_tensor("loss_masks", loss_masks)
    _summarize_masks(loss_masks)
    if attention_mask is not None:
        _summarize_tensor("attention_mask", attention_mask)


if __name__ == "__main__":
    main()
