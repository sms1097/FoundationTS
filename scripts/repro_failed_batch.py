import argparse
from pathlib import Path

import torch

from foundation_ts.models.training.utils import _forecast_loss, _prepare_batch
from foundation_ts.models.tsmoe import TSMOE


def _build_model(args: argparse.Namespace, device: torch.device) -> TSMOE:
    model = TSMOE(
        hidden_size=args.hidden_size,
        n_decoder_layers=args.n_decoder_layers,
        patch=args.patch,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        num_experts=args.num_experts,
        num_expert_layers=args.num_expert_layers,
        k=args.k,
        n_head=args.n_head,
        horizons=[1, 8, 32, 64],
    )
    model.to(device)
    model.train()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Repro a failed batch on a fresh model.")
    parser.add_argument("batch_path", type=Path, help="Path to failed_batch_*.pt")
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--n-decoder-layers", type=int, default=12)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--num-expert-layers", type=int, default=1)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--patch", action="store_true")
    parser.add_argument("--patch-len", type=int, default=32)
    parser.add_argument("--patch-stride", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    payload = torch.load(args.batch_path, map_location="cpu")
    input_ids = payload["input_ids"]
    labels = payload["labels"]
    loss_masks = payload["loss_masks"]
    attention_mask = payload.get("attention_mask")

    model = _build_model(args, device)
    loss_fn = torch.nn.HuberLoss(reduction="none", delta=2.0)

    batch = {"input_ids": input_ids, "labels": labels, "loss_masks": loss_masks}
    input_ids, labels, loss_masks = _prepare_batch(batch, device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    try:
        outputs, _stats = model(input_ids, attention_mask=attention_mask)
        loss = _forecast_loss(outputs, labels, loss_masks, loss_fn)
        loss.backward()
    except Exception as exc:
        raise RuntimeError(f"Repro failed: {exc}") from exc

    print("Repro succeeded without errors.")


if __name__ == "__main__":
    main()
