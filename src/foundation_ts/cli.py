import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from foundation_ts.models.training.config import DatasetConfig, ModelConfig, RunnerConfig, TrainingConfig
from foundation_ts.models.training.loop import train

DEBUG_PARTITIONS = ["other/m4_daily", "healthcare/hospital", "sales/dominick"]
TRAIN_PARTITION_SET = [
    "nature/beijing_air_quality",
    "nature/china_air_quality",
    "nature/era5_1998",
    "nature/era5_1999",
    "nature/era5_2000",
    "nature/era5_2001",
    "nature/era5_2002",
    "nature/era5_2003",
    "nature/era5_2004",
    "nature/era5_2005",
    "nature/era5_2006",
    "nature/era5_2007",
    "nature/cmip6_1850",
    "nature/cmip6_1900",
    "nature/cmip6_1950",
    "nature/cmip6_2000",
    "nature/cmip6_2010",
    "energy/**",
    "synthetic/**",
    "web/**",
    "sales/**",
]
PARTITION_SETS = {"debug": DEBUG_PARTITIONS, "train": TRAIN_PARTITION_SET}


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--seq-max-len", type=int, default=4096)
    parser.add_argument("--seq-stride", type=int, default=4096)
    parser.add_argument("--normalization-func", choices=["max", "zero"], default="zero")


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--n-decoder-layers", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--num-expert-layers", type=int, default=1)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--d-expert", type=int, default=None)
    parser.add_argument("--patch", action="store_true")
    parser.add_argument("--patch-len", type=int, default=32)
    parser.add_argument("--patch-stride", type=int, default=32)


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    _add_model_args(parser)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--aux-loss-weight", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--val-split", type=float, default=0.01)
    parser.add_argument("--val-max-batches", type=int, default=10)
    parser.add_argument("--ood-val-dataset-path", default=None)
    parser.add_argument("--ood-val-max-batches", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--profile-dir", default=None)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--log-perf-metrics", action="store_true", default=False)
    parser.add_argument("--mfu-peak-tflops", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", action="store_true")


def _add_download_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--partition-set", choices=sorted(PARTITION_SETS.keys()), default="train")
    parser.add_argument("--partitions", default=None)
    parser.add_argument("--time300b-dir", default="time300b_selected")
    parser.add_argument("--no-time300b", action="store_true")


def _validate_required(args: argparse.Namespace, required: list[str]) -> None:
    missing = [name for name in required if getattr(args, name) in (None, "")]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required args: {missing_str}")


def _build_train_config(args: argparse.Namespace) -> RunnerConfig:
    dataset_config = DatasetConfig(
        dataset_path=args.dataset_path,
        seq_max_len=args.seq_max_len,
        seq_stride=args.seq_stride,
        normalization_func=args.normalization_func,
    )
    model_config = ModelConfig(
        hidden_size=args.hidden_size,
        n_decoder_layers=args.n_decoder_layers,
        patch=args.patch,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        num_experts=args.num_experts,
        num_expert_layers=args.num_expert_layers,
        k=args.k,
        n_head=args.n_head,
        d_ff=args.d_ff,
        d_expert=args.d_expert,
    )
    train_config = TrainingConfig(
        model_config=model_config,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        aux_loss_weight=args.aux_loss_weight,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        grad_accum_steps=args.grad_accum_steps,
        device=args.device,
        use_amp=not args.no_amp,
        use_bf16=not args.no_bf16,
        val_split=args.val_split,
        val_max_batches=args.val_max_batches,
        ood_val_dataset_path=args.ood_val_dataset_path,
        ood_val_max_batches=args.ood_val_max_batches,
        log_every=args.log_every,
        val_every=args.val_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        resume_from_checkpoint=args.resume_checkpoint,
        profile=args.profile,
        profile_dir=args.profile_dir,
        compile=args.compile,
        log_perf_metrics=args.log_perf_metrics,
        mfu_peak_tflops=args.mfu_peak_tflops,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=(args.pin_memory and not args.no_pin_memory),
    )
    return RunnerConfig(dataset_config=dataset_config, train_config=train_config)


def _handle_download(args: argparse.Namespace) -> None:
    if not args.no_time300b:
        if args.partitions:
            partitions = [p.strip() for p in args.partitions.split(",") if p.strip()]
            if not partitions:
                raise ValueError("--partitions provided but empty after parsing.")
        else:
            partitions = PARTITION_SETS[args.partition_set]
        print(f"Downloading Time-300B partitions to {args.time300b_dir}...")

        local_dir = Path(args.time300b_dir)
        allow_patterns = [f"{p}/**" for p in partitions]
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="Maple728/Time-300B",
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=["**/*.lock"],
        )

        print("Time-300B download complete.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FoundationTS CLI.")
    subparsers = parser.add_subparsers(dest="command")

    dataset_parent = argparse.ArgumentParser(add_help=False)
    runtime_parent = argparse.ArgumentParser(add_help=False)
    _add_dataset_args(dataset_parent)
    _add_runtime_args(runtime_parent)

    data_parser = subparsers.add_parser("data", help="Data utilities.")
    data_subparsers = data_parser.add_subparsers(dest="data_command")
    download_parser = data_subparsers.add_parser("download", help="Download datasets.")
    _add_download_args(download_parser)

    train_parser = subparsers.add_parser(
        "train",
        help="Train a model.",
        parents=[dataset_parent, runtime_parent],
    )
    _add_train_args(train_parser)

    args = parser.parse_args(argv)
    if args.command == "data":
        if args.data_command == "download":
            _handle_download(args)
            return
        data_parser.print_help()
        return

    if args.command == "train":
        _validate_required(args, ["dataset_path"])
        cfg = _build_train_config(args)
        train(cfg)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
