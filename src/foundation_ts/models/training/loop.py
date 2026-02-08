import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler, Subset, random_split
from torch.utils.data.distributed import DistributedSampler

from foundation_ts.dataset import build_ts_dataset
from foundation_ts.models.training.config import RunnerConfig
from foundation_ts.models.training.utils import (
    _build_attention_mask,
    _build_horizon_targets,
    _forecast_loss,
    _patch_labels_and_masks,
    _prepare_batch,
    _set_seed,
    aux_loss,
)
from foundation_ts.models.tsmoe import TSMOE
from foundation_ts.models.tsmoe.layers import MOELayer


def _get_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


class BucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        bucket_indices: dict[int, list[int]],
        batch_size: int,
        drop_last: bool,
        shuffle: bool,
        seed: int | None = None,
    ) -> None:
        self.bucket_indices = bucket_indices
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        batches: list[list[int]] = []
        for indices in self.bucket_indices.values():
            indices = list(indices)
            if self.shuffle:
                rng.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        total = 0
        for indices in self.bucket_indices.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total


def _build_dataloaders(
    config: RunnerConfig,
    *,
    ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None, DistributedSampler | None]:
    ds_config = config.dataset_config
    train_config = config.train_config
    if ds_config.pack_buckets and not ds_config.pack_sequences:
        raise ValueError("pack_buckets requires pack_sequences.")
    if ds_config.pack_sequences and train_config.model_config.patch:
        raise ValueError("pack_sequences is not supported with patching enabled.")
    if ddp and ds_config.pack_buckets:
        raise ValueError("DDP with pack_buckets is not supported.")
    ds = build_ts_dataset(
        ds_config.dataset_path,
        max_length=ds_config.seq_max_len,
        stride=ds_config.seq_stride,
        normalization_method=ds_config.normalization_func,
        pack_sequences=ds_config.pack_sequences,
        pack_buckets=ds_config.pack_buckets,
    )

    def _make_loader(dataset, shuffle: bool, drop_last: bool) -> DataLoader:
        if ds_config.pack_sequences and ds_config.pack_buckets:
            if hasattr(dataset, "pack_bucket_indices"):
                bucket_indices = dataset.pack_bucket_indices
            elif isinstance(dataset, Subset) and hasattr(dataset.dataset, "pack_plan"):
                bucket_indices = {}
                for local_idx, global_idx in enumerate(dataset.indices):
                    size = dataset.dataset.pack_plan[global_idx][0]
                    bucket_indices.setdefault(size, []).append(local_idx)
            else:
                raise ValueError("pack_buckets requested but dataset does not expose bucket indices.")
            if not bucket_indices:
                raise ValueError("pack_buckets requested but no bucket indices were built.")
            batch_sampler = BucketBatchSampler(
                bucket_indices=bucket_indices,
                batch_size=train_config.batch_size,
                drop_last=drop_last,
                shuffle=shuffle,
                seed=train_config.seed,
            )
            dl_kwargs = dict(
                batch_sampler=batch_sampler,
                num_workers=train_config.num_workers,
                pin_memory=train_config.pin_memory,
            )
            if train_config.num_workers > 0:
                dl_kwargs["prefetch_factor"] = train_config.prefetch_factor
            return DataLoader(
                dataset,
                **dl_kwargs,
            )

        dl_kwargs = dict(
            batch_size=train_config.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=train_config.num_workers,
            pin_memory=train_config.pin_memory,
        )
        if train_config.num_workers > 0:
            dl_kwargs["prefetch_factor"] = train_config.prefetch_factor
        return DataLoader(dataset, **dl_kwargs)

    if train_config.val_split > 0:
        val_size = max(1, int(len(ds) * train_config.val_split))
        train_size = len(ds) - val_size
        train_ds, val_ds = random_split(ds, [train_size, val_size])
    else:
        train_ds, val_ds = ds, None

    train_sampler = None
    if ddp:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=train_config.seed,
            drop_last=train_config.drop_last,
        )
    data_loader = _make_loader(train_ds, shuffle=not ddp, drop_last=train_config.drop_last)
    if train_sampler is not None:
        dl_kwargs = dict(
            batch_size=train_config.batch_size,
            shuffle=False,
            drop_last=train_config.drop_last,
            sampler=train_sampler,
            num_workers=train_config.num_workers,
            pin_memory=train_config.pin_memory,
        )
        if train_config.num_workers > 0:
            dl_kwargs["prefetch_factor"] = train_config.prefetch_factor
        data_loader = DataLoader(train_ds, **dl_kwargs)

    val_loader = None
    if val_ds is not None:
        val_loader = _make_loader(val_ds, shuffle=False, drop_last=False)

    ood_val_loader = None
    if train_config.ood_val_dataset_path:
        ood_ds = build_ts_dataset(
            train_config.ood_val_dataset_path,
            max_length=ds_config.seq_max_len,
            stride=ds_config.seq_stride,
            normalization_method=ds_config.normalization_func,
            pack_sequences=ds_config.pack_sequences,
            pack_buckets=ds_config.pack_buckets,
            include_patterns=train_config.ood_val_partitions,
        )
        ood_val_loader = _make_loader(ood_ds, shuffle=False, drop_last=False)

    return data_loader, val_loader, ood_val_loader, train_sampler


def _build_model(model_config, device: torch.device) -> TSMOE:
    model = TSMOE(
        hidden_size=model_config.hidden_size,
        n_decoder_layers=model_config.n_decoder_layers,
        input_size=model_config.input_size,
        patch=model_config.patch,
        patch_len=model_config.patch_len,
        patch_stride=model_config.patch_stride,
        num_experts=model_config.num_experts,
        num_expert_layers=model_config.num_expert_layers,
        k=model_config.k,
        n_head=model_config.n_head,
        attention_backend=model_config.attention_backend,
        horizons=model_config.horizons,
        d_ff=model_config.d_ff,
        d_expert=model_config.d_expert,
        moe_m_tile=model_config.moe_m_tile,
    )
    model.to(device)
    return model


def _build_optimizer_scheduler(
    model: TSMOE, train_config, device: torch.device
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(train_config.beta1, train_config.beta2),
    )
    total_steps = train_config.epochs * train_config.steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step <= 0:
            return 0.0
        if step < train_config.warmup_steps:
            return step / max(1, train_config.warmup_steps)
        progress = (step - train_config.warmup_steps) / max(1, total_steps - train_config.warmup_steps)
        return 0.5 * (1.0 + math.cos(progress * math.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def _maybe_resume_from_checkpoint(
    model: TSMOE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    train_config,
    device: torch.device,
) -> int:
    if not train_config.resume_from_checkpoint:
        return 0

    ckpt_path = Path(train_config.resume_from_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_step = int(ckpt["step"])
    total_steps = train_config.epochs * train_config.steps_per_epoch
    if start_step >= total_steps:
        raise ValueError(f"Checkpoint step {start_step} >= total_steps {total_steps}")
    if _is_main_process():
        print(f"Resumed from {ckpt_path} at step {start_step}")
    return start_step


def _save_checkpoint(
    checkpoint_dir: Path,
    step_idx: int,
    model: TSMOE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    model_config,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"step_{step_idx}.pt"
    model_state = _unwrap_model(model).state_dict()
    torch.save(
        {
            "step": step_idx,
            "model_state": model_state,
            "model_config": asdict(model_config),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        },
        ckpt_path,
    )


@torch.no_grad()
def _run_validation(
    model: TSMOE,
    val_loader: DataLoader,
    device: torch.device,
    loss_fn: torch.nn.Module,
    patch: bool,
    patch_len: int,
    patch_stride: int,
    max_batches: int = 10,
    use_bf16: bool = True,
    use_amp: bool = True,
) -> tuple[float, float, float, float]:
    model.eval()
    total_pred = torch.zeros((), device=device)
    total_aux = torch.zeros((), device=device)
    total_mae = torch.zeros((), device=device)
    total_mse = torch.zeros((), device=device)
    total_count = torch.zeros((), device=device)
    count = 0
    for batch in val_loader:
        input_ids, labels, loss_masks, segment_ids = _prepare_batch(batch, device)
        attention_mask = _build_attention_mask(loss_masks, patch, patch_len, patch_stride)
        if use_amp and use_bf16 and device.type == "cuda":
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = None

        if autocast_dtype is None:
            outputs, stats = model(input_ids, attention_mask=attention_mask, segment_ids=segment_ids)
            pred_loss = _forecast_loss(
                outputs,
                labels,
                loss_masks,
                loss_fn,
                patch=patch,
                patch_len=patch_len,
                patch_stride=patch_stride,
            )
        else:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                outputs, stats = model(input_ids, attention_mask=attention_mask, segment_ids=segment_ids)
                pred_loss = _forecast_loss(
                    outputs,
                    labels,
                    loss_masks,
                    loss_fn,
                    patch=patch,
                    patch_len=patch_len,
                    patch_stride=patch_stride,
                )

        total_pred += pred_loss.detach()
        total_aux += aux_loss(stats).detach()
        if patch:
            labels, loss_masks = _patch_labels_and_masks(labels, loss_masks, patch_len, patch_stride)
        for horizon, preds in outputs.items():
            targets, masks = _build_horizon_targets(labels, loss_masks, horizon)
            input_size = preds.size(-1) // horizon
            preds = preds.view(preds.size(0), preds.size(1), horizon, input_size)
            if targets.size(-1) == 1 and input_size > 1:
                targets = targets.expand(-1, -1, -1, input_size)
            if masks.size(-1) == 1 and input_size > 1:
                masks = masks.expand(-1, -1, -1, input_size)
            diff = (preds - targets) * masks
            total_mae += diff.abs().sum()
            total_mse += (diff**2).sum()
            total_count += masks.sum()
        count += 1
        if count >= max_batches:
            break

    model.train()
    if count == 0:
        return 0.0, 0.0, 0.0, 0.0
    if isinstance(total_count, torch.Tensor):
        total_pred = total_pred.detach().cpu()
        total_aux = total_aux.detach().cpu()
        total_mae = total_mae.detach().cpu()
        total_mse = total_mse.detach().cpu()
        total_count = total_count.detach().cpu()
    denom = max(1.0, float(total_count))
    return (
        float(total_pred) / count,
        float(total_aux) / count,
        float(total_mae) / denom,
        float(total_mse) / denom,
    )


def _format_param_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def _estimate_active_params(model: torch.nn.Module) -> tuple[int, int]:
    unwrapped = _unwrap_model(model)
    total_params = sum(p.numel() for p in unwrapped.parameters())
    expert_params = 0
    active_expert_params = 0.0
    for module in unwrapped.modules():
        if isinstance(module, MOELayer):
            layer_expert_params = sum(p.numel() for p in module.expert_layers.parameters())
            expert_params += layer_expert_params
            if module.num_experts:
                active_expert_params += layer_expert_params * (module.k / module.num_experts)
    active_params = int(round(total_params - expert_params + active_expert_params))
    return total_params, active_params


def _estimate_flops_per_token(active_params: int) -> float:
    # Rough estimate: 6 * params for forward, 6 * params for backward.
    return 12.0 * active_params


def _format_precision(train_config, device: torch.device) -> str:
    if device.type == "cuda" and train_config.use_amp and train_config.use_bf16:
        return "bf16"
    return "fp32"


def _train_microbatches(
    model: TSMOE,
    data_loader: DataLoader,
    data_iter,
    device: torch.device,
    loss_fn: torch.nn.Module,
    model_config,
    accum_steps: int,
    autocast_dtype,
    aux_weight: float,
    ddp_no_sync: Callable[[], object] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, object]:
    accum_total = torch.zeros((), device=device)
    accum_pred = torch.zeros((), device=device)
    accum_aux = torch.zeros((), device=device)
    accum_tokens = 0
    for _micro in range(accum_steps):
        ctx = ddp_no_sync() if (ddp_no_sync is not None and _micro < accum_steps - 1) else nullcontext()
        with ctx:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                batch = next(data_iter)

            input_ids, labels, loss_masks, segment_ids = _prepare_batch(batch, device)
            attention_mask = _build_attention_mask(
                loss_masks,
                model_config.patch,
                model_config.patch_len,
                model_config.patch_stride,
            )

            if autocast_dtype is None:
                outputs, stats = model(input_ids, attention_mask=attention_mask, segment_ids=segment_ids)
                pred_loss = _forecast_loss(
                    outputs,
                    labels,
                    loss_masks,
                    loss_fn,
                    patch=model_config.patch,
                    patch_len=model_config.patch_len,
                    patch_stride=model_config.patch_stride,
                )
            else:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    outputs, stats = model(input_ids, attention_mask=attention_mask, segment_ids=segment_ids)
                    pred_loss = _forecast_loss(
                        outputs,
                        labels,
                        loss_masks,
                        loss_fn,
                        patch=model_config.patch,
                        patch_len=model_config.patch_len,
                        patch_stride=model_config.patch_stride,
                    )

            aux = aux_loss(stats)
            total_loss = pred_loss + aux_weight * aux
            (total_loss / accum_steps).backward()

        accum_total += total_loss.detach()
        accum_pred += pred_loss.detach()
        accum_aux += aux.detach()
        accum_tokens += input_ids.numel()
    return (
        accum_total,
        accum_pred,
        accum_aux,
        accum_tokens,
        data_iter,
    )


def train(config: RunnerConfig) -> TSMOE:
    loss_fn = torch.nn.HuberLoss(reduction="none", delta=2.0)

    train_config = config.train_config
    model_config = train_config.model_config
    ddp_enabled = train_config.ddp
    if ddp_enabled:
        if not dist.is_available():
            raise RuntimeError("DDP requested but torch.distributed is not available.")
        if not dist.is_initialized():
            backend = train_config.ddp_backend
            if backend is None:
                backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    else:
        rank = 0
        world_size = 1
        device = _get_device(train_config.device)

    seed = train_config.seed
    if seed is not None and ddp_enabled:
        seed = seed + rank
    _set_seed(seed)

    data_loader, val_loader, ood_val_loader, train_sampler = _build_dataloaders(
        config,
        ddp=ddp_enabled,
        rank=rank,
        world_size=world_size,
    )
    model = _build_model(model_config, device)
    total_params, active_params = _estimate_active_params(model)
    flops_per_token = _estimate_flops_per_token(active_params)
    peak_flops = train_config.mfu_peak_tflops * 1e12 if train_config.mfu_peak_tflops else None
    if ddp_enabled:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                    find_unused_parameters=train_config.ddp_find_unused_parameters)
    if _is_main_process():
        print(
            "params "
            f"total={_format_param_count(total_params)} ({total_params:,}) "
            f"active={_format_param_count(active_params)} ({active_params:,})"
        )
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(device)
        else:
            gpu_name = "cpu"
        precision = _format_precision(train_config, device)
        print(f"device model={gpu_name} precision={precision}")

    optimizer, scheduler = _build_optimizer_scheduler(model, train_config, device)
    start_step = _maybe_resume_from_checkpoint(model, optimizer, scheduler, train_config, device)

    total_steps = train_config.epochs * train_config.steps_per_epoch

    model.train()
    checkpoint_dir = Path(train_config.checkpoint_dir)
    accum_steps = max(1, train_config.grad_accum_steps)
    use_amp = train_config.use_amp and train_config.use_bf16 and device.type == "cuda"
    autocast_dtype = torch.bfloat16 if use_amp else None
    data_iter = iter(data_loader)
    last_epoch = None
    run_start = time.perf_counter()
    deadline = None
    if train_config.max_wall_time_s is not None:
        if train_config.max_wall_time_s <= 0:
            print("max_wall_time_s <= 0, skipping training.")
            return model
        deadline = run_start + train_config.max_wall_time_s
        print(f"budget seconds={train_config.max_wall_time_s:.1f}")
    ema_step_s = None
    budget_exhausted = False
    last_global_step = start_step
    for step_idx in range(start_step, total_steps):
        if train_sampler is not None:
            epoch_idx = step_idx // train_config.steps_per_epoch
            if last_epoch != epoch_idx:
                train_sampler.set_epoch(epoch_idx)
                last_epoch = epoch_idx
        if deadline is not None:
            now = time.perf_counter()
            est_step = ema_step_s if ema_step_s is not None else 0.0
            if now + est_step >= deadline:
                if _is_main_process():
                    print(f"budget hit: stopping before step={step_idx + 1}")
                budget_exhausted = True
                break
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        (
            accum_total,
            accum_pred,
            accum_aux,
            accum_tokens,
            data_iter,
        ) = _train_microbatches(
            model,
            data_loader,
            data_iter,
            device,
            loss_fn,
            model_config,
            accum_steps,
            autocast_dtype,
            train_config.aux_loss_weight,
            ddp_no_sync=model.no_sync if ddp_enabled else None,
        )

        if train_config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        step_end = time.perf_counter()
        global_step = step_idx + 1
        last_global_step = global_step

        step_time = step_end - step_start
        if ema_step_s is None:
            ema_step_s = step_time
        else:
            ema_step_s = 0.9 * ema_step_s + 0.1 * step_time

        avg_total = accum_total / accum_steps
        avg_pred = accum_pred / accum_steps
        avg_aux = accum_aux / accum_steps

        if _is_main_process() and train_config.log_every and global_step % train_config.log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = max(step_end - step_start, 1e-12)
            toks_per_sec = accum_tokens / elapsed
            mfu = (toks_per_sec * flops_per_token) / peak_flops if peak_flops else None
            avg_total_val = float(avg_total)
            avg_pred_val = float(avg_pred)
            avg_aux_val = float(avg_aux)
            mfu_str = f" mfu={mfu * 100:.2f}%" if mfu is not None else ""
            print(
                f"step={global_step} loss={avg_total_val:.4f} "
                f"pred={avg_pred_val:.4f} aux={avg_aux_val:.4f} "
                f"lr={lr:.2e} toks/s={toks_per_sec:,.0f}{mfu_str}"
            )

        if deadline is not None and step_end >= deadline:
            if _is_main_process():
                print(f"budget hit: stopping after step={global_step}")
            budget_exhausted = True
            break

        if _is_main_process() and train_config.val_every and global_step % train_config.val_every == 0:
            if val_loader is not None:
                val_pred, val_aux, val_mae, val_mse = _run_validation(
                    model,
                    val_loader,
                    device,
                    loss_fn,
                    model_config.patch,
                    model_config.patch_len,
                    model_config.patch_stride,
                    max_batches=train_config.val_max_batches,
                    use_bf16=train_config.use_bf16,
                    use_amp=train_config.use_amp,
                )
                print(
                    f"val step={global_step} pred={val_pred:.4f} aux={val_aux:.4f} "
                    f"mae={val_mae:.4f} mse={val_mse:.4f}"
                )
            if ood_val_loader is not None:
                ood_pred, ood_aux, ood_mae, ood_mse = _run_validation(
                    model,
                    ood_val_loader,
                    device,
                    loss_fn,
                    model_config.patch,
                    model_config.patch_len,
                    model_config.patch_stride,
                    max_batches=train_config.ood_val_max_batches,
                    use_bf16=train_config.use_bf16,
                    use_amp=train_config.use_amp,
                )
                print(
                    f"val_ood step={global_step} pred={ood_pred:.4f} aux={ood_aux:.4f} "
                    f"mae={ood_mae:.4f} mse={ood_mse:.4f}"
                )

        if _is_main_process() and train_config.checkpoint_every and global_step % train_config.checkpoint_every == 0:
            _save_checkpoint(checkpoint_dir, global_step, model, optimizer, scheduler, model_config)
    if budget_exhausted:
        if _is_main_process() and train_config.final_val_on_budget:
            if val_loader is not None:
                val_pred, val_aux, val_mae, val_mse = _run_validation(
                    model,
                    val_loader,
                    device,
                    loss_fn,
                    model_config.patch,
                    model_config.patch_len,
                    model_config.patch_stride,
                    max_batches=train_config.val_max_batches,
                    use_bf16=train_config.use_bf16,
                    use_amp=train_config.use_amp,
                )
                print(
                    f"val step={last_global_step} pred={val_pred:.4f} aux={val_aux:.4f} "
                    f"mae={val_mae:.4f} mse={val_mse:.4f}"
                )
            if ood_val_loader is not None:
                ood_pred, ood_aux, ood_mae, ood_mse = _run_validation(
                    model,
                    ood_val_loader,
                    device,
                    loss_fn,
                    model_config.patch,
                    model_config.patch_len,
                    model_config.patch_stride,
                    max_batches=train_config.ood_val_max_batches,
                    use_bf16=train_config.use_bf16,
                    use_amp=train_config.use_amp,
                )
                print(
                    f"val_ood step={last_global_step} pred={ood_pred:.4f} aux={ood_aux:.4f} "
                    f"mae={ood_mae:.4f} mse={ood_mse:.4f}"
                )
        if _is_main_process() and train_config.final_ckpt_on_budget and last_global_step > 0:
            _save_checkpoint(checkpoint_dir, last_global_step, model, optimizer, scheduler, model_config)
    if ddp_enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return model
