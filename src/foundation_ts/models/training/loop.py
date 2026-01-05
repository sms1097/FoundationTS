import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from foundation_ts.dataset import build_ts_dataset
from foundation_ts.models.training.config import RunnerConfig
from foundation_ts.models.training.utils import (
    _build_attention_mask,
    _build_horizon_targets,
    _patch_labels_and_masks,
    _forecast_loss,
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


def _build_dataloaders(
    config: RunnerConfig,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    ds_config = config.dataset_config
    train_config = config.train_config
    ds = build_ts_dataset(
        ds_config.dataset_path,
        max_length=ds_config.seq_max_len,
        stride=ds_config.seq_stride,
        normalization_method=ds_config.normalization_func,
    )

    if train_config.val_split > 0:
        val_size = max(1, int(len(ds) * train_config.val_split))
        train_size = len(ds) - val_size
        train_ds, val_ds = random_split(ds, [train_size, val_size])
    else:
        train_ds, val_ds = ds, None

    dl_kwargs = dict(
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=train_config.drop_last,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
    )
    if train_config.num_workers > 0:
        dl_kwargs["prefetch_factor"] = train_config.prefetch_factor
    data_loader = DataLoader(train_ds, **dl_kwargs)

    val_loader = None
    if val_ds is not None:
        val_kwargs = dict(
            batch_size=train_config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=train_config.num_workers,
            pin_memory=train_config.pin_memory,
        )
        if train_config.num_workers > 0:
            val_kwargs["prefetch_factor"] = train_config.prefetch_factor
        val_loader = DataLoader(val_ds, **val_kwargs)

    ood_val_loader = None
    if train_config.ood_val_dataset_path:
        ood_ds = build_ts_dataset(
            train_config.ood_val_dataset_path,
            max_length=ds_config.seq_max_len,
            stride=ds_config.seq_stride,
            normalization_method=ds_config.normalization_func,
        )
        ood_kwargs = dict(
            batch_size=train_config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=train_config.num_workers,
            pin_memory=train_config.pin_memory,
        )
        if train_config.num_workers > 0:
            ood_kwargs["prefetch_factor"] = train_config.prefetch_factor
        ood_val_loader = DataLoader(ood_ds, **ood_kwargs)

    return data_loader, val_loader, ood_val_loader


def _build_model(model_config, device: torch.device) -> TSMOE:
    model = TSMOE(
        hidden_size=model_config.hidden_size,
        n_decoder_layers=model_config.n_decoder_layers,
        patch=model_config.patch,
        patch_len=model_config.patch_len,
        patch_stride=model_config.patch_stride,
        num_experts=model_config.num_experts,
        num_expert_layers=model_config.num_expert_layers,
        k=model_config.k,
        n_head=model_config.n_head,
        horizons=model_config.horizons,
    )
    model.to(device)
    return model


def _maybe_compile_model(model, train_config):
    if not train_config.compile:
        return model
    return torch.compile(model)


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
    torch.save(
        {
            "step": step_idx,
            "model_state": model.state_dict(),
            "model_config": asdict(model_config),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        },
        ckpt_path,
    )


def _build_profiler(train_config, device: torch.device, checkpoint_dir: Path):
    if not train_config.profile:
        return None
    profile_wait = 10
    profile_warmup = 10
    profile_active = 1
    profile_repeat = 1
    profile_dir = (
        Path(train_config.profile_dir)
        if train_config.profile_dir
        else checkpoint_dir / "profiler"
    )
    profile_dir.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    schedule = torch.profiler.schedule(
        wait=profile_wait,
        warmup=profile_warmup,
        active=profile_active,
        repeat=profile_repeat,
    )
    trace_handler = torch.profiler.tensorboard_trace_handler(str(profile_dir))
    chrome_trace_path = profile_dir / "chrome_trace.json"
    exported = False

    def _latest_trace_file() -> Path | None:
        candidates = list(profile_dir.rglob("*.pt.trace.json"))
        candidates += [p for p in profile_dir.rglob("*trace.json") if p.name != chrome_trace_path.name]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def on_trace_ready(prof: torch.profiler.profile) -> None:
        nonlocal exported
        trace_handler(prof)
        if exported:
            return
        latest = _latest_trace_file()
        if latest is None:
            return
        shutil.copy2(latest, chrome_trace_path)
        exported = True

    profiler = torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=on_trace_ready,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )
    return profiler


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
        input_ids, labels, loss_masks = _prepare_batch(batch, device)
        attention_mask = _build_attention_mask(loss_masks, patch, patch_len, patch_stride)
        if use_amp and use_bf16 and device.type == "cuda":
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = None

        if autocast_dtype is None:
            outputs, stats = model(input_ids, attention_mask=attention_mask)
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
                outputs, stats = model(input_ids, attention_mask=attention_mask)
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


def _log_training_step(
    writer: SummaryWriter | None,
    global_step: int,
    avg_total: float,
    avg_pred: float,
    avg_aux: float,
    lr: float,
    toks_per_sec: float,
    mem_stats: dict[str, float] | None,
) -> None:
    if writer is None:
        return
    writer.add_scalar("train/loss", avg_total, global_step)
    writer.add_scalar("train/pred_loss", avg_pred, global_step)
    writer.add_scalar("train/aux_loss", avg_aux, global_step)
    writer.add_scalar("train/lr", lr, global_step)
    writer.add_scalar("train/toks_per_sec", toks_per_sec, global_step)
    if mem_stats:
        _log_cuda_memory(writer, global_step, mem_stats)


def _log_validation(
    writer: SummaryWriter | None,
    prefix: str,
    global_step: int,
    pred: float,
    aux: float,
    mae: float,
    mse: float,
) -> None:
    if writer is None:
        return
    writer.add_scalar(f"{prefix}/pred_loss", pred, global_step)
    writer.add_scalar(f"{prefix}/aux_loss", aux, global_step)
    writer.add_scalar(f"{prefix}/mae", mae, global_step)
    writer.add_scalar(f"{prefix}/mse", mse, global_step)


def _format_param_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def _estimate_active_params(model: torch.nn.Module) -> tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    expert_params = 0
    active_expert_params = 0.0
    for module in model.modules():
        if isinstance(module, MOELayer):
            layer_expert_params = sum(p.numel() for p in module.expert_layers.parameters())
            expert_params += layer_expert_params
            if module.num_experts:
                active_expert_params += layer_expert_params * (module.k / module.num_experts)
    active_params = int(round(total_params - expert_params + active_expert_params))
    return total_params, active_params


def _bytes_to_mib(value: float) -> float:
    return value / (1024.0 * 1024.0)


def _format_cuda_memory(mem_stats: dict[str, float]) -> str:
    alloc_cur = _bytes_to_mib(mem_stats["allocated_current"])
    alloc_peak = _bytes_to_mib(mem_stats["allocated_peak"])
    alloc_min = _bytes_to_mib(mem_stats["allocated_min"])
    alloc_max = _bytes_to_mib(mem_stats["allocated_max"])
    res_cur = _bytes_to_mib(mem_stats["reserved_current"])
    res_peak = _bytes_to_mib(mem_stats["reserved_peak"])
    res_min = _bytes_to_mib(mem_stats["reserved_min"])
    res_max = _bytes_to_mib(mem_stats["reserved_max"])
    active_cur = _bytes_to_mib(mem_stats["active_current"])
    active_peak = _bytes_to_mib(mem_stats["active_peak"])
    extras = []
    if mem_stats["alloc_retries"] > 0:
        extras.append(f"alloc_retries={int(mem_stats['alloc_retries'])}")
    if mem_stats["ooms"] > 0:
        extras.append(f"ooms={int(mem_stats['ooms'])}")
    extra_str = f" {' '.join(extras)}" if extras else ""
    return (
        "gpu_mem(MiB) "
        f"alloc={alloc_cur:.0f}/{alloc_peak:.0f} min/max={alloc_min:.0f}/{alloc_max:.0f} "
        f"res={res_cur:.0f}/{res_peak:.0f} min/max={res_min:.0f}/{res_max:.0f} "
        f"active={active_cur:.0f}/{active_peak:.0f}{extra_str}"
    )


def _log_cuda_memory(writer: SummaryWriter, global_step: int, mem_stats: dict[str, float]) -> None:
    writer.add_scalar(
        "memory/allocated_current_mb", _bytes_to_mib(mem_stats["allocated_current"]), global_step
    )
    writer.add_scalar("memory/allocated_peak_mb", _bytes_to_mib(mem_stats["allocated_peak"]), global_step)
    writer.add_scalar("memory/allocated_min_mb", _bytes_to_mib(mem_stats["allocated_min"]), global_step)
    writer.add_scalar("memory/allocated_max_mb", _bytes_to_mib(mem_stats["allocated_max"]), global_step)
    writer.add_scalar("memory/reserved_current_mb", _bytes_to_mib(mem_stats["reserved_current"]), global_step)
    writer.add_scalar("memory/reserved_peak_mb", _bytes_to_mib(mem_stats["reserved_peak"]), global_step)
    writer.add_scalar("memory/reserved_min_mb", _bytes_to_mib(mem_stats["reserved_min"]), global_step)
    writer.add_scalar("memory/reserved_max_mb", _bytes_to_mib(mem_stats["reserved_max"]), global_step)
    writer.add_scalar("memory/active_current_mb", _bytes_to_mib(mem_stats["active_current"]), global_step)
    writer.add_scalar("memory/active_peak_mb", _bytes_to_mib(mem_stats["active_peak"]), global_step)
    writer.add_scalar("memory/alloc_retries", mem_stats["alloc_retries"], global_step)
    writer.add_scalar("memory/ooms", mem_stats["ooms"], global_step)


class _CudaMemoryTracker:
    def __init__(self, device: torch.device, enabled: bool) -> None:
        self.device = device
        self.enabled = enabled and device.type == "cuda"
        self.reset_interval()

    def reset_interval(self) -> None:
        if not self.enabled:
            return
        self.min_allocated = None
        self.max_allocated = 0.0
        self.min_reserved = None
        self.max_reserved = 0.0
        torch.cuda.reset_peak_memory_stats(self.device)

    def update(self) -> None:
        if not self.enabled:
            return
        allocated = float(torch.cuda.memory_allocated(self.device))
        reserved = float(torch.cuda.memory_reserved(self.device))
        if self.min_allocated is None or allocated < self.min_allocated:
            self.min_allocated = allocated
        if self.min_reserved is None or reserved < self.min_reserved:
            self.min_reserved = reserved
        if allocated > self.max_allocated:
            self.max_allocated = allocated
        if reserved > self.max_reserved:
            self.max_reserved = reserved

    def snapshot(self) -> dict[str, float] | None:
        if not self.enabled:
            return None
        stats = torch.cuda.memory_stats(self.device)
        return {
            "allocated_current": float(stats.get("allocated_bytes.all.current", 0.0)),
            "allocated_peak": float(stats.get("allocated_bytes.all.peak", 0.0)),
            "allocated_min": float(self.min_allocated or 0.0),
            "allocated_max": float(self.max_allocated),
            "reserved_current": float(stats.get("reserved_bytes.all.current", 0.0)),
            "reserved_peak": float(stats.get("reserved_bytes.all.peak", 0.0)),
            "reserved_min": float(self.min_reserved or 0.0),
            "reserved_max": float(self.max_reserved),
            "active_current": float(stats.get("active_bytes.all.current", 0.0)),
            "active_peak": float(stats.get("active_bytes.all.peak", 0.0)),
            "alloc_retries": float(stats.get("num_alloc_retries", 0.0)),
            "ooms": float(stats.get("num_ooms", 0.0)),
        }


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
    log_timers: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float, float, object]:
    accum_total = torch.zeros((), device=device)
    accum_pred = torch.zeros((), device=device)
    accum_aux = torch.zeros((), device=device)
    accum_tokens = 0
    accum_data_time = 0.0
    accum_model_time = 0.0
    for _micro in range(accum_steps):
        data_start = time.perf_counter() if log_timers else None
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)
        if log_timers:
            accum_data_time += time.perf_counter() - data_start

        input_ids, labels, loss_masks = _prepare_batch(batch, device)
        attention_mask = _build_attention_mask(
            loss_masks,
            model_config.patch,
            model_config.patch_len,
            model_config.patch_stride,
        )

        model_start = time.perf_counter() if log_timers else None
        if autocast_dtype is None:
            outputs, stats = model(input_ids, attention_mask=attention_mask)
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
                outputs, stats = model(input_ids, attention_mask=attention_mask)
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
        if log_timers:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            accum_model_time += time.perf_counter() - model_start

        accum_total += total_loss.detach()
        accum_pred += pred_loss.detach()
        accum_aux += aux.detach()
        accum_tokens += input_ids.numel()
    return (
        accum_total,
        accum_pred,
        accum_aux,
        accum_tokens,
        accum_data_time,
        accum_model_time,
        data_iter,
    )


def train(config: RunnerConfig) -> TSMOE:
    loss_fn = torch.nn.HuberLoss(reduction="none", delta=2.0)

    train_config = config.train_config
    model_config = train_config.model_config
    device = _get_device(train_config.device)

    _set_seed(train_config.seed)

    data_loader, val_loader, ood_val_loader = _build_dataloaders(config)
    model = _build_model(model_config, device)
    model = _maybe_compile_model(model, train_config)
    optimizer, scheduler = _build_optimizer_scheduler(model, train_config, device)
    start_step = _maybe_resume_from_checkpoint(model, optimizer, scheduler, train_config, device)

    total_steps = train_config.epochs * train_config.steps_per_epoch

    total_params, active_params = _estimate_active_params(model)
    print(
        "params "
        f"total={_format_param_count(total_params)} ({total_params:,}) "
        f"active={_format_param_count(active_params)} ({active_params:,})"
    )

    model.train()
    checkpoint_dir = Path(train_config.checkpoint_dir)
    last_log_time = time.time()
    tokens_since_log = 0
    writer = None
    if train_config.tensorboard:
        log_dir = (
            Path(train_config.tensorboard_dir)
            if train_config.tensorboard_dir
            else Path(train_config.checkpoint_dir) / "tensorboard"
        )
        writer = SummaryWriter(log_dir=str(log_dir))
    accum_steps = max(1, train_config.grad_accum_steps)
    use_amp = train_config.use_amp and train_config.use_bf16 and device.type == "cuda"
    autocast_dtype = torch.bfloat16 if use_amp else None
    mem_logging_enabled = bool(
        device.type == "cuda"
        and (
            train_config.log_gpu_memory_stdout
            or (writer is not None and train_config.log_gpu_memory_tensorboard)
        )
    )
    mem_tracker = _CudaMemoryTracker(device, mem_logging_enabled)

    profiler = _build_profiler(train_config, device, checkpoint_dir)
    profiler_ctx = profiler if profiler is not None else nullcontext()

    data_iter = iter(data_loader)
    with profiler_ctx:
        for step_idx in range(start_step, total_steps):
            optimizer.zero_grad(set_to_none=True)
            (
                accum_total,
                accum_pred,
                accum_aux,
                accum_tokens,
                accum_data_time,
                accum_model_time,
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
                train_config.log_timers,
            )

            if train_config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step = step_idx + 1
            tokens_since_log += accum_tokens
            mem_tracker.update()

            avg_total = accum_total / accum_steps
            avg_pred = accum_pred / accum_steps
            avg_aux = accum_aux / accum_steps

            if profiler is not None:
                profiler.step()

            if train_config.log_every and global_step % train_config.log_every == 0:
                now = time.time()
                elapsed = max(1e-6, now - last_log_time)
                toks_per_sec = tokens_since_log / elapsed
                lr = optimizer.param_groups[0]["lr"]
                mem_stats = mem_tracker.snapshot() if mem_logging_enabled else None
                mem_str = (
                    f" {_format_cuda_memory(mem_stats)}"
                    if mem_stats and train_config.log_gpu_memory_stdout
                    else ""
                )
                avg_total_val = float(avg_total)
                avg_pred_val = float(avg_pred)
                avg_aux_val = float(avg_aux)
                print(
                    f"step={global_step} loss={avg_total_val:.4f} "
                    f"pred={avg_pred_val:.4f} aux={avg_aux_val:.4f} "
                    f"lr={lr:.2e} toks/s={toks_per_sec:,.0f}{mem_str}"
                )
                _log_training_step(
                    writer,
                    global_step,
                    avg_total_val,
                    avg_pred_val,
                    avg_aux_val,
                    lr,
                    toks_per_sec,
                    mem_stats if train_config.log_gpu_memory_tensorboard else None,
                )
                if train_config.log_timers:
                    step_time = accum_data_time + accum_model_time
                    data_ms = (accum_data_time / accum_steps) * 1000.0
                    model_ms = (accum_model_time / accum_steps) * 1000.0
                    step_ms = (step_time / accum_steps) * 1000.0
                    data_pct = (accum_data_time / max(step_time, 1e-9)) * 100.0
                    print(
                        f"timing data={data_ms:.2f}ms model={model_ms:.2f}ms "
                        f"step={step_ms:.2f}ms data%={data_pct:.1f}"
                    )
                    if writer is not None:
                        writer.add_scalar("train/data_time_ms", data_ms, global_step)
                        writer.add_scalar("train/model_time_ms", model_ms, global_step)
                        writer.add_scalar("train/step_time_ms", step_ms, global_step)
                        writer.add_scalar("train/data_time_pct", data_pct, global_step)
                last_log_time = now
                tokens_since_log = 0
                mem_tracker.reset_interval()

            if train_config.val_every and global_step % train_config.val_every == 0:
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
                    _log_validation(writer, "val", global_step, val_pred, val_aux, val_mae, val_mse)
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
                    _log_validation(writer, "val_ood", global_step, ood_pred, ood_aux, ood_mae, ood_mse)

            if train_config.checkpoint_every and global_step % train_config.checkpoint_every == 0:
                _save_checkpoint(checkpoint_dir, global_step, model, optimizer, scheduler, model_config)


    if writer is not None:
        writer.close()

    return model
