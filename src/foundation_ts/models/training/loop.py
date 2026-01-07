import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
try:
    import pynvml
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None

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
        d_ff=model_config.d_ff,
        d_expert=model_config.d_expert,
    )
    model.to(device)
    return model


def _maybe_compile_model(model, train_config):
    if not train_config.compile:
        return model
    return torch.compile(model, mode="max-autotune")


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
    chrome_trace_path = profile_dir / "chrome_trace.json"
    exported = False

    def on_trace_ready(prof: torch.profiler.profile) -> None:
        nonlocal exported
        if exported:
            return
        try:
            prof.export_chrome_trace(str(chrome_trace_path))
        except Exception:
            return
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


def _build_perf_profiler(train_config, device: torch.device):
    if not train_config.log_perf_metrics or train_config.profile:
        return None
    if device.type != "cuda":
        return None
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
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


def _estimate_flops_per_token(active_params: int) -> float:
    # Rough estimate: 6 * params for forward, 6 * params for backward.
    return 12.0 * active_params


def _format_precision(train_config, device: torch.device) -> str:
    if device.type == "cuda" and train_config.use_amp and train_config.use_bf16:
        return "bf16"
    return "fp32"


def _count_cuda_events(events: list[object]) -> int:
    count = 0
    for evt in events:
        device_type = getattr(evt, "device_type", None)
        if device_type is None:
            continue
        if isinstance(device_type, str):
            is_cuda = device_type.lower() == "cuda"
        else:
            is_cuda = str(device_type).lower().endswith("cuda")
        if is_cuda:
            count += 1
    return count


def _bytes_to_gib(value: float) -> float:
    return value / (1024.0**3)


class _NvmlUtilTracker:
    def __init__(self, device: torch.device, enabled: bool) -> None:
        self.enabled = bool(enabled and device.type == "cuda" and pynvml is not None)
        self.handle = None
        if not self.enabled:
            return
        try:
            pynvml.nvmlInit()
            index = device.index if device.index is not None else 0
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception:
            self.enabled = False

    def snapshot(self) -> dict[str, float] | None:
        if not self.enabled or self.handle is None:
            return None
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            return {"sm_util": float(util.gpu), "mem_util": float(util.memory)}
        except Exception:
            return None

    def shutdown(self) -> None:
        if not self.enabled:
            return
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, object]:
    accum_total = torch.zeros((), device=device)
    accum_pred = torch.zeros((), device=device)
    accum_aux = torch.zeros((), device=device)
    accum_tokens = 0
    for _micro in range(accum_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        input_ids, labels, loss_masks = _prepare_batch(batch, device)
        attention_mask = _build_attention_mask(
            loss_masks,
            model_config.patch,
            model_config.patch_len,
            model_config.patch_stride,
        )

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
    device = _get_device(train_config.device)

    _set_seed(train_config.seed)

    data_loader, val_loader, ood_val_loader = _build_dataloaders(config)
    model = _build_model(model_config, device)
    model = _maybe_compile_model(model, train_config)
    optimizer, scheduler = _build_optimizer_scheduler(model, train_config, device)
    start_step = _maybe_resume_from_checkpoint(model, optimizer, scheduler, train_config, device)

    total_steps = train_config.epochs * train_config.steps_per_epoch

    total_params, active_params = _estimate_active_params(model)
    flops_per_token = _estimate_flops_per_token(active_params)
    peak_flops = (
        train_config.mfu_peak_tflops * 1e12 if train_config.mfu_peak_tflops else None
    )
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

    model.train()
    checkpoint_dir = Path(train_config.checkpoint_dir)
    last_log_time = time.time()
    tokens_since_log = 0
    steps_since_log = 0
    step_time_since_log = 0.0
    accum_steps = max(1, train_config.grad_accum_steps)
    use_amp = train_config.use_amp and train_config.use_bf16 and device.type == "cuda"
    autocast_dtype = torch.bfloat16 if use_amp else None
    if device.type == "cuda" and train_config.log_perf_metrics:
        torch.cuda.reset_peak_memory_stats(device)
    nvml_tracker = _NvmlUtilTracker(device, train_config.log_perf_metrics)
    peak_vram_bytes = 0.0
    kernel_launches_total = 0
    kernel_steps_total = 0

    profiler = _build_profiler(train_config, device, checkpoint_dir)
    perf_profiler = _build_perf_profiler(train_config, device)
    active_profiler = profiler or perf_profiler
    profiler_ctx = active_profiler if active_profiler is not None else nullcontext()
    kernel_profiler = active_profiler if train_config.log_perf_metrics else None
    last_kernel_event_count = 0

    data_iter = iter(data_loader)
    with profiler_ctx:
        for step_idx in range(start_step, total_steps):
            step_start = time.perf_counter() if train_config.log_perf_metrics else None
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
            )

            if train_config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            if step_start is not None:
                step_time_since_log += time.perf_counter() - step_start
            global_step = step_idx + 1
            tokens_since_log += accum_tokens
            steps_since_log += 1

            avg_total = accum_total / accum_steps
            avg_pred = accum_pred / accum_steps
            avg_aux = accum_aux / accum_steps

            if active_profiler is not None:
                active_profiler.step()

            if train_config.log_every and global_step % train_config.log_every == 0:
                now = time.time()
                elapsed = max(1e-6, now - last_log_time)
                toks_per_sec = tokens_since_log / elapsed
                lr = optimizer.param_groups[0]["lr"]
                if device.type == "cuda" and train_config.log_perf_metrics:
                    peak_vram_bytes = max(peak_vram_bytes, torch.cuda.max_memory_reserved(device))
                tflops = None
                mfu = None
                step_time_ms = None
                sm_util = None
                hbm_util = None
                mem_ctrl_util = None
                kernel_launches_per_step = None
                if train_config.log_perf_metrics:
                    tflops = (toks_per_sec * flops_per_token) / 1e12
                    if peak_flops:
                        mfu = (toks_per_sec * flops_per_token) / peak_flops
                    if steps_since_log > 0:
                        step_time_ms = (step_time_since_log / steps_since_log) * 1000.0
                    util_stats = nvml_tracker.snapshot()
                    if util_stats is not None:
                        sm_util = util_stats.get("sm_util")
                        hbm_util = util_stats.get("mem_util")
                        mem_ctrl_util = util_stats.get("mem_util")
                    if kernel_profiler is not None:
                        try:
                            events = kernel_profiler.events()
                        except Exception:
                            events = None
                        if events:
                            event_count = _count_cuda_events(events)
                            delta = event_count - last_kernel_event_count
                            if delta > 0 and steps_since_log > 0:
                                kernel_launches_per_step = delta / steps_since_log
                                kernel_launches_total += delta
                                kernel_steps_total += steps_since_log
                            last_kernel_event_count = event_count
                perf_parts = []
                if tflops is not None:
                    perf_parts.append(f"tflops={tflops:.2f}")
                if mfu is not None:
                    perf_parts.append(f"mfu={mfu * 100:.2f}%")
                if step_time_ms is not None:
                    perf_parts.append(f"step_ms={step_time_ms:.2f}")
                if sm_util is not None:
                    perf_parts.append(f"sm_util={sm_util:.1f}%")
                if hbm_util is not None:
                    perf_parts.append(f"hbm_util={hbm_util:.1f}%")
                if mem_ctrl_util is not None:
                    perf_parts.append(f"mem_ctrl_util={mem_ctrl_util:.1f}%")
                if kernel_launches_per_step is not None:
                    perf_parts.append(f"kernels/step={kernel_launches_per_step:.1f}")
                perf_str = f" {' '.join(perf_parts)}" if perf_parts else ""
                avg_total_val = float(avg_total)
                avg_pred_val = float(avg_pred)
                avg_aux_val = float(avg_aux)
                print(
                    f"step={global_step} loss={avg_total_val:.4f} "
                    f"pred={avg_pred_val:.4f} aux={avg_aux_val:.4f} "
                    f"lr={lr:.2e} toks/s={toks_per_sec:,.0f}{perf_str}"
                )
                last_log_time = now
                tokens_since_log = 0
                steps_since_log = 0
                step_time_since_log = 0.0

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

            if train_config.checkpoint_every and global_step % train_config.checkpoint_every == 0:
                _save_checkpoint(checkpoint_dir, global_step, model, optimizer, scheduler, model_config)


    if train_config.log_perf_metrics:
        if device.type == "cuda":
            peak_vram_bytes = max(peak_vram_bytes, torch.cuda.max_memory_reserved(device))
        summary_parts = [f"model={gpu_name}", f"precision={precision}"]
        if device.type == "cuda":
            summary_parts.append(f"peak_vram_gb={_bytes_to_gib(peak_vram_bytes):.2f}")
        if kernel_steps_total > 0:
            kernel_avg = kernel_launches_total / kernel_steps_total
            summary_parts.append(f"kernels/step={kernel_avg:.1f}")
        print(f"run {' '.join(summary_parts)}")

    if profiler is not None:
        sort_key = "cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        print("top_kernels")
        print(profiler.key_averages().table(sort_by=sort_key, row_limit=10))

    nvml_tracker.shutdown()
    return model
