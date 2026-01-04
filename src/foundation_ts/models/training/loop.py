import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from foundation_ts.dataset import build_ts_dataset
from foundation_ts.models.training.config import RunnerConfig
from foundation_ts.models.training.utils import (
    _build_attention_mask,
    _build_horizon_targets,
    _forecast_loss,
    _prepare_batch,
    _set_seed,
    aux_loss,
)
from foundation_ts.models.tsmoe import TSMOE


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
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress) * torch.pi)).item()

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
    total_pred = 0.0
    total_aux = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_count = 0.0
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
            pred_loss = _forecast_loss(outputs, labels, loss_masks, loss_fn)
        else:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                outputs, stats = model(input_ids, attention_mask=attention_mask)
                pred_loss = _forecast_loss(outputs, labels, loss_masks, loss_fn)

        total_pred += pred_loss.item()
        total_aux += aux_loss(stats).item()
        for horizon, preds in outputs.items():
            targets, masks = _build_horizon_targets(labels, loss_masks, horizon)
            diff = (preds - targets) * masks
            total_mae += diff.abs().sum().item()
            total_mse += (diff**2).sum().item()
            total_count += masks.sum().item()
        count += 1
        if count >= max_batches:
            break

    model.train()
    if count == 0:
        return 0.0, 0.0, 0.0, 0.0
    denom = max(1.0, total_count)
    return total_pred / count, total_aux / count, total_mae / denom, total_mse / denom


def _log_training_step(
    writer: SummaryWriter | None,
    global_step: int,
    avg_total: float,
    avg_pred: float,
    avg_aux: float,
    lr: float,
    toks_per_sec: float,
) -> None:
    if writer is None:
        return
    writer.add_scalar("train/loss", avg_total, global_step)
    writer.add_scalar("train/pred_loss", avg_pred, global_step)
    writer.add_scalar("train/aux_loss", avg_aux, global_step)
    writer.add_scalar("train/lr", lr, global_step)
    writer.add_scalar("train/toks_per_sec", toks_per_sec, global_step)


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
) -> tuple[float, float, float, int, object]:
    accum_total = 0.0
    accum_pred = 0.0
    accum_aux = 0.0
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
            pred_loss = _forecast_loss(outputs, labels, loss_masks, loss_fn)
        else:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                outputs, stats = model(input_ids, attention_mask=attention_mask)
                pred_loss = _forecast_loss(outputs, labels, loss_masks, loss_fn)

        aux = aux_loss(stats)
        total_loss = pred_loss + aux_weight * aux
        (total_loss / accum_steps).backward()

        accum_total += total_loss.item()
        accum_pred += pred_loss.item()
        accum_aux += aux.item()
        accum_tokens += input_ids.numel()
    return accum_total, accum_pred, accum_aux, accum_tokens, data_iter


def train(config: RunnerConfig) -> TSMOE:
    loss_fn = torch.nn.HuberLoss(reduction="none", delta=2.0)

    train_config = config.train_config
    model_config = train_config.model_config
    device = _get_device(train_config.device)

    _set_seed(train_config.seed)

    data_loader, val_loader, ood_val_loader = _build_dataloaders(config)
    model = _build_model(model_config, device)
    optimizer, scheduler = _build_optimizer_scheduler(model, train_config, device)
    start_step = _maybe_resume_from_checkpoint(model, optimizer, scheduler, train_config, device)

    total_steps = train_config.epochs * train_config.steps_per_epoch

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

    data_iter = iter(data_loader)
    for step_idx in range(start_step, total_steps):
        optimizer.zero_grad(set_to_none=True)
        accum_total, accum_pred, accum_aux, accum_tokens, data_iter = _train_microbatches(
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
        global_step = step_idx + 1
        tokens_since_log += accum_tokens

        avg_total = accum_total / accum_steps
        avg_pred = accum_pred / accum_steps
        avg_aux = accum_aux / accum_steps

        if train_config.log_every and global_step % train_config.log_every == 0:
            now = time.time()
            elapsed = max(1e-6, now - last_log_time)
            toks_per_sec = tokens_since_log / elapsed
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"step={global_step} loss={avg_total:.4f} "
                f"pred={avg_pred:.4f} aux={avg_aux:.4f} "
                f"lr={lr:.2e} toks/s={toks_per_sec:,.0f}"
            )
            _log_training_step(writer, global_step, avg_total, avg_pred, avg_aux, lr, toks_per_sec)
            last_log_time = now
            tokens_since_log = 0

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
