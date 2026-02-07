import time
from pathlib import Path

import torch

from foundation_ts.models.tsmoe.layers import MOELayer
from foundation_ts.models.tsmoe.stats import MoEStats

# Hard-coded benchmark settings.
BATCH_SIZE = 128
SEQ_LEN = 4096
HIDDEN_SIZE = 512
NUM_EXPERTS = 8
TOP_K = 2
D_EXPERT = None
D_FF = None
WARMUP_ITERS = 10
MEASURE_ITERS = 50
USE_BF16 = True
RUN_BACKWARD = True
PEAK_TFLOPS = None  # Set a float value to compute MFU.
PROFILE_TRACES = True
PROFILE_DIR = Path("scripts/profiler_traces_no_mask")


def _ensure_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    return torch.device("cuda")


def _estimate_active_params(layer: torch.nn.Module) -> tuple[int, int]:
    total_params = sum(p.numel() for p in layer.parameters())
    expert_params = 0
    if hasattr(layer, "experts"):
        expert_params = sum(p.numel() for p in layer.experts.parameters())
    elif hasattr(layer, "expert_layers"):
        expert_params = sum(p.numel() for p in layer.expert_layers.parameters())
    if getattr(layer, "num_experts", 0):
        active_expert_params = expert_params * (layer.k / layer.num_experts)
    else:
        active_expert_params = 0.0
    active_params = int(round(total_params - expert_params + active_expert_params))
    return total_params, active_params


def _estimate_flops_per_token(active_params: int) -> float:
    return 12.0 * active_params


def _format_param_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def _sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _run_layer(
    name: str,
    layer: torch.nn.Module,
    x: torch.Tensor,
    attention_mask: torch.Tensor,
    use_amp: bool,
) -> None:
    device = x.device
    tokens_per_iter = x.numel() // x.shape[-1]
    stats = MoEStats.zeros(NUM_EXPERTS, device)
    total_params, active_params = _estimate_active_params(layer)
    flops_per_token = _estimate_flops_per_token(active_params)

    for _ in range(WARMUP_ITERS):
        if RUN_BACKWARD:
            layer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                y, _ = layer(x, stats, attention_mask=attention_mask)
                loss = y.sum()
        else:
            y, _ = layer(x, stats, attention_mask=attention_mask)
            loss = y.sum()
        if RUN_BACKWARD:
            loss.backward()
        _sync_if_cuda()

    _sync_if_cuda()
    if PROFILE_TRACES:
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        trace_path = PROFILE_DIR / f"{name.replace(' ', '_')}.json"
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            if RUN_BACKWARD:
                layer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    y, _ = layer(x, stats, attention_mask=attention_mask)
                    loss = y.sum()
            else:
                y, _ = layer(x, stats, attention_mask=attention_mask)
                loss = y.sum()
            if RUN_BACKWARD:
                loss.backward()
            _sync_if_cuda()
            prof.step()
        prof.export_chrome_trace(str(trace_path))
    start = time.perf_counter()
    for _ in range(MEASURE_ITERS):
        if RUN_BACKWARD:
            layer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                y, _ = layer(x, stats, attention_mask=attention_mask)
                loss = y.sum()
        else:
            y, _ = layer(x, stats, attention_mask=attention_mask)
            loss = y.sum()
        if RUN_BACKWARD:
            loss.backward()
        _sync_if_cuda()
    elapsed = time.perf_counter() - start

    toks_per_sec = (tokens_per_iter * MEASURE_ITERS) / max(elapsed, 1e-12)
    step_ms = (elapsed / MEASURE_ITERS) * 1000.0
    tflops = (toks_per_sec * flops_per_token) / 1e12
    if PEAK_TFLOPS:
        mfu = (toks_per_sec * flops_per_token) / (PEAK_TFLOPS * 1e12)
    else:
        mfu = None

    print(name)
    print(
        "  params "
        f"total={_format_param_count(total_params)} ({total_params:,}) "
        f"active={_format_param_count(active_params)} ({active_params:,})"
    )
    print(f"  toks/s={toks_per_sec:,.0f} step_ms={step_ms:.2f} tflops={tflops:.2f}")
    if mfu is not None:
        print(f"  mfu={mfu * 100:.2f}%")


def main() -> None:
    device = _ensure_cuda()
    if USE_BF16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    x = torch.randn((BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE), device=device, dtype=dtype)
    min_len = max(1, SEQ_LEN // 2)
    lengths = torch.linspace(min_len, SEQ_LEN, steps=BATCH_SIZE, device=device).round().to(torch.int32)
    positions = torch.arange(SEQ_LEN, device=device, dtype=torch.int32).unsqueeze(0)
    attention_mask = (positions < lengths.unsqueeze(1)).to(torch.int32)
    # attention_mask = None

    layers = [
        (
            "MOELayer",
            MOELayer(
                hidden_size=HIDDEN_SIZE,
                num_experts=NUM_EXPERTS,
                k=TOP_K,
                d_ff=D_FF,
                d_expert=D_EXPERT,
            ).to(device=device, dtype=dtype),
        ),
    ]

    print(
        "Benchmark settings "
        f"batch={BATCH_SIZE} seq={SEQ_LEN} hidden={HIDDEN_SIZE} "
        f"experts={NUM_EXPERTS} k={TOP_K} dtype={dtype} "
        f"backward={RUN_BACKWARD}"
    )
    for name, layer in layers:
        _run_layer(name, layer, x, attention_mask, use_amp=USE_BF16)


if __name__ == "__main__":
    main()
