import argparse
import gc
import random
import statistics
import time

import torch

from foundation_ts.dataset import build_ts_dataset


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    ordered = sorted(values)
    idx = int((pct / 100.0) * (len(ordered) - 1))
    return ordered[idx]


def _format_stats(label: str, values: list[float]) -> str:
    if not values:
        return f"{label}: n=0"
    mean_val = statistics.mean(values)
    return (
        f"{label}: n={len(values)} "
        f"mean={mean_val * 1000:.3f}ms "
        f"p50={_percentile(values, 50) * 1000:.3f}ms "
        f"p95={_percentile(values, 95) * 1000:.3f}ms "
        f"min={min(values) * 1000:.3f}ms "
        f"max={max(values) * 1000:.3f}ms"
    )


def _time_index_access(dataset, indices: list[int]) -> list[float]:
    timings = []
    for idx in indices:
        start = time.perf_counter()
        _ = dataset[idx]
        timings.append(time.perf_counter() - start)
    return timings


def _run_benchmark(dataset, args, rng: random.Random) -> dict[str, object]:
    index_population = list(range(len(dataset)))
    index_samples = min(args.index_samples + args.index_warmup, len(dataset))
    sampled = rng.sample(index_population, k=index_samples)
    warmup_indices = sampled[: args.index_warmup]
    measure_indices = sampled[args.index_warmup :]

    _ = _time_index_access(dataset, warmup_indices)
    if args.gc_collect:
        gc.collect()
    index_timings = _time_index_access(dataset, measure_indices)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=args.shuffle,
        drop_last=False,
    )

    iterator = iter(loader)
    first_start = time.perf_counter()
    try:
        _ = next(iterator)
    except StopIteration:
        return {"index_timings": index_timings, "batch_timings": []}
    first_batch_time = time.perf_counter() - first_start

    for _ in range(args.warmup_batches):
        try:
            _ = next(iterator)
        except StopIteration:
            break

    if args.gc_collect:
        gc.collect()
    batch_timings = []
    total_examples = 0
    total_tokens = 0
    for _ in range(args.measure_batches):
        start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        elapsed = time.perf_counter() - start
        batch_timings.append(elapsed)
        if isinstance(batch, dict) and "input_ids" in batch:
            batch_size = batch["input_ids"].shape[0]
            seq_len = batch["input_ids"].shape[1]
            total_examples += int(batch_size)
            total_tokens += int(batch_size * seq_len)
        else:
            total_examples += args.batch_size

    return {
        "index_timings": index_timings,
        "first_batch_time": first_batch_time,
        "batch_timings": batch_timings,
        "total_examples": total_examples,
        "total_tokens": total_tokens,
    }


def _build_dataset(args, use_mmap: bool):
    return build_ts_dataset(
        args.data_path,
        max_length=args.context_length,
        stride=args.stride,
        normalization_method=args.normalization,
        use_mmap=use_mmap,
        mmap_cache_size=args.mmap_cache_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark windowed dataset performance.")
    parser.add_argument("--data-path", required=True, help="Path to data or dataset folder.")
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--normalization", default=None, help="Normalization method or None.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--use-mmap",
        dest="use_mmap",
        action="store_true",
        help="Use mmap when reading binary datasets.",
    )
    parser.add_argument(
        "--no-mmap",
        dest="use_mmap",
        action="store_false",
        help="Disable mmap when reading binary datasets.",
    )
    parser.set_defaults(use_mmap=True)
    parser.add_argument(
        "--compare-mmap",
        action="store_true",
        help="Run the benchmark twice (mmap on/off) for binary datasets.",
    )
    parser.add_argument(
        "--mmap-cache-size",
        type=int,
        default=32,
        help="LRU cache size for mmap shards; 0 disables caching.",
    )
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--measure-batches", type=int, default=50)
    parser.add_argument("--index-samples", type=int, default=200)
    parser.add_argument("--index-warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=5, help="Repeat the benchmark.")
    parser.add_argument(
        "--discard-first",
        action="store_true",
        help="Discard the first run when reporting aggregate stats.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Set torch.set_num_threads for more stable timing.",
    )
    parser.add_argument(
        "--torch-interop-threads",
        type=int,
        default=None,
        help="Set torch.set_num_interop_threads for more stable timing.",
    )
    parser.add_argument(
        "--gc-collect",
        action="store_true",
        help="Run gc.collect() before timed sections.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    if args.torch_interop_threads is not None:
        torch.set_num_interop_threads(args.torch_interop_threads)

    use_mmap_values = [args.use_mmap]
    if args.compare_mmap:
        use_mmap_values = [True, False]

    for use_mmap in use_mmap_values:
        build_start = time.perf_counter()
        dataset = _build_dataset(args, use_mmap=use_mmap)
        build_time = time.perf_counter() - build_start

        if args.compare_mmap:
            print(f"Dataset (use_mmap={use_mmap})")
        else:
            print("Dataset")
        print(f"  build_time: {build_time * 1000:.2f}ms")
        print(f"  sequences: {len(dataset.dataset)}")
        print(f"  windows: {len(dataset)}")
        print(f"  window_size: {dataset.window_size}")
        print(f"  stride: {dataset.stride}")

        if len(dataset) == 0:
            print("No windows available; exiting.")
            return

        repeat_count = max(1, args.repeat)
        discard_first = args.discard_first or repeat_count > 1
        aggregate_index = []
        aggregate_batch = []
        total_examples = 0
        total_tokens = 0
        total_elapsed = 0.0

        for rep in range(repeat_count):
            rng = random.Random(args.seed + rep)
            torch.manual_seed(args.seed + rep)
            result = _run_benchmark(dataset, args, rng)
            if not result["batch_timings"]:
                print("Batching")
                print("  no batches available; exiting.")
                return

            print(f"Run {rep + 1}/{repeat_count}")
            print("Indexing (random access)")
            print(f"  {_format_stats('index', result['index_timings'])}")
            print("Batching")
            print(f"  first_batch: {result['first_batch_time'] * 1000:.3f}ms")
            print(f"  {_format_stats('batch', result['batch_timings'])}")

            elapsed = sum(result["batch_timings"])
            if elapsed > 0:
                print("Throughput")
                print(f"  examples_per_sec: {result['total_examples'] / elapsed:.2f}")
                if result["total_tokens"]:
                    print(f"  tokens_per_sec: {result['total_tokens'] / elapsed:.2f}")

            if discard_first and rep == 0 and repeat_count > 1:
                continue
            aggregate_index.extend(result["index_timings"])
            aggregate_batch.extend(result["batch_timings"])
            total_examples += int(result["total_examples"])
            total_tokens += int(result["total_tokens"])
            total_elapsed += elapsed

        if repeat_count > 1:
            label = "Aggregate (discarded first)" if discard_first else "Aggregate"
            print(label)
            print(f"  {_format_stats('index', aggregate_index)}")
            print(f"  {_format_stats('batch', aggregate_batch)}")
            if total_elapsed > 0:
                print("  throughput")
                print(f"    examples_per_sec: {total_examples / total_elapsed:.2f}")
                if total_tokens:
                    print(f"    tokens_per_sec: {total_tokens / total_elapsed:.2f}")


if __name__ == "__main__":
    main()
