# Performance Tuning the dataset module

This includes some rough notes for improving performance of the dataset module. The primary difference is using mmap for the binary files, which results in substantial performance boost.

```
PARTITIONS = [
    "energy/electricity",
    "energy/energy_load",
    "finance/crypto_prices",
    "finance/stock_prices",
    "healthcare/hospital",
    "nature/sunspot",
    "other/m4_daily",
    "sales/favorita",
    "sales/dominick",
    "transport/traffic",
]
```

#### Before Optimizations

```
python scripts/benchmark_windowed.py \
  --data-path time300b_selected \
  --context-length 16 \
  --stride 4 \
  --batch-size 128 \
  --num-workers 4 \
  --warmup-batches 10 \
  --measure-batches 100
Dataset
  build_time: 1546.80ms
  sequences: 8613
  windows: 2648143
  window_size: 16
  stride: 4
Run 1/5
Indexing (random access)
  index: n=200 mean=2.226ms p50=0.198ms p95=0.276ms min=0.038ms max=405.194ms
Batching
  first_batch: 130.705ms
  batch: n=100 mean=0.872ms p50=0.131ms p95=2.865ms min=0.079ms max=9.780ms
Throughput
  examples_per_sec: 146,716.22
  tokens_per_sec: 2,347,459.59
Aggregate (discarded first)
  index: n=800 mean=0.128ms p50=0.142ms p95=0.171ms min=0.021ms max=0.402ms
  batch: n=400 mean=0.905ms p50=0.205ms p95=3.411ms min=0.046ms max=13.458ms
  throughput
    examples_per_sec: 141,426.16
    tokens_per_sec: 2,262,818.54
```

#### Loss Mask Optimziation

Removed use of the use `F.pad`. Instead, allocate an empty tensor and fill in with sequence.

```
Dataset
  build_time: 6.75ms
  sequences: 8613
  windows: 2648143
  window_size: 16
  stride: 4
Run 1/5
Indexing (random access)
  index: n=200 mean=0.106ms p50=0.123ms p95=0.161ms min=0.019ms max=0.407ms
Batching
  first_batch: 112.816ms
  batch: n=100 mean=0.830ms p50=0.314ms p95=2.539ms min=0.062ms max=9.972ms
Throughput
  examples_per_sec: 154,243.65
  tokens_per_sec: 2,467,898.46
Aggregate (discarded first)
  index: n=800 mean=0.144ms p50=0.134ms p95=0.182ms min=0.019ms max=8.301ms
  batch: n=400 mean=0.823ms p50=0.349ms p95=2.918ms min=0.038ms max=4.794ms
  throughput
    examples_per_sec: 155,551.73
    tokens_per_sec: 2,488,827.71
```

#### MMap in Binary Data

```
python scripts/benchmark_windowed.py \
  --data-path time300b_selected \
  --context-length 16 \
  --stride 4 \
  --batch-size 128 \
  --num-workers 4 \
  --warmup-batches 10 \
  --measure-batches 100
Dataset
  build_time: 5.94ms
  sequences: 8613
  windows: 2,648,143
  window_size: 16
  stride: 4
Run 1/5
Indexing (random access)
  index: n=200 mean=0.266ms p50=0.334ms p95=0.484ms min=0.010ms max=0.598ms
Batching
  first_batch: 133.505ms
  batch: n=100 mean=0.535ms p50=0.157ms p95=1.614ms min=0.036ms max=8.555ms
Throughput
  examples_per_sec: 239,452.99
  tokens_per_sec: 3,831,247.82
Aggregate (discarded first)
  index: n=800 mean=0.015ms p50=0.014ms p95=0.020ms min=0.010ms max=0.061ms
  batch: n=400 mean=0.454ms p50=0.167ms p95=1.541ms min=0.025ms max=2.108ms
  throughput
    examples_per_sec: 281,822.96
    tokens_per_sec: 4,509,167.31
```

#### Direct MMap Comparisons

```
python scripts/benchmark_windowed.py \
  --data-path time300b_selected \
  --context-length 16 \
  --stride 4 \
  --batch-size 128 \
  --num-workers 4 \
  --warmup-batches 10 \
  --measure-batches 100 \
  --compare-mmap
```

With MMap

```
Dataset (use_mmap=True) (cache_size=32)
  build_time: 6.44ms
  sequences: 8613
  windows: 2,648,143
  window_size: 16
  stride: 4
Aggregate (discarded first)
  index: n=800 mean=0.014ms p50=0.014ms p95=0.018ms min=0.009ms max=0.089ms
  batch: n=400 mean=0.534ms p50=0.145ms p95=1.613ms min=0.027ms max=29.520ms
  throughput
    examples_per_sec: 239,500.18
    tokens_per_sec: 3,832,002.81

Dataset (use_mmap=False)
  build_time: 1,679.55ms
  sequences: 8,613
  windows: 2,648,143
  window_size: 16
  stride: 4
Aggregate (discarded first)
  index: n=800 mean=0.145ms p50=0.151ms p95=0.189ms min=0.019ms max=1.219ms
  batch: n=400 mean=0.762ms p50=0.155ms p95=2.569ms min=0.078ms max=3.694ms
  throughput
    examples_per_sec: 168,080.38
    tokens_per_sec: 2,689,286.04

Dataset (use_mmap=True) (cache_size=0)
  build_time: 6.45ms
  sequences: 8613
  windows: 2,648,143
  window_size: 16
  stride: 4
Aggregate (discarded first)
  index: n=800 mean=0.014ms p50=0.014ms p95=0.018ms min=0.011ms max=0.138ms
  batch: n=400 mean=0.522ms p50=0.159ms p95=1.535ms min=0.026ms max=31.693ms
  throughput
    examples_per_sec: 245,409.95
    tokens_per_sec: 3,926,559.13
```

Note on caching: Cache likely does not make a difference because the OS caches on this small example. We will have to see as the dataset increases in size how the dataloader performance, but that will be down the line.
