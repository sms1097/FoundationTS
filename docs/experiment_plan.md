# First experiments

Quick correctness and profiling runs (around 10 minutes):

1) Debug partition, tiny model, fast val

```bash
foundationts data download --partition-set debug
foundationts train \
  --dataset-path time300b_selected \
  --steps-per-epoch 200 \
  --epochs 1 \
  --batch-size 32 \
  --seq-max-len 512 \
  --seq-stride 512 \
  --hidden-size 64 \
  --n-decoder-layers 2 \
  --num-experts 2 \
  --num-expert-layers 1 \
  --k 1 \
  --n-head 2 \
  --val-split 0.01 \
  --val-every 50 \
  --val-max-batches 2 \
  --log-every 10 \
  --checkpoint-every 0
```

2) OOD validation sanity check (finance)

```bash
foundationts train \
  --dataset-path time300b_selected \
  --ood-val-dataset-path time300b_selected/finance \
  --steps-per-epoch 200 \
  --epochs 1 \
  --batch-size 32 \
  --seq-max-len 512 \
  --seq-stride 512 \
  --hidden-size 64 \
  --n-decoder-layers 2 \
  --num-experts 2 \
  --num-expert-layers 1 \
  --k 1 \
  --n-head 2 \
  --val-split 0.01 \
  --val-every 50 \
  --val-max-batches 2 \
  --ood-val-max-batches 2 \
  --log-every 10 \
  --checkpoint-every 0
```

Real experiments:

1) Full train partitions, mid-size model, regular checkpoints

```bash
foundationts data download --partition-set train
foundationts train \
  --dataset-path time300b_selected \
  --steps-per-epoch 10000 \
  --epochs 1 \
  --batch-size 256 \
  --seq-max-len 4096 \
  --seq-stride 4096 \
  --hidden-size 256 \
  --n-decoder-layers 4 \
  --num-experts 4 \
  --num-expert-layers 1 \
  --k 2 \
  --n-head 8 \
  --val-split 0.01 \
  --val-every 1000 \
  --val-max-batches 10 \
  --checkpoint-every 2000
```

2) Longer run with OOD validation

```bash
foundationts train \
  --dataset-path time300b_selected \
  --ood-val-dataset-path time300b_selected/finance \
  --steps-per-epoch 10000 \
  --epochs 3 \
  --batch-size 256 \
  --seq-max-len 4096 \
  --seq-stride 4096 \
  --hidden-size 256 \
  --n-decoder-layers 4 \
  --num-experts 4 \
  --num-expert-layers 1 \
  --k 2 \
  --n-head 8 \
  --val-split 0.01 \
  --val-every 1000 \
  --val-max-batches 10 \
  --ood-val-max-batches 10 \
  --checkpoint-every 2000
```

Notes:

- The Time-300B downloader uses `huggingface_hub.snapshot_download`; set
  `HUGGINGFACE_HUB_TOKEN` if your environment requires auth.
- `build_ts_dataset` can load a single file or a directory of supported
  datasets. See `docs/data.md`.
