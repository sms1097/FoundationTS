# Experiment plan

## Sanity checks

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

## Single-GPU capacity + architecture probes

Goal: find the largest batch/model that fits on one GPU and decide whether pointwise or patch tokenization
is more efficient for the same model size. Use the same config for both and sweep `--batch-size` until OOM.

Model target (roughly TIME-MOE base scale for a single GPU probe):
- hidden_size 384, n_decoder_layers 12, num_experts 8, num_expert_layers 1, k 2, n_head 12

Paper reference sizes (for scale intuition):
- base: 12L/12H/8E k=2 dmodel=384 dff=1536 dexpert=192 (50M active / 113M total)
- large: 12L/12H/8E k=2 dmodel=768 dff=3072 dexpert=384 (200M active / 453M total)
- ultra: 36L/16H/8E k=2 dmodel=1024 dff=4096 dexpert=512 (1.1B active / 2.4B total)

Pointwise tokens (no patch):

```bash
foundationts train \
  --dataset-path time300b_selected \
  --steps-per-epoch 200 \
  --epochs 1 \
  --batch-size 128 \
  --seq-max-len 512 \
  --seq-stride 512 \
  --hidden-size 384 \
  --n-decoder-layers 12 \
  --num-experts 8 \
  --num-expert-layers 1 \
  --k 2 \
  --n-head 12 \
  --val-split 0 \
  --log-every 10 \
  --checkpoint-every 0
```

Patch tokens (same model, `patch_len=32`):

```bash
foundationts train \
  --dataset-path time300b_selected \
  --steps-per-epoch 200 \
  --epochs 1 \
  --batch-size 128 \
  --seq-max-len 512 \
  --seq-stride 512 \
  --hidden-size 384 \
  --n-decoder-layers 12 \
  --num-experts 8 \
  --num-expert-layers 1 \
  --k 2 \
  --n-head 12 \
  --patch \
  --patch-len 32 \
  --patch-stride 32 \
  --val-split 0 \
  --log-every 10 \
  --checkpoint-every 0
```

If both fit comfortably, increase `--batch-size` (or `--hidden-size`) and re-run until OOM. Use the new
param count printout to track total vs active params as you scale.

## Real experiments

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

