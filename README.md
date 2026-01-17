# FoundationTS

Minimal time-series dataset utilities with torch-first, lazy windowing.

Install (editable)

The runtime dependencies `torch`, `psutil`, and `flash-attn` must be installed in order; they are listed under the `runtime` extra in `pyproject.toml` but should be installed manually with `--no-build-isolation` for `flash-attn`.


Regular install:

```bash
pip install torch psutil
pip install --no-build-isolation flash-attn
pip install .
```

Quick start

```python
from foundation_ts.dataset import build_ts_dataset

ds = build_ts_dataset(
    data_path="time300b_selected",
    max_length=4096,
    stride=128,
    normalization_method="zero",
)

sample = ds[0]
print(sample["input_ids"].shape, sample["labels"].shape, sample["loss_masks"].shape)
```

## Setup and running

Install dependencies (editable):

```bash
pip install torch
pip install psutil
pip install --no-build-isolation flash-attn
pip install -e .
```

Download a dataset (pick one option):

Time-300B partitions (default downloader):

```bash
foundationts data download
```

Time-300B debug partitions:

```bash
foundationts data download --partition-set debug
```

Train a model:

```bash
foundationts train \
  --dataset-path time300b_selected \
  --seq-max-len 4096 \
  --seq-stride 4096 \
  --hidden-size 256 \
  --n-decoder-layers 4
```

MoE implementation selection:

```bash
foundationts train \
  --dataset-path time300b_selected \
  --moe-impl efficient
```

Optional: enable sequence packing (no cross-sequence attention).

```bash
foundationts train \
  --dataset-path time300b_selected \
  --seq-max-len 4096 \
  --seq-stride 4096 \
  --pack-sequences
```

Note: `--pack-sequences` is not supported with `--patch`.

Bucketed packing (reduce dynamic shapes when packing):

```bash
foundationts train \
  --dataset-path time300b_selected \
  --seq-max-len 4096 \
  --seq-stride 4096 \
  --pack-sequences \
  --pack-buckets 512,1024,2048,4096
```

Time-MoE paper-aligned sizes (using horizons {1,8,32,64}, k=2, E=8, input_size=1):

```bash
# TIME-MOEbase (50M active / 113M total)
foundationts train \
  --dataset-path time300b_selected \
  --seq-max-len 4096 \
  --seq-stride 4096 \
  --hidden-size 384 \
  --n-decoder-layers 12 \
  --n-head 12 \
  --num-experts 8 \
  --k 2 \
  --d-ff 1536 \
  --d-expert 192

# TIME-MOElarge (200M active / 453M total)
foundationts train \
  --dataset-path time300b_selected \
  --seq-max-len 4096 \
  --seq-stride 4096 \
  --hidden-size 768 \
  --n-decoder-layers 12 \
  --n-head 12 \
  --num-experts 8 \
  --k 2 \
  --d-ff 3072 \
  --d-expert 384

# TIME-MOEultra (1.1B active / 2.4B total)
foundationts train \
  --dataset-path time300b_selected \
  --seq-max-len 4096 \
  --seq-stride 4096 \
  --hidden-size 1024 \
  --n-decoder-layers 36 \
  --n-head 16 \
  --num-experts 8 \
  --k 2 \
  --d-ff 4096 \
  --d-expert 512
```

Paper training settings (already default unless noted):
- batch size 1024, steps 100,000, seq len 4096, horizons {1,8,32,64}
- optimizer AdamW: `--learning-rate 1e-3 --weight-decay 1e-1 --beta1 0.9 --beta2 0.95`
- aux loss: `--aux-loss-weight 0.02`
- lr schedule: warmup 10,000 then cosine decay (built-in)

Train with an out-of-distribution validation dataset (finance):

```bash
foundationts train \
  --dataset-path time300b_selected \
  --ood-val-dataset-path time300b_selected/finance \
  --val-split 0.01
```

Custom usage

```python
from foundation_ts.dataset import (
    BinarySequenceDataset,
    GeneralSequenceDataset,
    ConcatSequenceDataset,
    WindowedDataset,
)

seq_ds = GeneralSequenceDataset("data/my_sequences.jsonl")
concat = ConcatSequenceDataset([seq_ds])
windowed = WindowedDataset(concat, context_length=256, prediction_length=0, stride=64)
```

See `docs/data.md` for the on-disk sequence formats.

Docs

- `docs/data.md`: data formats, download, scalers, and windowing

### Attribution

The dataset module is heavily inspired by the Time-MoE implementation. Much of the work was direclty copied from their repo:

```
@misc{shi2024timemoe,
      title={Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts},
      author={Xiaoming Shi and Shiyu Wang and Yuqi Nie and Dianqi Li and Zhou Ye and Qingsong Wen and Ming Jin},
      year={2024},
      eprint={2409.16040},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2409.16040},
}
```
