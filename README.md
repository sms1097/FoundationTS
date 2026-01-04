# FoundationTS

Minimal time-series dataset utilities with torch-first, lazy windowing.

Install

```bash
pip install -e .
```

For a regular install:

```bash
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
