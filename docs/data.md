# Data Formats

This project supports two kinds of on-disk data sources:

1. Single-file sequence datasets
2. Binary shard datasets (meta.json + .bin files)

The loader turns each _sequence_ into fixed-length windows with optional padding. A
sequence is a 1D list/array of numeric values. If a sequence is stored as a dict,
the loader expects a `sequence` key.

Single-file sequence datasets
Supported file extensions:

- `.json` (list of sequences or list of dicts with `sequence`)
- `.jsonl` (one sequence or sequence dict per line)
- `.npy`, `.npz`
- `.npy.gz`
- `.pkl`, `.pickle`
- `.yaml`, `.yml`

Example JSON (list of sequences)

```json
[
  [0.1, 0.2, 0.3, 0.4],
  [5, 6, 7, 8, 9]
]
```

Example JSON (list of dicts with `sequence`)

```json
[{ "sequence": [0.1, 0.2, 0.3] }, { "sequence": [1.0, 1.1, 1.2, 1.3] }]
```

Example JSONL (one sequence per line)

```
[0.1, 0.2, 0.3]
[4, 5, 6, 7]
```

Binary shard datasets
Folder layout:

```
my_dataset/
  meta.json
  data-1-of-1.bin
```

`meta.json` schema:

```json
{
  "num_sequences": 2,
  "dtype": "float32",
  "files": {
    "data-1-of-1.bin": 7
  },
  "scales": [
    { "offset": 0, "length": 3 },
    { "offset": 3, "length": 4 }
  ]
}
```

Interpretation:

- `files` maps each `.bin` file to its number of elements.
- Each entry in `scales` defines one sequence slice by `offset` and `length`.
- If `mean` and `std` appear in a scale, the loader will de-normalize with
  `sequence * std + mean`.
- Offsets are absolute across the concatenated `.bin` files in the order implied
  by `files` (sorted by the numeric shard index in the filename).

Binary file layout

- Each `.bin` is a raw, contiguous array of values in the dtype specified by
  `meta.json` (e.g., `float32`). There is no header inside the `.bin` file.
- The loader treats all `.bin` files as one long logical array. `files` tells the
  loader how many elements belong to each file.

Sliding window derivation
Given a sequence with length `L`, context length `C`, prediction length `P`, and
stride `S`, the total window size is `C + P + 1`.

The dataset yields windows starting at offsets:

```
0, S, 2S, 3S, ...
```

until the last window would exceed the sequence. The number of windows is:

```
1 + max(0, (L - (C + P + 1)) // S)
```

Each window produces:

- `input_ids`: the first `C + P` values
- `labels`: the last `C + P` values shifted by 1
- `loss_masks`: 1s for real values, 0s for padding

If the sequence is shorter than `C + P + 1`, it is zero-padded to fit and
`loss_masks` is padded with 0s.

How to build your own binary dataset

1. Concatenate all sequences into one long 1D array per shard file.
2. Record the per-shard element counts in `files`.
3. For each sequence, record `{offset, length}` in `scales`.
4. Optionally store `mean`/`std` in each scale to de-normalize on read.

Example: two sequences `[10, 11, 12]` and `[20, 21, 22, 23]` in one shard.

- `.bin` content (float32): `[10, 11, 12, 20, 21, 22, 23]`
- `files`: `{"data-1-of-1.bin": 7}`
- `scales`:
  - `{"offset": 0, "length": 3}`
  - `{"offset": 3, "length": 4}`

Windowing behavior
For each sequence:

- A window of size `context_length + prediction_length + 1` is sliced.
- Output: `input_ids` is all but the last element, `labels` is all but the first.
- If the sequence is too short, the window is zero-padded and `loss_masks` is set
  to 0 for padded positions.

Code examples

```python
from foundation_ts.dataset import build_ts_dataset

ds = build_ts_dataset("my_dataset", max_length=8, stride=4, normalization_method="max")
print(ds[0].keys())
```

```python
from foundation_ts.dataset import GeneralSequenceDataset, WindowedDataset

seq_ds = GeneralSequenceDataset("data/sequences.json")
win_ds = WindowedDataset(seq_ds, context_length=8, prediction_length=0, stride=4)
```

### Getting started: download data

If you want a ready-made dataset, use the script in this folder to pull the
Time-300B partitions used in the benchmarks.

```bash
foundationts data download
```

This downloads into `time300b_selected/` by default. The script uses
`huggingface_hub.snapshot_download`, so you may need to set your HF token in
the environment (e.g., `HUGGINGFACE_HUB_TOKEN`) if required by your setup.

Loading data and adding scalers
The loader can ingest either a single dataset file, a binary dataset folder,
or a directory containing many datasets. You can attach a scaler either by
passing `normalization_method` to `build_ts_dataset`, or by passing a custom
callable as `transform` to dataset classes.

Built-in scalers:

- `"max"`: scale by max absolute value
- `"zero"`: zero-mean / unit-variance

```python
import sys
sys.path.insert(0, "src")

from foundation_ts.dataset import build_ts_dataset

ds = build_ts_dataset(
    data_path="time300b_selected",
    max_length=256,
    stride=64,
    normalization_method="zero",
)
```

Custom scaler:

```python
import torch
from foundation_ts.dataset import BinarySequenceDataset, WindowedDataset

def scale_to_01(seq: torch.Tensor) -> torch.Tensor:
    seq = seq.to(torch.float32)
    min_val = seq.min()
    max_val = seq.max()
    denom = (max_val - min_val) if max_val != min_val else 1.0
    return (seq - min_val) / denom

seq_ds = BinarySequenceDataset("data/my_sequences_bin", transform=scale_to_01)
win_ds = WindowedDataset(seq_ds, context_length=128, prediction_length=0, stride=64)
```

Predictions and window shapes
Windowing produces autoregressive training samples. For a sequence slice of
length `context_length + prediction_length + 1`, the dataset returns:

- `input_ids`: all but the last point (length = `context_length + prediction_length`)
- `labels`: all but the first point (same length)
- `loss_masks`: 1 for real points, 0 for padding

For next-step prediction only, set `prediction_length=0`. For multi-step
forecasting, increase `prediction_length`; you can evaluate only the last
`prediction_length` positions of `labels` if you want a strict horizon.
