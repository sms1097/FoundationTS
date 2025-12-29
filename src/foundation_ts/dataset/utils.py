import bisect
import gzip
import json
import pickle
from typing import Callable, Optional, Sequence

import numpy as np
import torch

"""
I/O and Indexing Utilities
"""


def binary_search(sorted_list: list[int], value: int) -> int:
    """Return the index of the largest element <= value."""
    return bisect.bisect_right(sorted_list, value) - 1


def read_file_by_extension(fn: str):
    """Load a sequence container based on file extension."""
    if fn.endswith(".json"):
        with open(fn, encoding="utf-8") as file:
            data = json.load(file)
    elif fn.endswith(".jsonl"):
        with open(fn, "r", encoding="utf-8") as file:
            data = [json.loads(line) for line in file.readlines()]
    elif fn.endswith(".yaml") or fn.endswith(".yml"):
        import yaml  # type: ignore[import-untyped]

        with open(fn, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    elif fn.endswith(".npy"):
        data = np.load(fn, allow_pickle=True)
    elif fn.endswith(".npz"):
        data = np.load(fn, allow_pickle=True)
    elif fn.endswith(".npy.gz"):
        with gzip.GzipFile(fn, "r") as file:
            data = np.load(file, allow_pickle=True)
    elif fn.endswith(".pkl") or fn.endswith(".pickle"):
        out_list = []
        with open(fn, "rb") as file:
            while True:
                try:
                    data = pickle.load(file)
                    out_list.append(data)
                except EOFError:
                    break
        if len(out_list) == 0:
            data = None
        elif len(out_list) == 1:
            data = out_list[0]
        else:
            data = out_list
    else:
        raise RuntimeError(f"Unknown file extension: {fn}")
    return data


def to_tensor(seq: Sequence | dict | np.ndarray | torch.Tensor) -> torch.Tensor:
    """Normalize an incoming sequence to a torch.Tensor."""
    if isinstance(seq, dict):
        seq = seq.get("sequence", seq)
    if torch.is_tensor(seq):
        return seq
    if isinstance(seq, np.ndarray):
        return torch.from_numpy(seq)
    return torch.as_tensor(seq)


"""
Datset Scalers
"""


def max_scaler(seq: torch.Tensor) -> torch.Tensor:
    """Scale by the maximum absolute value."""
    if not torch.is_tensor(seq):
        seq = torch.as_tensor(seq)
    seq = seq.to(torch.float32)
    max_val = seq.abs().max()
    if max_val == 0:
        return seq
    return seq / max_val


def zero_scaler(seq: torch.Tensor) -> torch.Tensor:
    """Zero-mean, unit-variance normalization."""
    if not torch.is_tensor(seq):
        seq = torch.as_tensor(seq)
    seq = seq.to(torch.float32)
    mean_val = seq.mean()
    std_val = seq.std(unbiased=False)
    if not torch.isfinite(std_val) or std_val == 0:
        return seq - mean_val
    return (seq - mean_val) / std_val


def resolve_normalizer(normalization_method: Optional[Callable | str]) -> Optional[Callable]:
    """Resolve a normalization method into a callable."""
    if normalization_method is None:
        return None
    if callable(normalization_method):
        return normalization_method
    if isinstance(normalization_method, str):
        method = normalization_method.lower()
        if method == "max":
            return max_scaler
        if method == "zero":
            return zero_scaler
    raise ValueError(f"Unknown normalization method: {normalization_method}")
