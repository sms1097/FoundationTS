"""Dataset utilities inspired by Time-MoE (https://arxiv.org/abs/2409.16040)."""

from .build import build_ts_dataset
from .concat import ConcatSequenceDataset
from .sources import BinarySequenceDataset, GeneralSequenceDataset
from .windowed import WindowedDataset

__all__ = [
    "BinarySequenceDataset",
    "ConcatSequenceDataset",
    "GeneralSequenceDataset",
    "WindowedDataset",
    "build_ts_dataset",
]
