from typing import Iterable, List

import torch

from .types import SequenceDataset
from .utils import binary_search


class ConcatSequenceDataset(torch.utils.data.Dataset):
    """Concatenate multiple sequence datasets with random-access indexing."""

    def __init__(self, datasets: Iterable[SequenceDataset]):
        self.datasets: List[SequenceDataset] = list(datasets)
        self.cumsum_lengths = [0]
        for ds in self.datasets:
            self.cumsum_lengths.append(self.cumsum_lengths[-1] + len(ds))
        self.num_sequences = self.cumsum_lengths[-1]
        self.num_tokens: int | None = None

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, seq_idx: int) -> torch.Tensor:
        if seq_idx >= self.cumsum_lengths[-1]:
            raise ValueError(
                f"Index out of the dataset length: {seq_idx} >= {self.cumsum_lengths[-1]}"
            )
        if seq_idx < 0:
            raise ValueError(f"Index out of the dataset length: {seq_idx} < 0")

        dataset_idx = binary_search(self.cumsum_lengths, seq_idx)
        dataset_offset = seq_idx - self.cumsum_lengths[dataset_idx]
        return self.datasets[dataset_idx][dataset_offset]

    def get_sequence_length_by_idx(self, seq_idx: int) -> int:
        """Return the length of a single sequence."""
        if seq_idx >= self.cumsum_lengths[-1]:
            raise ValueError(
                f"Index out of the dataset length: {seq_idx} >= {self.cumsum_lengths[-1]}"
            )
        if seq_idx < 0:
            raise ValueError(f"Index out of the dataset length: {seq_idx} < 0")

        dataset_idx = binary_search(self.cumsum_lengths, seq_idx)
        dataset_offset = seq_idx - self.cumsum_lengths[dataset_idx]
        return self.datasets[dataset_idx].get_sequence_length_by_idx(dataset_offset)

    def get_num_tokens(self) -> int:
        """Return the total number of values across all datasets."""
        if self.num_tokens is None:
            self.num_tokens = sum(ds.get_num_tokens() for ds in self.datasets)
        return self.num_tokens
