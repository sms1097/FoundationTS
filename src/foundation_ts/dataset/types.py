from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class SequenceDataset(Protocol):
    """Protocol for sequence datasets used by windowing and concatenation."""

    def __len__(self) -> int: ...

    def __getitem__(self, seq_idx: int) -> torch.Tensor: ...

    def get_sequence_length_by_idx(self, seq_idx: int) -> int: ...

    def get_num_tokens(self) -> int: ...
