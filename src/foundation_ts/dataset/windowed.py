import torch

from .types import SequenceDataset
from .utils import binary_search


class WindowedDataset(torch.utils.data.Dataset):
    """Generate fixed-length windows from variable-length sequences."""

    def __init__(
        self,
        dataset: SequenceDataset,
        context_length: int,
        prediction_length: int = 0,
        stride: int | None = None,
    ):
        self.dataset = dataset
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_size = context_length + prediction_length
        self.window_size_plus_one = self.window_size + 1
        self.stride = stride if stride else self.window_size

        self.window_counts = []
        self.cumulative_counts = [0]

        # Determine how many sliding windows we can derive from each source sequence
        for seq_idx in range(len(self.dataset)):
            n_points = self.dataset.get_sequence_length_by_idx(seq_idx)
            if n_points < 2:
                count = 0
            else:
                extra = max(0, (n_points - self.window_size_plus_one) // self.stride)
                count = 1 + extra
            self.window_counts.append(count)
            self.cumulative_counts.append(self.cumulative_counts[-1] + count)
        self.num_windows = self.cumulative_counts[-1]

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, seq_idx: int) -> dict[str, torch.Tensor]:
        if seq_idx >= self.cumulative_counts[-1]:
            raise ValueError(f"Index out of the dataset length: {seq_idx} >= {self.cumulative_counts[-1]}")
        if seq_idx < 0:
            raise ValueError(f"Index out of the dataset length: {seq_idx} < 0")

        dataset_idx = binary_search(self.cumulative_counts, seq_idx)
        local_idx = seq_idx - self.cumulative_counts[dataset_idx]
        offset = 0 if local_idx == 0 else local_idx * self.stride

        seq = self.dataset[dataset_idx]
        seq = seq[offset : offset + self.window_size_plus_one].to(torch.float32)

        # TODO: Implement packing here
        seq_len = len(seq)
        if seq_len == self.window_size_plus_one:
            loss_mask = torch.ones(self.window_size, dtype=torch.int32)
        else:
            padded = torch.zeros(self.window_size_plus_one, dtype=seq.dtype)
            padded[:seq_len] = seq
            seq = padded
            loss_mask = torch.zeros(self.window_size, dtype=torch.int32)
            if seq_len > 1:
                loss_mask[: seq_len - 1] = 1

        return {"input_ids": seq[:-1], "labels": seq[1:], "loss_masks": loss_mask}
