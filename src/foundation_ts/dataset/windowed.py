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
        pack_sequences: bool = False,
        pack_buckets: list[int] | None = None,
    ):
        self.dataset = dataset
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_size = context_length + prediction_length
        self.window_size_plus_one = self.window_size + 1
        self.stride = stride if stride else self.window_size
        self.pack_sequences = pack_sequences
        self.pack_buckets = sorted(pack_buckets) if pack_buckets else None

        self.window_counts = []
        self.cumulative_counts = [0]
        self.pack_plan: list[tuple[int, list[tuple[int, int, int]]]] = []
        self.pack_bucket_indices: dict[int, list[int]] = {}

        if self.pack_sequences:
            self._build_pack_plan()
            self.num_windows = len(self.pack_plan)
        else:
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

    def _build_pack_plan(self) -> None:
        current: list[tuple[int, int, int]] = []
        current_len = 0
        target_len = self.window_size

        if self.pack_buckets:
            max_bucket = max(self.pack_buckets)
            if max_bucket > self.window_size:
                raise ValueError("pack_buckets cannot exceed context_length + prediction_length.")
        for seq_idx in range(len(self.dataset)):
            n_points = self.dataset.get_sequence_length_by_idx(seq_idx)
            usable = n_points - 1
            if usable <= 0:
                continue
            start = 0
            while start < usable:
                remaining = usable - start
                if current_len == 0:
                    if self.pack_buckets:
                        bucket_len = min(remaining, max(self.pack_buckets))
                        for size in self.pack_buckets:
                            if size >= bucket_len:
                                target_len = size
                                break
                    else:
                        target_len = self.window_size

                seg_len = min(remaining, target_len - current_len)
                if current_len + seg_len > target_len and current_len > 0:
                    self.pack_plan.append((target_len, current))
                    current = []
                    current_len = 0
                    continue
                current.append((seq_idx, start, seg_len))
                current_len += seg_len
                if current_len == target_len:
                    self.pack_plan.append((target_len, current))
                    current = []
                    current_len = 0
                start += seg_len
        if current:
            self.pack_plan.append((target_len, current))

        if self.pack_buckets:
            self.pack_bucket_indices = {size: [] for size in self.pack_buckets}
            for idx, (size, _) in enumerate(self.pack_plan):
                self.pack_bucket_indices.setdefault(size, []).append(idx)

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, seq_idx: int) -> dict[str, torch.Tensor]:
        if seq_idx < 0:
            raise ValueError(f"Index out of the dataset length: {seq_idx} < 0")

        if self.pack_sequences:
            if seq_idx >= self.num_windows:
                raise ValueError(f"Index out of the dataset length: {seq_idx} >= {self.num_windows}")
            target_len, segments = self.pack_plan[seq_idx]
            if not segments:
                raise ValueError(f"Packed sample {seq_idx} has no segments.")
            seq0 = self.dataset[segments[0][0]].to(torch.float32)
            token_shape = seq0.shape[1:] if seq0.dim() > 1 else ()
            input_ids = torch.zeros((target_len, *token_shape), dtype=seq0.dtype)
            labels = torch.zeros((target_len, *token_shape), dtype=seq0.dtype)
            loss_mask = torch.zeros(target_len, dtype=torch.int32)
            segment_ids = torch.full((target_len,), -1, dtype=torch.int32)
            cursor = 0
            for seg_id, (dataset_idx, start, seg_len) in enumerate(segments):
                seq = self.dataset[dataset_idx].to(torch.float32)
                seq_slice = seq[start : start + seg_len + 1]
                input_ids[cursor : cursor + seg_len] = seq_slice[:-1]
                labels[cursor : cursor + seg_len] = seq_slice[1:]
                loss_mask[cursor : cursor + seg_len] = 1
                segment_ids[cursor : cursor + seg_len] = seg_id
                cursor += seg_len
            return {
                "input_ids": input_ids,
                "labels": labels,
                "loss_masks": loss_mask,
                "segment_ids": segment_ids,
                "pack_bucket": torch.tensor(target_len, dtype=torch.int32),
            }

        if seq_idx >= self.cumulative_counts[-1]:
            raise ValueError(f"Index out of the dataset length: {seq_idx} >= {self.cumulative_counts[-1]}")

        dataset_idx = binary_search(self.cumulative_counts, seq_idx)
        local_idx = seq_idx - self.cumulative_counts[dataset_idx]
        offset = 0 if local_idx == 0 else local_idx * self.stride

        seq = self.dataset[dataset_idx]
        seq = seq[offset : offset + self.window_size_plus_one].to(torch.float32)

        seq_len = len(seq)
        if seq_len == self.window_size_plus_one:
            loss_mask = torch.ones(self.window_size, dtype=torch.int32)
        else:
            token_shape = seq.shape[1:] if seq.dim() > 1 else ()
            padded = torch.zeros((self.window_size_plus_one, *token_shape), dtype=seq.dtype)
            padded[:seq_len] = seq
            seq = padded
            loss_mask = torch.zeros(self.window_size, dtype=torch.int32)
            if seq_len > 1:
                loss_mask[: seq_len - 1] = 1

        return {
            "input_ids": seq[:-1],
            "labels": seq[1:],
            "loss_masks": loss_mask
        }
