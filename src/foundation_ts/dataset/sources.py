import json
import os
import warnings
from collections import OrderedDict
from typing import Callable, Optional

import numpy as np
import torch

from foundation_ts.dataset.utils import read_file_by_extension, to_tensor

from .types import SequenceDataset


class GeneralSequenceDataset(torch.utils.data.Dataset):
    """Sequence dataset backed by a single local file."""

    def __init__(self, data_path: str, transform: Optional[Callable] = None):
        warnings.warn(
            "GeneralSequenceDataset is not recommended; prefer BinarySequenceDataset.",
            category=UserWarning,
            stacklevel=2,
        )
        self.data = read_file_by_extension(data_path)
        self.transform = transform
        self.num_tokens: int | None = None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, seq_idx: int) -> torch.Tensor:
        seq = self.data[seq_idx]
        seq = to_tensor(seq)
        if self.transform is not None:
            seq = self.transform(seq)
        return seq

    def get_num_tokens(self) -> int:
        """Return the total number of values across all sequences."""
        if self.num_tokens is None:
            self.num_tokens = sum(len(self[i]) for i in range(len(self)))
        return self.num_tokens

    def get_sequence_length_by_idx(self, seq_idx: int) -> int:
        """Return the length of a single sequence."""
        return len(self[seq_idx])

    @staticmethod
    def is_valid_path(data_path: str) -> bool:
        """Check whether a path is a supported single-file dataset."""
        if os.path.exists(data_path) and os.path.isfile(data_path):
            parts = data_path.split(".")
            if len(parts) == 0:
                return False
            suffix = parts[-1]
            if suffix in (
                "json",
                "jsonl",
                "npy",
                "npy.gz",
                "npz",
                "pkl",
                "pickle",
                "yaml",
                "yml",
            ):
                return True
            return False
        return False


class BinarySequenceDataset(torch.utils.data.Dataset):
    """Sequence dataset backed by raw binary shards described by a meta.json.

    Expects a folder containing a `meta.json` plus one or more `.bin` files. The
    `.bin` files are raw, contiguous arrays in the dtype specified by `meta.json`
    (no header). The `meta.json` provides a `files` map of shard name -> length
    (number of elements) and a `scales` list where each entry is a sequence slice
    with `offset` and `length` into the logical concatenation of all shards.

    If a scale entry includes `mean` and `std`, the returned sequence is
    de-normalized as `sequence * std + mean`.
    """

    meta_file_name = "meta.json"

    def __init__(
        self,
        data_path: str,
        transform: Optional[Callable] = None,
        use_mmap: bool = True,
        mmap_cache_size: int = 32,
    ):
        """Initialize from a shard directory and optional post-transform."""
        if not self.is_valid_path(data_path):
            raise ValueError(f"Folder {data_path} is not a valid dataset.")

        self.data_path = data_path
        self.transform = transform
        self.use_mmap = use_mmap
        self.mmap_cache_size = max(0, mmap_cache_size)
        self._mmap_cache: OrderedDict[str, np.memmap] = OrderedDict()

        meta_file_path = os.path.join(data_path, self.meta_file_name)
        with open(meta_file_path, encoding="utf-8") as file:
            self.meta_info = json.load(file)

        self.num_sequences = self.meta_info["num_sequences"]
        self.dtype = np.dtype(self.meta_info["dtype"])
        self.seq_infos = self.meta_info["scales"]

        self.file_start_idxes = []
        self.file_lengths: dict[str, int] = {}
        s_idx = 0
        for fn, length in sorted(self.meta_info["files"].items(), key=lambda x: int(x[0].split("-")[1])):
            full_path = os.path.join(data_path, fn)
            self.file_start_idxes.append((full_path, s_idx, length))
            self.file_lengths[full_path] = length
            s_idx += length
        self.num_tokens = s_idx

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, seq_idx: int) -> torch.Tensor:
        if seq_idx < 0:
            raise ValueError(f"Index out of the dataset length: {seq_idx} < 0")
        seq_info = self.seq_infos[seq_idx]
        read_info_list = self._get_read_infos_by_offset_length(seq_info["offset"], seq_info["length"])
        out = []
        for fn, offset_in_file, length in read_info_list:
            out.append(self._read_sequence_in_file(fn, offset_in_file, length))

        if len(out) == 1:
            sequence = out[0]
        else:
            sequence = torch.cat(out, dim=0)

        if "mean" in seq_info and "std" in seq_info:
            sequence = sequence * seq_info["std"] + seq_info["mean"]

        if self.transform is not None:
            sequence = self.transform(sequence)
        return sequence

    def get_num_tokens(self) -> int:
        """Return the total number of values across all shards."""
        return self.num_tokens

    def get_sequence_length_by_idx(self, seq_idx: int) -> int:
        """Return the length of a single sequence."""
        return self.seq_infos[seq_idx]["length"]

    def _get_read_infos_by_offset_length(self, offset: int, length: int) -> list[tuple[str, int, int]]:
        """Map a logical slice to shard file reads."""
        binary_read_info_list = []
        end_offset = offset + length
        for fn, start_idx, fn_length in self.file_start_idxes:
            end_idx = start_idx + fn_length
            if start_idx <= offset < end_idx:
                if end_offset <= end_idx:
                    binary_read_info_list.append((fn, offset - start_idx, length))
                    break
                binary_read_info_list.append((fn, offset - start_idx, end_idx - offset))
                length = end_offset - end_idx
                offset = end_idx
        return binary_read_info_list

    def _get_memmap(self, fn: str) -> np.memmap:
        """Return a cached memmap for a shard, evicting LRU entries as needed."""
        if fn in self._mmap_cache:
            mmap_obj = self._mmap_cache.pop(fn)
            self._mmap_cache[fn] = mmap_obj
            return mmap_obj

        mmap_obj = np.memmap(fn, dtype=self.dtype, mode="r", shape=(self.file_lengths[fn],))
        self._mmap_cache[fn] = mmap_obj
        if self.mmap_cache_size and len(self._mmap_cache) > self.mmap_cache_size:
            _, evicted = self._mmap_cache.popitem(last=False)
            try:
                evicted._mmap.close()
            except AttributeError:
                pass
        return mmap_obj

    def _read_sequence_in_file(self, fn: str, offset_in_file: int, length: int) -> torch.Tensor:
        """Read a contiguous slice from a shard file."""
        if self.use_mmap:
            mmap_obj = self._get_memmap(fn)
            view = mmap_obj[offset_in_file : offset_in_file + length]
            return torch.from_numpy(np.array(view, copy=True))

        sequence = np.empty(length, dtype=self.dtype)
        with open(fn, mode="rb", buffering=0) as file_handler:
            file_handler.seek(offset_in_file * sequence.itemsize)
            file_handler.readinto(sequence)
        return torch.from_numpy(sequence)

    @staticmethod
    def is_valid_path(data_path: str) -> bool:
        """Check whether a path is a valid binary dataset folder."""
        if (
            os.path.exists(data_path)
            and os.path.isdir(data_path)
            and os.path.exists(os.path.join(data_path, "meta.json"))
        ):
            for sub in os.listdir(data_path):
                if os.path.isfile(os.path.join(data_path, sub)) and sub.endswith(".bin"):
                    return True
        return False


def discover_sequence_datasets(
    data_path: str,
    transform: Optional[Callable] = None,
    use_mmap: bool = True,
    mmap_cache_size: int = 32,
) -> list[SequenceDataset]:
    """Discover and load datasets from a file or directory."""
    if BinarySequenceDataset.is_valid_path(data_path):
        return [
            BinarySequenceDataset(
                data_path,
                transform=transform,
                use_mmap=use_mmap,
                mmap_cache_size=mmap_cache_size,
            )
        ]
    if GeneralSequenceDataset.is_valid_path(data_path):
        return [GeneralSequenceDataset(data_path, transform=transform)]

    datasets: list[SequenceDataset] = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            fn_path = os.path.join(root, file)
            if file != BinarySequenceDataset.meta_file_name and GeneralSequenceDataset.is_valid_path(fn_path):
                general_ds = GeneralSequenceDataset(fn_path, transform=transform)
                if len(general_ds) > 0:
                    datasets.append(general_ds)
        for sub_folder in dirs:
            folder_path = os.path.join(root, sub_folder)
            if BinarySequenceDataset.is_valid_path(folder_path):
                binary_ds = BinarySequenceDataset(
                    folder_path,
                    transform=transform,
                    use_mmap=use_mmap,
                    mmap_cache_size=mmap_cache_size,
                )
                if len(binary_ds) > 0:
                    datasets.append(binary_ds)

    return datasets
