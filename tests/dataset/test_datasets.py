import json

import numpy as np
import pytest
import torch

from foundation_ts.dataset.concat import ConcatSequenceDataset
from foundation_ts.dataset.sources import (
    BinarySequenceDataset,
    GeneralSequenceDataset,
    discover_sequence_datasets,
)
from foundation_ts.dataset.windowed import WindowedDataset


class DummySequenceDataset:
    def __init__(self, sequences: list[list[int]]):
        self.sequences = [torch.as_tensor(seq) for seq in sequences]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, seq_idx: int) -> torch.Tensor:
        return self.sequences[seq_idx]

    def get_sequence_length_by_idx(self, seq_idx: int) -> int:
        return len(self.sequences[seq_idx])

    def get_num_tokens(self) -> int:
        return sum(len(seq) for seq in self.sequences)


def test_general_sequence_dataset_and_discover(tmp_path):
    data = [[1, 2, 3], [4, 5]]
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")

    dataset = GeneralSequenceDataset(str(json_path))
    assert len(dataset) == 2
    assert torch.equal(dataset[0], torch.tensor([1, 2, 3]))
    assert dataset.get_sequence_length_by_idx(1) == 2
    assert dataset.get_num_tokens() == 5

    discovered = discover_sequence_datasets(str(json_path))
    assert len(discovered) == 1
    assert isinstance(discovered[0], GeneralSequenceDataset)


def test_binary_sequence_dataset_and_discover(tmp_path):
    shard = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    shard_path = tmp_path / "shard-1-of-1.bin"
    shard.tofile(shard_path)

    meta = {
        "num_sequences": 2,
        "dtype": "float32",
        "files": {"shard-1-of-1.bin": len(shard)},
        "scales": [
            {"offset": 0, "length": 2},
            {"offset": 2, "length": 3},
        ],
    }
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    dataset = BinarySequenceDataset(str(tmp_path))
    assert len(dataset) == 2
    assert torch.equal(dataset[0], torch.tensor([1.0, 2.0], dtype=torch.float32))
    assert torch.equal(dataset[1], torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32))
    assert dataset.get_sequence_length_by_idx(1) == 3
    assert dataset.get_num_tokens() == 5

    discovered = discover_sequence_datasets(str(tmp_path))
    assert len(discovered) == 1
    assert isinstance(discovered[0], BinarySequenceDataset)


def test_binary_sequence_dataset_rejects_bad_shard_name(tmp_path):
    shard = np.array([1.0, 2.0], dtype=np.float32)
    shard_path = tmp_path / "data.bin"
    shard.tofile(shard_path)

    meta = {
        "num_sequences": 1,
        "dtype": "float32",
        "files": {"data.bin": len(shard)},
        "scales": [{"offset": 0, "length": 2}],
    }
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match="shard-i-of-n.bin"):
        BinarySequenceDataset(str(tmp_path))


def test_concat_sequence_dataset():
    ds1 = DummySequenceDataset([[1, 2], [3]])
    ds2 = DummySequenceDataset([[4, 5, 6]])
    concat = ConcatSequenceDataset([ds1, ds2])

    assert len(concat) == 3
    assert torch.equal(concat[0], torch.tensor([1, 2]))
    assert torch.equal(concat[1], torch.tensor([3]))
    assert torch.equal(concat[2], torch.tensor([4, 5, 6]))
    assert concat.get_sequence_length_by_idx(2) == 3
    assert concat.get_num_tokens() == 6


def test_windowed_dataset_basic():
    dataset = DummySequenceDataset([[0, 1, 2, 3], [10, 11]])
    windowed = WindowedDataset(dataset, context_length=2, prediction_length=1)

    assert len(windowed) == 2

    first = windowed[0]
    assert torch.equal(first["input_ids"], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(first["labels"], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(first["loss_masks"], torch.tensor([1, 1, 1], dtype=torch.int32))

    second = windowed[1]
    assert torch.equal(second["input_ids"], torch.tensor([10.0, 11.0, 0.0]))
    assert torch.equal(second["labels"], torch.tensor([11.0, 0.0, 0.0]))
    assert torch.equal(second["loss_masks"], torch.tensor([1, 0, 0], dtype=torch.int32))
