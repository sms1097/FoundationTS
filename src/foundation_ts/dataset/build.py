from typing import Callable, Optional

from foundation_ts.dataset.utils import resolve_normalizer

from .concat import ConcatSequenceDataset
from .sources import discover_sequence_datasets
from .windowed import WindowedDataset


def build_ts_dataset(
    data_path: str,
    max_length: int,
    stride: int,
    normalization_method: Optional[Callable | str] = None,
    use_mmap: bool = True,
    mmap_cache_size: int = 32,
) -> WindowedDataset:
    """Build a windowed dataset from local files or a dataset folder."""
    normalizer = resolve_normalizer(normalization_method)
    datasets = discover_sequence_datasets(
        data_path, transform=normalizer, use_mmap=use_mmap, mmap_cache_size=mmap_cache_size
    )
    concat_dataset = ConcatSequenceDataset(datasets)
    return WindowedDataset(concat_dataset, context_length=max_length, prediction_length=0, stride=stride)
