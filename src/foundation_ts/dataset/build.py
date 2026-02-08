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
    pack_sequences: bool = False,
    pack_buckets: Optional[list[int]] = None,
    include_patterns: Optional[list[str]] = None,
) -> WindowedDataset:
    """Build a windowed dataset from local files or a dataset folder."""
    normalizer = resolve_normalizer(normalization_method)
    datasets = discover_sequence_datasets(
        data_path,
        transform=normalizer,
        use_mmap=use_mmap,
        mmap_cache_size=mmap_cache_size,
        include_patterns=include_patterns,
    )
    concat_dataset = ConcatSequenceDataset(datasets)
    return WindowedDataset(
        concat_dataset,
        context_length=max_length,
        prediction_length=0,
        stride=stride,
        pack_sequences=pack_sequences,
        pack_buckets=pack_buckets,
    )
