"""Dataset splitting, statistics, and metadata generation."""

from .dataset_info import DatasetInfo, build_dataset_info, save_dataset_info
from .split_dataset import (
    DatasetSplitError,
    DatasetSplitter,
    DuplicateError,
    SplitConfig,
    SplitResult,
)
from .statistics import DatasetStatistics, compute_statistics, save_statistics

__all__ = [
    "DatasetInfo",
    "DatasetSplitError",
    "DatasetSplitter",
    "DatasetStatistics",
    "DuplicateError",
    "SplitConfig",
    "SplitResult",
    "build_dataset_info",
    "compute_statistics",
    "save_dataset_info",
    "save_statistics",
]
