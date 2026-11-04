"""Train/eval split and leakage checks. Pure functions, no I/O."""

from src.dataset.leakage import (
    LeakageError,
    assert_no_group_overlap,
    assert_no_near_duplicate_across_split,
)
from src.dataset.splitter import SplitConfig, SplitResult, split

__all__ = [
    "LeakageError",
    "SplitConfig",
    "SplitResult",
    "assert_no_group_overlap",
    "assert_no_near_duplicate_across_split",
    "split",
]
