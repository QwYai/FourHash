"""Traceable, resumable feature extraction for qualitative KBS experiments.

The package is deliberately independent from ``visualization_feature_pipeline``.
It binds every emitted image/text feature row to raw assets, labels, and the
frozen baseline split before it can be consumed by visualization code.
"""

from .core import (
    ContentHashSplit,
    SplitMembership,
    TraceContractError,
    TraceRow,
    ensure_output_safe,
    sha256_file,
)

__all__ = [
    "ContentHashSplit",
    "SplitMembership",
    "TraceContractError",
    "TraceRow",
    "ensure_output_safe",
    "sha256_file",
]
