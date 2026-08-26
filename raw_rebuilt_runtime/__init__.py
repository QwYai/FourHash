"""Fail-closed runtime bridge for sealed ``raw_rebuilt_v1`` trace bundles.

The bridge never reads ``ProcessData`` and never accepts legacy prepared
features.  A source bundle is admitted only after both independent validators
pass; materialized arrays remain bound to that exact source by SHA-256.
"""

from .contract import RuntimeBridgeError
from .loader import (
    LabelFreeRankInputs,
    MetricLabels,
    RuntimeDataset,
    TrainingInputs,
    load_indt_training_inputs,
    load_label_free_rank_inputs,
    load_metric_labels,
    open_runtime_dataset,
)
from .materialize import (
    materialize_runtime,
    verify_label_free_runtime_directory,
    verify_runtime_directory,
)
from .validation import SourceAdmission, admit_source_bundle

__all__ = [
    "LabelFreeRankInputs",
    "MetricLabels",
    "RuntimeBridgeError",
    "RuntimeDataset",
    "SourceAdmission",
    "TrainingInputs",
    "admit_source_bundle",
    "load_indt_training_inputs",
    "load_label_free_rank_inputs",
    "load_metric_labels",
    "materialize_runtime",
    "open_runtime_dataset",
    "verify_label_free_runtime_directory",
    "verify_runtime_directory",
]
