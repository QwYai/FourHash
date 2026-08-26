"""Fair fixed-CLIP512 neural baselines for sealed ``raw_rebuilt_v1`` data."""

from .adapters import BaselineRunConfig, DEFAULT_SEEDS, METHODS, METHOD_CLAIMS
from .checkpoint import (
    BaselineCheckpoint,
    BaselineCheckpointError,
    load_checkpoint,
    train_baseline,
)
from .contract import (
    BaselineBoundaryError,
    DatasetBinding,
    LabelFreeEncodingInputs,
    build_dataset_binding,
    label_free_inputs_from_runtime,
)
from .encoding import EncodedCodes, encode_label_free, write_code_artifact

__all__ = [
    "BaselineBoundaryError",
    "BaselineCheckpoint",
    "BaselineCheckpointError",
    "BaselineRunConfig",
    "DEFAULT_SEEDS",
    "DatasetBinding",
    "EncodedCodes",
    "LabelFreeEncodingInputs",
    "METHODS",
    "METHOD_CLAIMS",
    "build_dataset_binding",
    "encode_label_free",
    "label_free_inputs_from_runtime",
    "load_checkpoint",
    "train_baseline",
    "write_code_artifact",
]
