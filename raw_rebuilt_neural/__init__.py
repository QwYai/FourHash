"""Neural experiment runner for the sealed ``raw_rebuilt_v1`` protocol.

The public API deliberately separates four process roles:

* admission creates a content-addressed, indT-only fit artifact;
* training accepts only that artifact;
* rank freezing opens label-free query/database inputs; and
* metric evaluation opens labels only after a frozen rank contract exists.
"""

from .fit_artifact import (
    FitArtifact,
    FitArtifactError,
    open_fit_artifact,
    prepare_fit_artifact,
)
from .ccde_detail_bits import (
    DetailBitArtifact,
    open_detail_bit_artifact,
    select_detail_bits_from_fit,
)
from .ccde_ranking import CCDERankFreezeConfig, freeze_ccde_ranks
from .ccde_training import load_detail_checkpoint, train_detail_from_fit_artifact
from .training import (
    DEFAULT_SEEDS,
    NeuralTrainConfig,
    load_trained_checkpoint,
    train_from_fit_artifact,
)

__all__ = [
    "CCDERankFreezeConfig",
    "DEFAULT_SEEDS",
    "DetailBitArtifact",
    "FitArtifact",
    "FitArtifactError",
    "NeuralTrainConfig",
    "freeze_ccde_ranks",
    "load_detail_checkpoint",
    "load_trained_checkpoint",
    "open_detail_bit_artifact",
    "open_fit_artifact",
    "prepare_fit_artifact",
    "select_detail_bits_from_fit",
    "train_detail_from_fit_artifact",
    "train_from_fit_artifact",
]
