"""Interfaces for dataset-specific raw identity and extraction adapters.

No real-data adapter is implemented here.  An adapter may only emit exact raw
identity mappings; label-signature, nearest-neighbour, feature-similarity, and
other guessed identity recovery strategies are outside this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ExactRawSample:
    dataset: str
    sample_id: str
    official_global_row: int
    identity_method: str
    identity_evidence: Mapping[str, Any]
    raw_image_path: Path
    text_source_items: Sequence[Mapping[str, Any]]
    label_source_row: int


@dataclass(frozen=True)
class ExtractedSample:
    raw: ExactRawSample
    image_vector: np.ndarray
    text_vector: np.ndarray
    multilabel_vector: np.ndarray
    text_model_inputs: Sequence[Mapping[str, Any]]
    text_aggregation: Mapping[str, Any]


class DatasetAdapter(ABC):
    """Exact, row-preserving dataset adapter contract.

    Implementations must obtain identities from an authoritative ID/row map.
    They must never infer identity from labels or feature similarity.
    """

    @property
    @abstractmethod
    def dataset(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def iter_exact_samples(self) -> Iterable[ExactRawSample]:
        """Yield every official global row exactly once and in row order."""

        raise NotImplementedError

    @abstractmethod
    def official_split_source(self) -> Path:
        raise NotImplementedError

    @abstractmethod
    def label_source(self) -> Path:
        raise NotImplementedError


FORBIDDEN_IDENTITY_TOKENS = (
    "label_signature",
    "label-signature",
    "similarity",
    "cosine",
    "nearest",
    "heuristic",
    "guess",
    "candidate_search",
)

ALLOWED_EXACT_IDENTITY_METHODS = frozenset(
    {
        "official_id_map",
        "official_row_map",
        "direct_source_key",
        "direct_order_preserving_map",
    }
)
