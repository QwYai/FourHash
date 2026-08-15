"""Reference-compiled score standardization for two Hamming indexes.

Codes use the bipolar alphabet {-1, +1}.  A fixed reference bank is
summarized once by its first- and second-order bit statistics.  At query time,
those statistics recover the exact population mean and variance of the
query-to-reference Hamming radii without scanning the bank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ReferenceMoments:
    """Compiled sufficient statistics for one reference bank."""

    size: int
    bits: int
    bit_sum: np.ndarray
    gram: np.ndarray


def _bipolar(array: np.ndarray, *, ndim: int) -> np.ndarray:
    value = np.asarray(array, dtype=np.int64)
    if value.ndim != ndim:
        raise ValueError(f"expected a {ndim}-D array")
    if not np.all((value == -1) | (value == 1)):
        raise ValueError("codes must use the bipolar alphabet {-1, +1}")
    return value


def compile_reference(codes: np.ndarray) -> ReferenceMoments:
    """Compile an ``n x B`` reference bank in ``O(n B^2)`` time."""

    bank = _bipolar(codes, ndim=2)
    if bank.shape[0] == 0 or bank.shape[1] == 0:
        raise ValueError("the reference bank must be non-empty")
    return ReferenceMoments(
        size=int(bank.shape[0]),
        bits=int(bank.shape[1]),
        bit_sum=bank.sum(axis=0, dtype=np.int64),
        gram=bank.T @ bank,
    )


def radius_population(query: np.ndarray, state: ReferenceMoments) -> tuple[float, float]:
    """Return the exact population mean and variance of reference radii."""

    q = _bipolar(query, ndim=1)
    if q.size != state.bits:
        raise ValueError("query and reference codes have different bit lengths")

    n = state.size
    b = state.bits
    qc = int(q @ state.bit_sum)
    c2 = n * b - qc
    q4 = n * b * b - 2 * b * qc + int(q @ state.gram @ q)
    n4 = n * q4 - c2 * c2
    mean = c2 / (2.0 * n)
    variance = n4 / (4.0 * n * n)
    return mean, max(0.0, variance)


def radius_table(query: np.ndarray, state: ReferenceMoments) -> np.ndarray:
    """Build the higher-is-better score table for radii ``0, ..., B``."""

    mean, variance = radius_population(query, state)
    radii = np.arange(state.bits + 1, dtype=np.float64)
    if variance == 0.0:
        return -radii / float(state.bits)
    return (mean - radii) / np.sqrt(variance)


def merge_two_lists(
    image_ids: Sequence[int],
    image_radii: Sequence[int],
    text_ids: Sequence[int],
    text_radii: Sequence[int],
    image_table: np.ndarray,
    text_table: np.ndarray,
    k: int,
) -> list[tuple[str, int, int, float]]:
    """Merge two local Hamming lists under their reference coordinates.

    The input order is used as the deterministic secondary key, so equal-radius
    ties inside each source remain in their native order.
    """

    if len(image_ids) != len(image_radii) or len(text_ids) != len(text_radii):
        raise ValueError("each ID list must match its radius list")
    rows: list[tuple[float, int, str, int, int]] = []
    for position, (item_id, radius) in enumerate(zip(image_ids, image_radii)):
        rows.append((float(image_table[radius]), position, "image", int(item_id), int(radius)))
    offset = len(image_ids)
    for position, (item_id, radius) in enumerate(zip(text_ids, text_radii)):
        rows.append((float(text_table[radius]), offset + position, "text", int(item_id), int(radius)))
    rows.sort(key=lambda row: (-row[0], row[1]))
    return [(stream, item_id, radius, score) for score, _, stream, item_id, radius in rows[:k]]

