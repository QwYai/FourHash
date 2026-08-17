"""Reference-compiled merging for two frozen Hamming indexes.

Codes use the bipolar alphabet ``{-1, +1}``. A fixed reference bank is
summarized once by its first- and second-order bit statistics. At query time,
those statistics recover the exact population mean and variance of the
query-to-reference radii without scanning the bank.

The optional shell key refines only native Hamming ties. It never changes a
strict comparison made by the reference-standardized radius tables.
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


def _strict_float32_table(raw: np.ndarray) -> np.ndarray:
    """Round a decreasing score table to float32 without coalescing radii."""

    table = np.asarray(raw, dtype=np.float64).astype(np.float32)
    if table.ndim != 1 or table.size < 2 or not np.isfinite(table).all():
        raise ValueError("radius table must be a finite vector")
    for radius in range(1, table.size):
        if not table[radius - 1] > table[radius]:
            table[radius] = np.nextafter(
                table[radius - 1], np.float32(-np.inf), dtype=np.float32
            )
    if not np.all(table[:-1] > table[1:]):
        raise AssertionError("strict float32 projection failed")
    return table


def _table_from_population(bits: int, mean: float, variance: float) -> np.ndarray:
    radii = np.arange(bits + 1, dtype=np.float64)
    if variance == 0.0:
        return _strict_float32_table(-radii / float(bits))
    return _strict_float32_table((mean - radii) / np.sqrt(variance))


def radius_table(query: np.ndarray, state: ReferenceMoments) -> np.ndarray:
    """Build a strict float32, higher-is-better table for radii ``0, ..., B``."""

    mean, variance = radius_population(query, state)
    return _table_from_population(state.bits, mean, variance)


def dual_radius_tables(
    query: np.ndarray,
    image_state: ReferenceMoments,
    text_state: ReferenceMoments,
) -> tuple[np.ndarray, np.ndarray]:
    """Build both stream tables with one shared zero-variance fallback.

    If either reference distribution is degenerate, both streams use the same
    normalized raw Hamming score. This avoids inventing a cross-stream scale
    from an epsilon chosen at serving time.
    """

    if image_state.bits != text_state.bits:
        raise ValueError("both reference banks must use the same code length")
    image_mean, image_variance = radius_population(query, image_state)
    text_mean, text_variance = radius_population(query, text_state)
    if image_variance == 0.0 or text_variance == 0.0:
        shared = _table_from_population(image_state.bits, 0.0, 0.0)
        return shared.copy(), shared.copy()
    return (
        _table_from_population(image_state.bits, image_mean, image_variance),
        _table_from_population(text_state.bits, text_mean, text_variance),
    )


def csls_integer_key(
    radius: np.ndarray | Sequence[int],
    candidate_topk_radius_sum: np.ndarray | Sequence[int],
    k: int = 10,
) -> np.ndarray:
    """Return the exact integer CSLS key ``-2 K d(q,g) + sum_K d(g,b)``.

    This key may reorder candidates and is therefore intended only as a
    secondary key inside a single native ``(stream, radius)`` shell.
    """

    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    radii = np.asarray(radius, dtype=np.int64)
    bank_sum = np.asarray(candidate_topk_radius_sum, dtype=np.int64)
    if radii.shape != bank_sum.shape:
        raise ValueError("radius and candidate bank sums must have matching shapes")
    return -2 * k * radii + bank_sum


def shell_composite_key(
    primary_shell_scores: np.ndarray,
    candidate_shell_ids: np.ndarray | Sequence[int],
    secondary_keys: np.ndarray | Sequence[int],
) -> np.ndarray:
    """Compile RZ primary scores and shell-restricted secondary keys.

    ``primary_shell_scores`` gives one float32 score per possible native
    shell. An exact primary-score group is refined only when it contains one
    occupied shell. If multiple occupied shells coalesce to the same primary
    value, the entire group remains tied.
    """

    primary = np.asarray(primary_shell_scores, dtype=np.float32)
    shell = np.asarray(candidate_shell_ids, dtype=np.int64)
    secondary = np.asarray(secondary_keys, dtype=np.int64)
    if primary.ndim != 1 or primary.size == 0 or not np.isfinite(primary).all():
        raise ValueError("primary_shell_scores must be a nonempty finite vector")
    if shell.ndim != 1 or secondary.shape != shell.shape or shell.size == 0:
        raise ValueError("candidate shell IDs and secondary keys must match")
    if np.any((shell < 0) | (shell >= primary.size)):
        raise ValueError("candidate shell IDs fall outside the primary table")

    primary = primary.copy()
    primary[primary == np.float32(0.0)] = np.float32(0.0)
    _, shell_group = np.unique(primary, return_inverse=True)
    occupied_shell = np.bincount(shell, minlength=primary.size) > 0
    occupied_group_count = np.bincount(
        shell_group[occupied_shell], minlength=int(shell_group.max()) + 1
    )
    candidate_group = shell_group[shell]
    secondary_offset = secondary - int(secondary.min())
    stride = int(secondary_offset.max()) + 1
    use_secondary = occupied_group_count[candidate_group] == 1
    return candidate_group.astype(np.int64) * stride + np.where(
        use_secondary, secondary_offset, 0
    )


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


__all__ = [
    "ReferenceMoments",
    "compile_reference",
    "csls_integer_key",
    "dual_radius_tables",
    "merge_two_lists",
    "radius_population",
    "radius_table",
    "shell_composite_key",
]
