"""Exact expected-tie binary and soft-Jaccard metrics from Hamming radii."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, localcontext
import math
from typing import Any, Callable, Sequence
import weakref

import numpy as np


_STABLE_DECIMAL_PRECISION = 80


@dataclass(frozen=True)
class _StableEstimate:
    value: Decimal
    absolute_error: Decimal
    binary64_error: Decimal


_VERIFIED_IDEAL_TOKEN = object()
_VERIFIED_IDEAL_REGISTRY: dict[int, tuple[Any, ...]] = {}


@dataclass(frozen=True)
class _VerifiedJaccardIDCG:
    """In-process certificate binding an IDCG to the exact gains object."""

    gains: np.ndarray = field(repr=False, compare=False)
    cutoffs: tuple[int, ...]
    values: tuple[tuple[int, float], ...]
    _token: object = field(repr=False, compare=False)

    def as_dict(self) -> dict[int, float]:
        return dict(self.values)


def _decimal_rounding_bound(value: Decimal, operations: int) -> Decimal:
    """Higham gamma bound for positive precision-80 Decimal arithmetic."""

    if operations < 1:
        operations = 1
    # Round-to-nearest unit roundoff for a p-digit decimal significand.
    unit_roundoff = Decimal(5).scaleb(-_STABLE_DECIMAL_PRECISION)
    product = Decimal(operations) * unit_roundoff
    if product >= 1:
        raise FloatingPointError("stable Decimal operation bound is not contractive")
    gamma = product / (Decimal(1) - product)
    return abs(value) * gamma


def _binary64_rounding_bound(value: Decimal, operations: int) -> Decimal:
    """Conservative positive-arithmetic gamma bound for the fast float path."""

    if operations < 1:
        operations = 1
    unit_roundoff = Decimal.from_float(2.0**-53)
    product = Decimal(operations) * unit_roundoff
    if product >= 1:
        raise FloatingPointError("binary64 operation bound is not contractive")
    gamma = product / (Decimal(1) - product)
    return max(abs(value), Decimal(1)) * gamma


def _canonical_unit_interval(
    value: float,
    *,
    field: str,
    stable_value: Callable[[], _StableEstimate | Decimal | float] | None = None,
) -> float:
    """Return a canonical probability, using an independent fallback on overflow.

    There is deliberately no fixed epsilon.  A fast binary64 result outside the
    mathematical interval is accepted only when an independently accumulated
    high-precision value is finite and actually lies in ``[0,1]``.  Thus a real
    formula/IDCG error remains an error regardless of how close it is to a bound.
    """

    score = float(value)
    if not np.isfinite(score):
        raise FloatingPointError(f"{field} is not finite")
    if 0.0 <= score <= 1.0:
        return score
    if stable_value is None:
        raise FloatingPointError(
            f"{field} lies outside [0,1] and has no stable fallback"
        )
    stable_result = stable_value()
    if isinstance(stable_result, _StableEstimate):
        stable = stable_result.value
        error = stable_result.absolute_error
        fast_error = stable_result.binary64_error
        if (
            not stable.is_finite()
            or not error.is_finite()
            or not fast_error.is_finite()
            or error < 0
            or fast_error < 0
        ):
            raise FloatingPointError(f"stable {field} certificate is invalid")
        if (
            abs(Decimal.from_float(score) - stable)
            > error + fast_error
        ):
            raise FloatingPointError(
                f"{field} fast/stable difference exceeds forward-error certificate"
            )
        in_interval = (
            Decimal(0) <= stable <= Decimal(1)
            or (stable < 0 and stable + error >= 0)
            or (stable > 1 and stable - error <= 1)
        )
    elif isinstance(stable_result, Decimal):
        stable = stable_result
        if not stable.is_finite():
            raise FloatingPointError(f"stable {field} is not finite")
        in_interval = Decimal(0) <= stable <= Decimal(1)
    else:
        stable = float(stable_result)
        if not np.isfinite(stable):
            raise FloatingPointError(f"stable {field} is not finite")
        in_interval = 0.0 <= stable <= 1.0
    if not in_interval:
        raise FloatingPointError(
            f"{field} lies outside [0,1] under stable recomputation"
        )
    canonical = float(stable)
    if canonical <= 0.0:
        return 0.0
    if canonical >= 1.0:
        return 1.0
    return canonical


_METRIC_PREFIX_TOKEN = object()
_METRIC_PREFIX_REGISTRY: dict[int, tuple[Any, ...]] = {}


def _register_snapshot(
    registry: dict[int, tuple[Any, ...]],
    certificate: object,
    snapshot: tuple[Any, ...],
) -> None:
    """Keep immutable construction facts outside a certificate's ``__dict__``."""

    identity = id(certificate)

    def cleanup(reference: weakref.ReferenceType[object]) -> None:
        current = registry.get(identity)
        if current is not None and current[0] is reference:
            registry.pop(identity, None)

    reference = weakref.ref(certificate, cleanup)
    registry[identity] = (reference, *snapshot)


@dataclass(frozen=True)
class MetricPrefixes:
    database_rows: int
    max_cutoff: int
    harmonic: np.ndarray
    discount: np.ndarray
    _harmonic_identity: int = field(repr=False, compare=False)
    _discount_identity: int = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)


def build_metric_prefixes(database_rows: int, cutoffs: Sequence[int]) -> MetricPrefixes:
    if type(database_rows) is not int or database_rows < 1:
        raise ValueError("database_rows must be positive")
    normalized = tuple(sorted(set(int(value) for value in cutoffs)))
    if not normalized or normalized[0] < 1:
        raise ValueError("cutoffs must be positive")
    ranks = np.arange(1, database_rows + 1, dtype=np.float64)
    harmonic = np.empty(database_rows + 1, dtype=np.float64)
    harmonic[0] = 0.0
    np.cumsum(1.0 / ranks, out=harmonic[1:])
    max_cutoff = min(database_rows, normalized[-1])
    discount = np.empty(max_cutoff + 1, dtype=np.float64)
    discount[0] = 0.0
    discount_weights = np.fromiter(
        (1.0 / math.log2(rank + 2.0) for rank in range(max_cutoff)),
        dtype=np.float64,
        count=max_cutoff,
    )
    np.cumsum(discount_weights, out=discount[1:])
    # Immutable bytes backing prevents callers from re-enabling WRITEABLE on
    # either array.  Identity fields make dataclass replacement fail closed.
    harmonic_frozen = np.frombuffer(harmonic.tobytes(order="C"), dtype=np.float64)
    discount_frozen = np.frombuffer(discount.tobytes(order="C"), dtype=np.float64)
    certificate = MetricPrefixes(
        database_rows=database_rows,
        max_cutoff=max_cutoff,
        harmonic=harmonic_frozen,
        discount=discount_frozen,
        _harmonic_identity=id(harmonic_frozen),
        _discount_identity=id(discount_frozen),
        _token=_METRIC_PREFIX_TOKEN,
    )
    _register_snapshot(
        _METRIC_PREFIX_REGISTRY,
        certificate,
        (
            harmonic_frozen,
            discount_frozen,
            database_rows,
            max_cutoff,
            _METRIC_PREFIX_TOKEN,
            id(harmonic_frozen),
            id(discount_frozen),
        ),
    )
    return certificate


def _normalized_cutoffs(cutoffs: Sequence[int]) -> tuple[int, ...]:
    values = tuple(sorted(set(int(value) for value in cutoffs)))
    if not values or values[0] < 1:
        raise ValueError("metric cutoffs must be positive")
    return values


def _precompute_jaccard_idcg(
    graded_gains: np.ndarray,
    cutoffs: Sequence[int],
) -> _VerifiedJaccardIDCG:
    """Compute one reusable, identity-bound direct-weight IDCG per query."""

    gains = np.asarray(graded_gains, dtype=np.float64)
    if gains.ndim != 1 or not len(gains):
        raise ValueError("Jaccard gains must be a nonempty vector")
    if not np.isfinite(gains).all() or np.any(gains < 0.0) or np.any(gains > 1.0):
        raise ValueError("Jaccard gains must be finite in [0,1]")
    gains = np.frombuffer(gains.tobytes(order="C"), dtype=np.float64)
    cutoff_values = _normalized_cutoffs(cutoffs)
    maximum_effective = min(cutoff_values[-1], len(gains))
    if maximum_effective == len(gains):
        ideal_prefix = np.sort(gains)[::-1]
    else:
        ideal_prefix = np.partition(
            gains, len(gains) - maximum_effective
        )[-maximum_effective:]
        ideal_prefix.sort()
        ideal_prefix = ideal_prefix[::-1]
    values = []
    for cutoff in cutoff_values:
        effective = min(cutoff, len(gains))
        discounts = (
            1.0 / math.log2(rank + 2.0) for rank in range(effective)
        )
        value = math.fsum(
            float(gain) * discount
            for gain, discount in zip(ideal_prefix[:effective], discounts)
        )
        if not math.isfinite(value) or value < 0.0:
            raise FloatingPointError("precomputed Jaccard IDCG is invalid")
        values.append((cutoff, value))
    certificate = _VerifiedJaccardIDCG(
        gains=gains,
        cutoffs=cutoff_values,
        values=tuple(values),
        _token=_VERIFIED_IDEAL_TOKEN,
    )
    _register_snapshot(
        _VERIFIED_IDEAL_REGISTRY,
        certificate,
        (gains, cutoff_values, tuple(values), _VERIFIED_IDEAL_TOKEN),
    )
    return certificate


def expected_tie_metrics_from_distances(
    relevance: np.ndarray,
    distances: np.ndarray,
    *,
    bits: int,
    graded_gains: np.ndarray,
    cutoffs: Sequence[int] = (50, 100, 1000),
    prefixes: MetricPrefixes | None = None,
    ideal_jaccard_dcg: dict[int, float] | _VerifiedJaccardIDCG | None = None,
    distance_levels: int | None = None,
) -> dict[str, float | int | bool]:
    """Integrate metrics over uniform permutations inside Hamming shells.

    ``graded_gains`` is the ground-truth label Jaccard value in ``[0,1]``.
    J-NDCG uses this value directly as a linear graded gain; no model posterior
    enters metric computation.
    """

    relevant = np.asarray(relevance, dtype=bool)
    radius = np.asarray(distances)
    gains = np.asarray(graded_gains, dtype=np.float64)
    if relevant.ndim != 1 or radius.shape != relevant.shape or gains.shape != relevant.shape:
        raise ValueError("relevance, distances, and graded_gains must align as vectors")
    if len(relevant) == 0:
        raise ValueError("at least one database row is required")
    if radius.dtype.kind not in "iu" or bits not in (16, 32, 64):
        raise ValueError("distances must be integer keys for a 16/32/64-bit cell")
    levels = bits + 1 if distance_levels is None else distance_levels
    if type(levels) is not int or levels < bits + 1:
        raise ValueError("distance_levels must contain at least the primary Hamming radii")
    if np.any(radius < 0) or np.any(radius >= levels):
        raise ValueError("integer distance key lies outside its frozen level count")
    if not np.isfinite(gains).all() or np.any(gains < 0.0) or np.any(gains > 1.0):
        raise ValueError("soft-Jaccard gains must be finite in [0,1]")
    if np.any((gains > 0.0) != relevant):
        raise ValueError("binary relevance must equal positive ground-truth Jaccard gain")
    cutoff_values = _normalized_cutoffs(cutoffs)
    prefix = prefixes or build_metric_prefixes(len(relevant), cutoff_values)
    prefix_snapshot = (
        _METRIC_PREFIX_REGISTRY.get(id(prefix))
        if isinstance(prefix, MetricPrefixes)
        else None
    )
    if (
        not isinstance(prefix, MetricPrefixes)
        or prefix_snapshot is None
        or prefix_snapshot[0]() is not prefix
        or prefix.harmonic is not prefix_snapshot[1]
        or prefix.discount is not prefix_snapshot[2]
        or prefix.database_rows != prefix_snapshot[3]
        or prefix.max_cutoff != prefix_snapshot[4]
        or prefix._token is not prefix_snapshot[5]
        or prefix_snapshot[5] is not _METRIC_PREFIX_TOKEN
        or prefix._harmonic_identity != prefix_snapshot[6]
        or prefix._discount_identity != prefix_snapshot[7]
        or prefix.database_rows != len(relevant)
        or prefix.max_cutoff < min(len(relevant), cutoff_values[-1])
        or prefix.harmonic.dtype != np.dtype(np.float64)
        or prefix.discount.dtype != np.dtype(np.float64)
        or prefix.harmonic.shape != (prefix.database_rows + 1,)
        or prefix.discount.shape != (prefix.max_cutoff + 1,)
        or not prefix.harmonic.flags.c_contiguous
        or not prefix.discount.flags.c_contiguous
        or prefix.harmonic.flags.writeable
        or prefix.discount.flags.writeable
        or not isinstance(prefix.harmonic.base, bytes)
        or not isinstance(prefix.discount.base, bytes)
        or id(prefix.harmonic) != prefix._harmonic_identity
        or id(prefix.discount) != prefix._discount_identity
    ):
        raise ValueError("metric prefix certificate or geometry differs")
    ideal_is_verified = isinstance(ideal_jaccard_dcg, _VerifiedJaccardIDCG)
    if ideal_is_verified:
        assert isinstance(ideal_jaccard_dcg, _VerifiedJaccardIDCG)
        ideal_snapshot = _VERIFIED_IDEAL_REGISTRY.get(id(ideal_jaccard_dcg))
        if (
            ideal_snapshot is None
            or ideal_snapshot[0]() is not ideal_jaccard_dcg
            or ideal_jaccard_dcg.gains is not ideal_snapshot[1]
            or ideal_jaccard_dcg.cutoffs != ideal_snapshot[2]
            or ideal_jaccard_dcg.values != ideal_snapshot[3]
            or ideal_jaccard_dcg._token is not ideal_snapshot[4]
            or ideal_snapshot[4] is not _VERIFIED_IDEAL_TOKEN
            or ideal_jaccard_dcg.gains is not gains
            or ideal_jaccard_dcg.cutoffs != cutoff_values
            or gains.flags.writeable
            or not gains.flags.c_contiguous
            or not isinstance(gains.base, bytes)
        ):
            raise ValueError("verified Jaccard IDCG binding differs from this query")
        # Read the externally registered construction values directly.  An
        # instance ``__dict__`` can shadow non-data descriptors such as
        # ``as_dict`` even on a frozen dataclass.
        ideal_values: dict[int, float] | None = dict(ideal_snapshot[3])
    elif ideal_jaccard_dcg is None:
        ideal_values = None
    elif isinstance(ideal_jaccard_dcg, dict):
        if set(ideal_jaccard_dcg) != set(cutoff_values):
            raise ValueError("precomputed Jaccard IDCG cutoffs differ")
        ideal_values = dict(ideal_jaccard_dcg)
    else:
        raise TypeError("ideal_jaccard_dcg must be a verified certificate or dict")

    group_sizes = np.bincount(radius.astype(np.int64, copy=False), minlength=levels).astype(
        np.int64, copy=False
    )
    relevant_counts = np.bincount(
        radius[relevant].astype(np.int64, copy=False), minlength=levels
    ).astype(np.int64, copy=False)
    gain_sums = np.bincount(
        radius.astype(np.int64, copy=False), weights=gains, minlength=levels
    ).astype(np.float64, copy=False)
    total_relevant = int(relevant_counts.sum())

    stable_basic_cache: dict[int, dict[str, _StableEstimate]] = {}
    stable_j_cache: dict[int, _StableEstimate] = {}
    stable_gain_sums_cache: tuple[Decimal, ...] | None = None
    stable_discount_cache: dict[int, tuple[Decimal, ...]] = {}
    stable_idcg_cache: dict[int, tuple[Decimal, Decimal]] = {}
    stable_ap_cache: _StableEstimate | None = None

    def stable_gain_sums() -> tuple[Decimal, ...]:
        nonlocal stable_gain_sums_cache
        if stable_gain_sums_cache is None:
            with localcontext() as context:
                context.prec = _STABLE_DECIMAL_PRECISION
                totals = [Decimal(0) for _ in range(levels)]
                for shell_raw, gain_raw in zip(radius, gains):
                    totals[int(shell_raw)] += Decimal.from_float(float(gain_raw))
                stable_gain_sums_cache = tuple(totals)
        return stable_gain_sums_cache

    def stable_discount_weights(effective: int) -> tuple[Decimal, ...]:
        cached = stable_discount_cache.get(effective)
        if cached is not None:
            return cached
        with localcontext() as context:
            context.prec = _STABLE_DECIMAL_PRECISION
            cached = tuple(
                Decimal.from_float(1.0 / math.log2(rank + 2.0))
                for rank in range(effective)
            )
            stable_discount_cache[effective] = cached
            return cached

    def stable_direct_idcg(cutoff: int) -> tuple[Decimal, Decimal]:
        """Return direct Decimal IDCG and the float precompute error bound."""

        cached = stable_idcg_cache.get(cutoff)
        if cached is not None:
            return cached
        effective = min(cutoff, len(gains))
        with localcontext() as context:
            context.prec = _STABLE_DECIMAL_PRECISION
            weights = stable_discount_weights(effective)
            if effective == len(gains):
                ideal_gains = np.sort(gains)[::-1]
            else:
                ideal_gains = np.partition(
                    gains, len(gains) - effective
                )[-effective:]
                ideal_gains.sort()
                ideal_gains = ideal_gains[::-1]
            stable_idcg = sum(
                (
                    Decimal.from_float(float(gain)) * weights[position]
                    for position, gain in enumerate(ideal_gains)
                ),
                Decimal(0),
            )
            product_rounding = sum(
                (
                    Decimal.from_float(
                        math.ulp(float(gain) * float(weights[position]))
                    )
                    / Decimal(2)
                    for position, gain in enumerate(ideal_gains)
                ),
                Decimal(0),
            )
            # One result ULP covers fsum rounding and its documented rare
            # extended-precision double-round on non-Windows builds.
            rounded = float(stable_idcg)
            fsum_rounding = Decimal.from_float(math.ulp(rounded))
            cached = (stable_idcg, product_rounding + fsum_rounding)
            stable_idcg_cache[cutoff] = cached
            return cached

    def stable_basic_metrics(cutoff: int) -> dict[str, _StableEstimate]:
        """Stable P/R/binary-NDCG fallback without scanning database gains."""

        cached = stable_basic_cache.get(cutoff)
        if cached is not None:
            return cached
        effective = min(cutoff, len(relevant))
        with localcontext() as context:
            context.prec = _STABLE_DECIMAL_PRECISION
            weights = stable_discount_weights(effective)
            expected_rel = Decimal(0)
            expected_binary = Decimal(0)
            shadow_rel = Decimal(0)
            shadow_binary = Decimal(0)
            start_position = 0
            active_shells = 0
            for size_raw, block_relevant_raw in zip(group_sizes, relevant_counts):
                size = int(size_raw)
                if size == 0:
                    continue
                take = max(0, min(size, effective - start_position))
                if take:
                    active_shells += 1
                    probability = Decimal(int(block_relevant_raw)) / Decimal(size)
                    discount_sum = sum(
                        weights[start_position : start_position + take], Decimal(0)
                    )
                    expected_rel += Decimal(take) * probability
                    expected_binary += probability * discount_sum
                    shadow_probability = Decimal.from_float(
                        int(block_relevant_raw) / float(size)
                    )
                    shadow_discount = Decimal.from_float(
                        float(
                            prefix.discount[start_position + take]
                            - prefix.discount[start_position]
                        )
                    )
                    shadow_rel += Decimal(take) * shadow_probability
                    shadow_binary += shadow_probability * shadow_discount
                start_position += size
            ideal_relevant = min(total_relevant, effective)
            binary_idcg = sum(weights[:ideal_relevant], Decimal(0))
            shadow_binary_idcg = (
                Decimal.from_float(float(prefix.discount[ideal_relevant]))
                if ideal_relevant
                else Decimal(0)
            )

            def ratio(numerator: Decimal, denominator: Decimal) -> Decimal:
                if denominator == 0:
                    return Decimal(0) if numerator == 0 else Decimal("Infinity")
                return numerator / denominator

            values = {
                f"precision_at_{cutoff}_expected_ties": (
                    expected_rel / Decimal(effective)
                ),
                f"recall_at_{cutoff}_expected_ties": (
                    expected_rel / Decimal(total_relevant)
                    if total_relevant
                    else Decimal(0)
                ),
                f"binary_ndcg_at_{cutoff}_expected_ties": ratio(
                    expected_binary, binary_idcg
                ),
            }
            shadow_values = {
                f"precision_at_{cutoff}_expected_ties": (
                    shadow_rel / Decimal(effective)
                ),
                f"recall_at_{cutoff}_expected_ties": (
                    shadow_rel / Decimal(total_relevant)
                    if total_relevant
                    else Decimal(0)
                ),
                f"binary_ndcg_at_{cutoff}_expected_ties": ratio(
                    shadow_binary, shadow_binary_idcg
                ),
            }
            operations = effective + ideal_relevant + 8 * active_shells + 16
            cached = {}
            for key, value in values.items():
                fast_operations = 8 * active_shells + 24
                shadow_value = shadow_values[key]
                cached[key] = _StableEstimate(
                    value=value,
                    absolute_error=_decimal_rounding_bound(value, operations),
                    binary64_error=(
                        abs(shadow_value - value)
                        + _binary64_rounding_bound(
                            shadow_value, fast_operations
                        )
                    ),
                )
            stable_basic_cache[cutoff] = cached
            return cached

    def stable_j_ndcg(cutoff: int) -> _StableEstimate:
        """Stable J-NDCG fallback and precomputed-IDCG certificate check."""

        cached = stable_j_cache.get(cutoff)
        if cached is not None:
            return cached
        effective = min(cutoff, len(relevant))
        with localcontext() as context:
            context.prec = _STABLE_DECIMAL_PRECISION
            weights = stable_discount_weights(effective)
            gain_totals = stable_gain_sums()
            expected_jaccard = Decimal(0)
            shadow_jaccard = Decimal(0)
            start_position = 0
            active_shells = 0
            for shell, size_raw in enumerate(group_sizes):
                size = int(size_raw)
                if size == 0:
                    continue
                take = max(0, min(size, effective - start_position))
                if take:
                    active_shells += 1
                    discount_sum = sum(
                        weights[start_position : start_position + take], Decimal(0)
                    )
                    expected_jaccard += (
                        gain_totals[shell] / Decimal(size)
                    ) * discount_sum
                    shadow_mean_gain = Decimal.from_float(
                        float(gain_sums[shell]) / float(size)
                    )
                    shadow_discount = Decimal.from_float(
                        float(
                            prefix.discount[start_position + take]
                            - prefix.discount[start_position]
                        )
                    )
                    shadow_jaccard += shadow_mean_gain * shadow_discount
                start_position += size

            stable_idcg, _provided_error = stable_direct_idcg(cutoff)
            if ideal_values is not None:
                shadow_idcg = Decimal.from_float(float(ideal_values[cutoff]))
            else:
                if effective == len(gains):
                    shadow_ideal_gains = np.sort(gains)[::-1]
                else:
                    shadow_ideal_gains = np.partition(
                        gains, len(gains) - effective
                    )[-effective:]
                    shadow_ideal_gains.sort()
                    shadow_ideal_gains = shadow_ideal_gains[::-1]
                shadow_idcg = Decimal.from_float(
                    float(
                        np.dot(
                            shadow_ideal_gains,
                            np.diff(prefix.discount[: effective + 1]),
                        )
                    )
                )

            if stable_idcg == 0:
                value = (
                    Decimal(0)
                    if expected_jaccard == 0
                    else Decimal("Infinity")
                )
            else:
                value = expected_jaccard / stable_idcg
            if shadow_idcg == 0:
                shadow_value = (
                    Decimal(0)
                    if shadow_jaccard == 0
                    else Decimal("Infinity")
                )
            else:
                shadow_value = shadow_jaccard / shadow_idcg
            operations = (
                len(relevant)
                + 3 * effective
                + 8 * active_shells
                + 24
            )
            cached = _StableEstimate(
                value=value,
                absolute_error=(
                    _decimal_rounding_bound(value, operations)
                    if value.is_finite()
                    else Decimal(0)
                ),
                binary64_error=(
                    abs(shadow_value - value)
                    + _binary64_rounding_bound(
                        shadow_value, 6 * active_shells + 24
                    )
                    if value.is_finite() and shadow_value.is_finite()
                    else Decimal(0)
                ),
            )
            stable_j_cache[cutoff] = cached
            return cached

    def stable_average_precision() -> _StableEstimate:
        nonlocal stable_ap_cache
        if stable_ap_cache is not None:
            return stable_ap_cache
        if total_relevant == 0:
            stable_ap_cache = _StableEstimate(
                Decimal(0), Decimal(0), Decimal(0)
            )
            return stable_ap_cache
        with localcontext() as context:
            context.prec = _STABLE_DECIMAL_PRECISION
            terms: list[Decimal] = []
            shadow_terms: list[Decimal] = []
            previous = 0
            start_position = 0
            active_shells = 0
            for size_raw, block_relevant_raw in zip(group_sizes, relevant_counts):
                size = int(size_raw)
                if size == 0:
                    continue
                block_relevant = int(block_relevant_raw)
                if block_relevant:
                    active_shells += 1
                    probability = Decimal(block_relevant) / Decimal(size)
                    harmonic_sum = Decimal(0)
                    within_offset_sum = Decimal(0)
                    for offset in range(size):
                        rank = start_position + offset + 1
                        harmonic_sum += Decimal(1) / Decimal(rank)
                        within_offset_sum += Decimal(offset) / Decimal(rank)
                    alpha = (
                        Decimal(0)
                        if size == 1
                        else Decimal(block_relevant - 1) / Decimal(size - 1)
                    )
                    terms.append(
                        probability
                        * (
                            Decimal(previous + 1) * harmonic_sum
                            + alpha * within_offset_sum
                        )
                    )
                    shadow_probability = Decimal.from_float(
                        block_relevant / float(size)
                    )
                    shadow_harmonic = Decimal.from_float(
                        float(
                            prefix.harmonic[start_position + size]
                            - prefix.harmonic[start_position]
                        )
                    )
                    shadow_alpha = Decimal.from_float(
                        0.0
                        if size == 1
                        else (block_relevant - 1.0) / (size - 1.0)
                    )
                    shadow_within = Decimal.from_float(
                        float(size)
                        - (start_position + 1.0) * float(shadow_harmonic)
                    )
                    shadow_terms.append(
                        shadow_probability
                        * (
                            Decimal(previous + 1) * shadow_harmonic
                            + shadow_alpha * shadow_within
                        )
                    )
                previous += block_relevant
                start_position += size
            value = sum(terms, Decimal(0)) / Decimal(total_relevant)
            shadow_value = sum(shadow_terms, Decimal(0)) / Decimal(
                total_relevant
            )
            operations = 4 * len(relevant) + 12 * active_shells + 16
            stable_ap_cache = _StableEstimate(
                value=value,
                absolute_error=_decimal_rounding_bound(value, operations),
                binary64_error=(
                    abs(shadow_value - value)
                    + _binary64_rounding_bound(
                        shadow_value, 10 * active_shells + 24
                    )
                ),
            )
            return stable_ap_cache

    if ideal_values is not None:
        for cutoff in cutoff_values:
            effective = min(cutoff, len(gains))
            provided_float = float(ideal_values[cutoff])
            if not np.isfinite(provided_float) or provided_float < 0.0:
                raise ValueError(
                    "precomputed Jaccard IDCG must be finite and nonnegative"
                )
            if effective and np.any(gains > 0.0) and provided_float == 0.0:
                raise ValueError("positive Jaccard gains require positive IDCG")
            if not ideal_is_verified:
                stable_idcg, provided_error = stable_direct_idcg(cutoff)
                provided = Decimal.from_float(provided_float)
                if abs(provided - stable_idcg) > provided_error:
                    raise FloatingPointError(
                        "precomputed Jaccard IDCG differs from stable direct IDCG"
                    )

    result: dict[str, float | int | bool] = {
        "database_rows": int(len(relevant)),
        "relevant_rows": total_relevant,
        "has_relevant": bool(total_relevant > 0),
    }
    expected_ap_numerator = 0.0
    previous_relevant = 0
    start = 0
    expected_rel_at = {cutoff: 0.0 for cutoff in cutoff_values}
    expected_binary_dcg = {cutoff: 0.0 for cutoff in cutoff_values}
    expected_jaccard_dcg = {cutoff: 0.0 for cutoff in cutoff_values}
    for size_raw, block_relevant_raw, gain_sum_raw in zip(
        group_sizes, relevant_counts, gain_sums
    ):
        size = int(size_raw)
        if size == 0:
            continue
        block_relevant = int(block_relevant_raw)
        probability = block_relevant / float(size)
        harmonic_sum = float(prefix.harmonic[start + size] - prefix.harmonic[start])
        if block_relevant:
            alpha = 0.0 if size == 1 else (block_relevant - 1.0) / (size - 1.0)
            within_offset_sum = size - (start + 1.0) * harmonic_sum
            expected_ap_numerator += probability * (
                (previous_relevant + 1.0) * harmonic_sum + alpha * within_offset_sum
            )
        mean_gain = float(gain_sum_raw) / float(size)
        for cutoff in cutoff_values:
            effective = min(cutoff, len(relevant))
            take = max(0, min(size, effective - start))
            if not take:
                continue
            expected_rel_at[cutoff] += take * probability
            discount_sum = float(prefix.discount[start + take] - prefix.discount[start])
            expected_binary_dcg[cutoff] += probability * discount_sum
            expected_jaccard_dcg[cutoff] += mean_gain * discount_sum
        previous_relevant += block_relevant
        start += size
    if start != len(relevant):
        raise AssertionError("Hamming shell counts do not cover the database")
    result["average_precision_expected_ties"] = _canonical_unit_interval(
        expected_ap_numerator / total_relevant if total_relevant else 0.0,
        field="average_precision_expected_ties",
        stable_value=stable_average_precision,
    )
    for cutoff in cutoff_values:
        effective = min(cutoff, len(relevant))
        rel_at = expected_rel_at[cutoff]
        ideal_relevant = min(total_relevant, effective)
        binary_idcg = float(prefix.discount[ideal_relevant]) if ideal_relevant else 0.0
        if ideal_values is None:
            if effective == len(gains):
                ideal_gains = np.sort(gains)[::-1]
            else:
                ideal_gains = np.partition(gains, len(gains) - effective)[-effective:]
                ideal_gains.sort()
                ideal_gains = ideal_gains[::-1]
            jaccard_idcg = float(
                np.dot(
                    ideal_gains,
                    np.diff(prefix.discount[: effective + 1]),
                )
            )
        else:
            jaccard_idcg = float(ideal_values[cutoff])
        precision_key = f"precision_at_{cutoff}_expected_ties"
        recall_key = f"recall_at_{cutoff}_expected_ties"
        binary_ndcg_key = f"binary_ndcg_at_{cutoff}_expected_ties"
        j_ndcg_key = f"j_ndcg_at_{cutoff}_expected_ties"
        result[precision_key] = _canonical_unit_interval(
            rel_at / effective,
            field=precision_key,
            stable_value=lambda cutoff=cutoff, key=precision_key: (
                stable_basic_metrics(cutoff)[key]
            ),
        )
        result[recall_key] = _canonical_unit_interval(
            rel_at / total_relevant if total_relevant else 0.0,
            field=recall_key,
            stable_value=lambda cutoff=cutoff, key=recall_key: (
                stable_basic_metrics(cutoff)[key]
            ),
        )
        result[binary_ndcg_key] = _canonical_unit_interval(
            expected_binary_dcg[cutoff] / binary_idcg if binary_idcg else 0.0,
            field=binary_ndcg_key,
            stable_value=lambda cutoff=cutoff, key=binary_ndcg_key: (
                stable_basic_metrics(cutoff)[key]
            ),
        )
        result[j_ndcg_key] = _canonical_unit_interval(
            expected_jaccard_dcg[cutoff] / jaccard_idcg if jaccard_idcg else 0.0,
            field=j_ndcg_key,
            stable_value=lambda cutoff=cutoff: stable_j_ndcg(cutoff),
        )
    return result


def _stable_record_mean(
    records: Sequence[dict[str, Any]], key: str
) -> _StableEstimate:
    with localcontext() as context:
        context.prec = _STABLE_DECIMAL_PRECISION
        total = sum(
            (Decimal.from_float(float(record[key])) for record in records),
            Decimal(0),
        )
        value = total / Decimal(len(records))
        operations = len(records) + 1
        return _StableEstimate(
            value=value,
            absolute_error=_decimal_rounding_bound(value, operations),
            binary64_error=_binary64_rounding_bound(value, operations),
        )


def mean_query_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("metric result has no queries")
    scalar_keys = sorted(
        key
        for key, value in records[0].items()
        if isinstance(value, float)
    )
    summary = {}
    for key in scalar_keys:
        mean = float(
            np.mean([float(record[key]) for record in records], dtype=np.float64)
        )
        summary[key] = _canonical_unit_interval(
            mean,
            field=f"mean {key}",
            stable_value=lambda key=key: _stable_record_mean(records, key),
        )
    summary["map_expected_ties"] = summary.pop("average_precision_expected_ties")
    valid = [record for record in records if record["has_relevant"]]
    summary["queries"] = len(records)
    summary["queries_with_relevant"] = len(valid)
    summary["zero_relevant_policy"] = "included_as_zero_in_primary_mean"
    summary["map_expected_ties_valid_queries"] = (
        _canonical_unit_interval(
            float(
                np.mean(
                    [
                        float(record["average_precision_expected_ties"])
                        for record in valid
                    ],
                    dtype=np.float64,
                )
            ),
            field="map_expected_ties_valid_queries",
            stable_value=lambda: _stable_record_mean(
                valid, "average_precision_expected_ties"
            ),
        )
        if valid
        else 0.0
    )
    return summary


__all__ = [
    "MetricPrefixes",
    "build_metric_prefixes",
    "expected_tie_metrics_from_distances",
    "mean_query_metrics",
]
