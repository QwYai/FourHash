from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio


class TraceContractError(RuntimeError):
    """Raised when provenance cannot be proven without guessing."""


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: os.PathLike[str], chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(stable_json_bytes(value))


def canonical_row_id(dataset: str, source_id: str) -> str:
    return sha256_bytes(
        "\0".join(("kbs-trace-row-v1", dataset, str(source_id))).encode("utf-8")
    )


def atomic_write_bytes(path: os.PathLike[str], payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(target))


def atomic_write_json(path: os.PathLike[str], value: Any) -> None:
    atomic_write_bytes(path, stable_json_bytes(value) + b"\n")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def ensure_output_safe(
    output_root: os.PathLike[str], forbidden_roots: Iterable[os.PathLike[str]]
) -> Path:
    """Reject any output located in or above a protected data tree."""

    output = Path(output_root).expanduser().resolve()
    for raw in forbidden_roots:
        protected = Path(raw).expanduser().resolve()
        if output == protected or _is_relative_to(output, protected):
            raise TraceContractError(
                "output root must not be inside protected data root: "
                f"output={output}, protected={protected}"
            )
        if _is_relative_to(protected, output):
            raise TraceContractError(
                "output root must not contain a protected data root: "
                f"output={output}, protected={protected}"
            )
    return output


def require_file(path: os.PathLike[str], description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise TraceContractError(f"missing {description}: {resolved}")
    return resolved


def load_mat_variable(path: os.PathLike[str], key: str) -> Any:
    source = require_file(path, f"MAT file containing {key}")
    values = sio.loadmat(str(source))
    if key not in values:
        public = sorted(k for k in values if not k.startswith("__"))
        raise TraceContractError(f"{source} has no {key!r}; variables={public}")
    return values[key]


def dense_row(array: Any, index: int) -> np.ndarray:
    row = array[index]
    if hasattr(row, "toarray"):
        row = row.toarray()
    return np.asarray(row).reshape(-1)


def padded_ids_to_hot(values: Sequence[Any], classes: int) -> np.ndarray:
    hot = np.zeros(classes, dtype=np.uint8)
    observed = set()
    for raw in np.asarray(values).reshape(-1):
        value = int(raw)
        if value == 0:
            continue
        if not 1 <= value <= classes:
            raise TraceContractError(
                f"padded label ID {value} outside the valid range 1..{classes}"
            )
        if value in observed:
            raise TraceContractError(f"duplicate padded label ID {value}")
        observed.add(value)
        hot[value - 1] = 1
    return hot


@dataclass(frozen=True)
class SplitMembership:
    """Validated zero-based baseline ``indQ``/``indT``/``indD`` row sets."""

    rows: int
    ind_q: frozenset
    ind_t: frozenset
    ind_d: frozenset
    source_path: str

    @classmethod
    def from_mat(cls, path: os.PathLike[str], rows: int) -> "SplitMembership":
        source = require_file(path, "split MAT")
        values = sio.loadmat(str(source))
        arrays: Dict[str, np.ndarray] = {}
        for key in ("indQ", "indT", "indD"):
            if key not in values:
                raise TraceContractError(f"split {source} has no variable {key}")
            vector = np.asarray(values[key]).reshape(-1)
            if not np.issubdtype(vector.dtype, np.integer):
                if not np.all(np.equal(vector, np.floor(vector))):
                    raise TraceContractError(f"split {key} contains non-integers")
            vector = vector.astype(np.int64)
            if vector.size and (int(vector.min()) < 0 or int(vector.max()) >= rows):
                raise TraceContractError(
                    f"split {key} is not zero-based in [0,{rows}): "
                    f"min={int(vector.min())}, max={int(vector.max())}"
                )
            if np.unique(vector).size != vector.size:
                raise TraceContractError(f"split {key} contains duplicate rows")
            arrays[key] = vector

        q = frozenset(int(v) for v in arrays["indQ"])
        t = frozenset(int(v) for v in arrays["indT"])
        d = frozenset(int(v) for v in arrays["indD"])
        universe = frozenset(range(rows))
        if q & d:
            raise TraceContractError("indQ and indD overlap")
        if q | d != universe:
            missing = len(universe - (q | d))
            extra = len((q | d) - universe)
            raise TraceContractError(
                f"indQ union indD does not cover the row universe; missing={missing}, extra={extra}"
            )
        if not t <= d:
            raise TraceContractError("indT is not a subset of indD")
        return cls(rows, q, t, d, str(source))

    def flags(self, row_index: int) -> Dict[str, bool]:
        if not 0 <= row_index < self.rows:
            raise TraceContractError(f"row {row_index} outside [0,{self.rows})")
        return {
            "indQ": row_index in self.ind_q,
            "indT": row_index in self.ind_t,
            "indD": row_index in self.ind_d,
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "path": self.source_path,
            "index_base": 0,
            "indQ_rows": len(self.ind_q),
            "indT_rows": len(self.ind_t),
            "indD_rows": len(self.ind_d),
            "sha256": sha256_file(self.source_path),
        }


@dataclass(frozen=True)
class ContentHashSplit:
    """Cross-version split selected by domain-separated SHA-256 ordering.

    This deliberately avoids NumPy RNG state and any search over seeds.  Sample
    identity, dataset, role, algorithm version, and the one frozen seed all
    participate in the selection key.
    """

    rows: int
    dataset: str
    seed: int
    source_ids: Tuple[str, ...]
    ind_q: frozenset
    ind_t: frozenset
    ind_d: frozenset
    algorithm: str = "kbs-content-hash-split-v1"

    @staticmethod
    def _key(
        algorithm: str, seed: int, dataset: str, role: str, source_id: str
    ) -> bytes:
        payload = "\0".join(
            (algorithm, str(seed), dataset, role, source_id)
        ).encode("utf-8")
        return hashlib.sha256(payload).digest()

    @classmethod
    def build(
        cls,
        dataset: str,
        source_ids: Sequence[str],
        query_rows: int,
        train_rows: int,
        seed: int = 20_260_822,
    ) -> "ContentHashSplit":
        identities = tuple(str(value) for value in source_ids)
        rows = len(identities)
        if len(set(identities)) != rows:
            raise TraceContractError(f"{dataset} canonical source IDs are not unique")
        if not 0 < query_rows < rows:
            raise TraceContractError(
                f"invalid query size {query_rows} for {dataset} rows={rows}"
            )
        database_rows = rows - query_rows
        if not 0 < train_rows <= database_rows:
            raise TraceContractError(
                f"invalid train size {train_rows} for database rows={database_rows}"
            )
        algorithm = "kbs-content-hash-split-v1"
        all_rows = list(range(rows))
        query_order = sorted(
            all_rows,
            key=lambda row: (
                cls._key(algorithm, seed, dataset, "query", identities[row]),
                identities[row],
                row,
            ),
        )
        q = frozenset(query_order[:query_rows])
        d = frozenset(row for row in all_rows if row not in q)
        train_order = sorted(
            d,
            key=lambda row: (
                cls._key(algorithm, seed, dataset, "train", identities[row]),
                identities[row],
                row,
            ),
        )
        t = frozenset(train_order[:train_rows])
        return cls(rows, dataset, seed, identities, q, t, d, algorithm)

    def flags(self, row_index: int) -> Dict[str, bool]:
        if not 0 <= row_index < self.rows:
            raise TraceContractError(f"row {row_index} outside [0,{self.rows})")
        return {
            "indQ": row_index in self.ind_q,
            "indT": row_index in self.ind_t,
            "indD": row_index in self.ind_d,
        }

    def arrays(self) -> Dict[str, np.ndarray]:
        return {
            "indQ": np.asarray(sorted(self.ind_q), dtype=np.int64),
            "indT": np.asarray(sorted(self.ind_t), dtype=np.int64),
            "indD": np.asarray(sorted(self.ind_d), dtype=np.int64),
            "row_ids": np.asarray(
                [canonical_row_id(self.dataset, value) for value in self.source_ids],
                dtype="S64",
            ),
        }

    def summary(self) -> Dict[str, Any]:
        arrays = self.arrays()
        identity_digest = sha256_json(list(self.source_ids))
        selection = {
            key: arrays[key].tolist() for key in ("indQ", "indT", "indD")
        }
        ordered_row_ids = [value.decode("ascii") for value in arrays["row_ids"]]
        return {
            "algorithm": self.algorithm,
            "seed": self.seed,
            "index_base": 0,
            "identity_order_sha256": identity_digest,
            "ordered_row_ids_sha256": sha256_json(ordered_row_ids),
            "indQ_rows": len(self.ind_q),
            "indT_rows": len(self.ind_t),
            "indD_rows": len(self.ind_d),
            "selection_sha256": sha256_json(selection),
        }


@dataclass(frozen=True)
class TraceRow:
    dataset: str
    row_index: int
    source_index: int
    source_id: str
    image_path: str
    encoded_texts: Tuple[str, ...]
    raw_text: Mapping[str, Any]
    label_hot: Tuple[int, ...]
    split: Mapping[str, bool]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    raw_text_locators: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.row_index < 0 or self.source_index < 0:
            raise TraceContractError("row and source indices must be non-negative")
        if not self.source_id:
            raise TraceContractError("source_id must be non-empty")
        if not self.encoded_texts:
            raise TraceContractError("every row must declare at least one encoded text")
        labels = tuple(int(v) for v in self.label_hot)
        if any(v not in (0, 1) for v in labels):
            raise TraceContractError("label_hot must be binary")
        object.__setattr__(self, "label_hot", labels)

    def canonical_text_digest(self) -> str:
        return sha256_json(list(self.encoded_texts))

    def raw_text_digest(self) -> str:
        return sha256_json(self.raw_text)

    def label_digest(self) -> str:
        return sha256_bytes(np.asarray(self.label_hot, dtype=np.uint8).tobytes())

    def canonical_row_id(self) -> str:
        return canonical_row_id(self.dataset, self.source_id)

    @staticmethod
    def _enrich_source(
        locator: Mapping[str, Any],
        dependency_inventory: Optional[Mapping[str, Mapping[str, Any]]],
        hash_content: bool,
    ) -> Dict[str, Any]:
        if "path" not in locator:
            raise TraceContractError("raw source locator has no path")
        source = require_file(str(locator["path"]), "raw source")
        canonical = str(source)
        if dependency_inventory is not None:
            if canonical not in dependency_inventory:
                raise TraceContractError(
                    f"raw source is absent from dependency inventory: {canonical}"
                )
            inventory = dependency_inventory[canonical]
            size = int(inventory["bytes"])
            digest = str(inventory["sha256"])
        else:
            size = source.stat().st_size
            digest = sha256_file(source) if hash_content else None
        enriched = dict(locator)
        enriched["path"] = canonical
        enriched["bytes"] = size
        enriched["sha256"] = digest
        return enriched

    def record(
        self,
        hash_image: bool = True,
        dependency_inventory: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        image = require_file(self.image_path, "raw image")
        image_source = self._enrich_source(
            {"path": str(image), "kind": "raw_image"},
            dependency_inventory,
            hash_image,
        )
        raw_sources = [
            self._enrich_source(locator, dependency_inventory, hash_image)
            for locator in self.raw_text_locators
        ]
        if not raw_sources:
            raise TraceContractError("every trace row must bind at least one raw text source")
        record = {
            "schema_version": 1,
            "dataset": self.dataset,
            "row_index": self.row_index,
            "source_index": self.source_index,
            "source_id": self.source_id,
            "canonical_row_id": self.canonical_row_id(),
            "image_path": str(image),
            "raw_image_source": image_source,
            "encoded_texts": list(self.encoded_texts),
            "raw_text": dict(self.raw_text),
            "raw_text_sources": raw_sources,
            "label_hot": list(self.label_hot),
            "split": dict(self.split),
            "metadata": dict(self.metadata),
            "encoded_text_sha256": self.canonical_text_digest(),
            "raw_text_sha256": self.raw_text_digest(),
            "label_sha256": self.label_digest(),
        }
        record["image_sha256"] = image_source["sha256"]
        record["row_contract_sha256"] = sha256_json(record)
        return record


def validate_expected_rows(actual: int, expected: Optional[int], dataset: str) -> None:
    if expected is not None and actual != expected:
        raise TraceContractError(
            f"{dataset} canonical row count mismatch: expected={expected}, actual={actual}"
        )


def file_inventory(paths: Iterable[os.PathLike[str]]) -> Tuple[Dict[str, Any], ...]:
    inventory = []
    seen = set()
    for raw in paths:
        path = require_file(raw, "source artifact")
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        stat = path.stat()
        inventory.append(
            {
                "path": key,
                "bytes": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    return tuple(inventory)
