"""Fail-closed validator for independently extracted visualization features."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from .adapters import ALLOWED_EXACT_IDENTITY_METHODS, FORBIDDEN_IDENTITY_TOKENS
from .contract import (
    RAW_REBUILT_COUNTS,
    RAW_REBUILT_LABEL_DIMS,
    RAW_REBUILT_SPLIT_SEED,
    SCHEMA_VERSION,
    ContractError,
    canonical_json_bytes,
    canonical_sample_id_order,
    derive_bundle_id,
    derive_extractor_id,
    derive_row_id,
    feature_row_sha256,
    int64_array_sha256,
    is_within,
    normalize_text,
    numeric_array_sha256,
    ordered_ids_sha256,
    raw_rebuilt_assignment_sha256,
    raw_rebuilt_split_indices,
    require_content_id,
    require_sha256,
    resolve_contained,
    sha256_bytes,
    sha256_file,
    text_utf8_sha256,
)


MANIFEST_NAME = "bundle_manifest.json"
SEMANTICS = ("image", "text", "multilabel")
SPLIT_NAMES = ("indT", "indQ", "indD")
TOP_LEVEL_KEYS = {
    "schema_version",
    "bundle_id",
    "dataset",
    "created_utc",
    "row_count",
    "boundaries",
    "authority",
    "extractors",
    "rows",
    "deterministic_splits",
    "labels",
    "shards",
    "inventory",
}


@dataclass(frozen=True)
class ValidationReport:
    bundle_id: str
    dataset: str
    row_count: int
    shard_count: int
    source_file_count: int
    checks: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "PASS",
            "bundle_id": self.bundle_id,
            "dataset": self.dataset,
            "row_count": self.row_count,
            "shard_count": self.shard_count,
            "source_file_count": self.source_file_count,
            "checks": list(self.checks),
        }


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ContractError(f"{field} keys differ; missing={missing}, extra={extra}")


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{field} must be an object")
    return value


def _require_list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractError(f"{field} must be a list")
    return value


def _require_nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{field} must be a non-empty string")
    return value


def _require_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{field} must be a non-negative integer")
    return value


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _decode_json(payload: bytes, *, field: str) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError(f"{field} is not UTF-8") from error
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as error:
        if isinstance(error, ContractError):
            raise
        raise ContractError(f"invalid JSON in {field}: {error}") from error


def _load_canonical_manifest(path: Path) -> Mapping[str, Any]:
    payload = path.read_bytes()
    value = _require_mapping(_decode_json(payload, field=MANIFEST_NAME), field=MANIFEST_NAME)
    expected = canonical_json_bytes(value) + b"\n"
    if payload != expected:
        raise ContractError(f"{MANIFEST_NAME} must be canonical UTF-8 JSON plus one LF")
    return value


def _load_canonical_jsonl(path: Path) -> list[Mapping[str, Any]]:
    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise ContractError("rows JSONL must be non-empty and end with LF")
    rows: list[Mapping[str, Any]] = []
    for line_number, raw_line in enumerate(payload.splitlines(keepends=True), start=1):
        if not raw_line.endswith(b"\n") or raw_line == b"\n":
            raise ContractError(f"rows JSONL line {line_number} is empty or unterminated")
        value = _require_mapping(
            _decode_json(raw_line[:-1], field=f"rows line {line_number}"),
            field=f"rows line {line_number}",
        )
        if raw_line != canonical_json_bytes(value) + b"\n":
            raise ContractError(f"rows JSONL line {line_number} is not canonical JSON")
        rows.append(value)
    return rows


SourceCache = MutableMapping[Path, tuple[int, int, str]]


def _canonical_external_path(
    record: Mapping[str, Any], *, field: str, source_cache: SourceCache | None = None
) -> Path:
    _require_exact_keys(record, {"path", "size", "sha256"}, field=field)
    raw_path = _require_nonempty_string(record["path"], field=f"{field}.path")
    path = Path(raw_path)
    if not path.is_absolute():
        raise ContractError(f"{field}.path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ContractError(f"{field}.path is missing: {path}") from error
    if not resolved.is_file():
        raise ContractError(f"{field}.path is not a regular file: {resolved}")
    if str(resolved) != raw_path:
        raise ContractError(f"{field}.path must be its canonical resolved path")
    expected_size = _require_nonnegative_int(record["size"], field=f"{field}.size")
    stat = resolved.stat()
    if stat.st_size != expected_size:
        raise ContractError(f"{field} size mismatch")
    expected_sha = require_sha256(record["sha256"], field=f"{field}.sha256")
    cached = None if source_cache is None else source_cache.get(resolved)
    if cached is not None:
        cached_size, cached_mtime_ns, cached_sha = cached
        if (stat.st_size, stat.st_mtime_ns) != (cached_size, cached_mtime_ns):
            raise ContractError(f"{field} changed during validation")
        actual_sha = cached_sha
    else:
        actual_sha = sha256_file(resolved)
        if source_cache is not None:
            source_cache[resolved] = (stat.st_size, stat.st_mtime_ns, actual_sha)
    if actual_sha != expected_sha:
        raise ContractError(f"{field} SHA-256 mismatch")
    return resolved


def _reverify_source_cache(source_cache: SourceCache) -> None:
    for path, (expected_size, expected_mtime_ns, expected_sha) in source_cache.items():
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (expected_size, expected_mtime_ns):
            raise ContractError(f"external source changed during validation: {path}")
        if sha256_file(path) != expected_sha:
            raise ContractError(f"external source same-size/content poison detected: {path}")


def _validate_inventory(bundle: Path, manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = _require_list(manifest["inventory"], field="inventory")
    inventory: dict[str, Mapping[str, Any]] = {}
    previous = ""
    for index, raw in enumerate(records):
        record = _require_mapping(raw, field=f"inventory[{index}]")
        _require_exact_keys(record, {"path", "size", "sha256"}, field=f"inventory[{index}]")
        relative = _require_nonempty_string(record["path"], field=f"inventory[{index}].path")
        if relative <= previous:
            raise ContractError("inventory paths must be strictly sorted and unique")
        previous = relative
        path = resolve_contained(bundle, relative, field=f"inventory[{index}].path")
        expected_size = _require_nonnegative_int(record["size"], field=f"inventory[{index}].size")
        if path.stat().st_size != expected_size:
            raise ContractError(f"inventory size mismatch for {relative}")
        expected_sha = require_sha256(record["sha256"], field=f"inventory[{index}].sha256")
        if sha256_file(path) != expected_sha:
            raise ContractError(f"inventory SHA-256 mismatch for {relative}")
        inventory[relative] = record

    actual: list[str] = []
    for path in bundle.rglob("*"):
        if path.is_symlink():
            raise ContractError(f"bundle may not contain symlinks: {path}")
        if path.is_file() and path.name != MANIFEST_NAME:
            actual.append(path.relative_to(bundle).as_posix())
    actual.sort()
    if actual != list(inventory):
        raise ContractError(f"bundle inventory mismatch; declared={list(inventory)}, actual={actual}")
    return inventory


def _require_inventory_descriptor(
    descriptor: Mapping[str, Any],
    inventory: Mapping[str, Mapping[str, Any]],
    *,
    field: str,
) -> Path:
    relative = _require_nonempty_string(descriptor.get("path"), field=f"{field}.path")
    if relative not in inventory:
        raise ContractError(f"{field}.path is absent from inventory")
    item = inventory[relative]
    for key in ("size", "sha256"):
        if descriptor.get(key) != item[key]:
            raise ContractError(f"{field}.{key} disagrees with inventory")
    return Path(relative)


def _validate_boundaries(
    bundle: Path,
    manifest: Mapping[str, Any],
    forbidden_process_data_root: Path | None,
) -> tuple[Path | None, Path]:
    boundaries = _require_mapping(manifest["boundaries"], field="boundaries")
    _require_exact_keys(
        boundaries,
        {"raw_data_root", "output_kind", "process_data_policy", "raw_data_access"},
        field="boundaries",
    )
    process = (
        None
        if forbidden_process_data_root is None
        else forbidden_process_data_root.expanduser().resolve(strict=False)
    )
    raw = Path(
        _require_nonempty_string(boundaries["raw_data_root"], field="boundaries.raw_data_root")
    ).resolve(strict=True)
    if not raw.is_dir():
        raise ContractError("raw data root is not a directory")
    output = bundle.resolve(strict=True)
    if process is not None and (is_within(output, process) or is_within(process, output)):
        raise ContractError("visualization feature output must be completely outside ProcessData")
    if any(part.casefold() == "processdata" for part in output.parts):
        raise ContractError("visualization feature output path may not be named under ProcessData")
    if is_within(output, raw) or is_within(raw, output):
        raise ContractError("visualization feature output must be completely outside raw data")
    if boundaries["output_kind"] != "independent_visualization_features":
        raise ContractError("boundaries.output_kind is not the independent visualization namespace")
    if boundaries["process_data_policy"] != "forbidden_as_input_or_authority":
        raise ContractError("ProcessData must be forbidden as an input or authority")
    if boundaries["raw_data_access"] != "read_only":
        raise ContractError("raw data access must be read-only")
    return process, raw


def _validate_extractors(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    values = _require_list(manifest["extractors"], field="extractors")
    if len(values) != 2:
        raise ContractError("exactly one image and one text extractor are required")
    result: dict[str, Mapping[str, Any]] = {}
    semantic_seen: set[str] = set()
    expected_keys = {
        "extractor_id",
        "semantic",
        "model_name",
        "model_revision",
        "model_artifact_sha256",
        "library",
        "library_version",
        "code_version",
        "preprocess",
        "preprocess_sha256",
        "output_dtype",
        "output_shape",
    }
    for index, raw in enumerate(values):
        extractor = _require_mapping(raw, field=f"extractors[{index}]")
        _require_exact_keys(extractor, expected_keys, field=f"extractors[{index}]")
        semantic = extractor["semantic"]
        if semantic not in {"image", "text"} or semantic in semantic_seen:
            raise ContractError("extractors must cover image and text exactly once")
        semantic_seen.add(str(semantic))
        for name in ("model_name", "model_revision", "library", "library_version", "code_version"):
            _require_nonempty_string(extractor[name], field=f"extractors[{index}].{name}")
        require_sha256(extractor["model_artifact_sha256"], field="model_artifact_sha256")
        preprocess = _require_mapping(extractor["preprocess"], field="preprocess")
        if not preprocess:
            raise ContractError("extractor preprocess record may not be empty")
        expected_preprocess_sha = sha256_bytes(canonical_json_bytes(preprocess))
        if extractor["preprocess_sha256"] != expected_preprocess_sha:
            raise ContractError("extractor preprocess SHA-256 mismatch")
        dtype = np.dtype(_require_nonempty_string(extractor["output_dtype"], field="output_dtype"))
        if dtype.kind != "f":
            raise ContractError("image/text extractors must emit floating-point features")
        shape = extractor["output_shape"]
        if not isinstance(shape, list) or not shape or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape
        ):
            raise ContractError("extractor output_shape must contain positive dimensions")
        expected_id = derive_extractor_id(extractor)
        if extractor["extractor_id"] != expected_id:
            raise ContractError("extractor_id does not bind its complete configuration")
        result[expected_id] = extractor
    if semantic_seen != {"image", "text"}:
        raise ContractError("image/text extractor coverage is incomplete")
    return result


def _load_npz_vectors(path: Path, descriptor: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    row_ids_key = descriptor["row_ids_key"]
    vector_key = descriptor["vector_key"]
    try:
        with np.load(path, allow_pickle=False) as loaded:
            if set(loaded.files) != {row_ids_key, vector_key}:
                raise ContractError(f"NPZ {path} must contain exactly {row_ids_key!r} and {vector_key!r}")
            row_ids = np.asarray(loaded[row_ids_key])
            vectors = np.asarray(loaded[vector_key])
    except (OSError, ValueError) as error:
        if isinstance(error, ContractError):
            raise
        raise ContractError(f"cannot load NPZ shard {path}: {error}") from error
    if row_ids.dtype.kind not in "US" or row_ids.ndim != 1:
        raise ContractError(f"NPZ shard {path} row IDs must be a one-dimensional string array")
    return row_ids.astype(str), vectors


def _load_parquet_vectors(path: Path, descriptor: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as error:
        raise ContractError("Parquet validation requires pyarrow; refusing unverified Parquet content") from error
    try:
        table = pq.read_table(path)
    except Exception as error:
        raise ContractError(f"cannot read Parquet shard {path}: {error}") from error
    row_key = descriptor["row_ids_key"]
    vector_key = descriptor["vector_key"]
    if set(table.column_names) != {row_key, vector_key}:
        raise ContractError("Parquet shard must contain exactly the declared row-ID and vector columns")
    row_column = table[row_key].combine_chunks()
    vector_column = table[vector_key].combine_chunks()
    if not pa.types.is_string(row_column.type) and not pa.types.is_large_string(row_column.type):
        raise ContractError("Parquet row-ID column must be string")
    if not pa.types.is_list(vector_column.type) and not pa.types.is_fixed_size_list(vector_column.type):
        raise ContractError("Parquet vector column must be list or fixed-size-list")
    if row_column.null_count or vector_column.null_count:
        raise ContractError("Parquet shard may not contain null rows")
    row_ids = np.asarray(row_column.to_pylist(), dtype=str)
    try:
        vectors = np.asarray(vector_column.to_pylist(), dtype=np.dtype(descriptor["dtype"]))
    except (TypeError, ValueError) as error:
        raise ContractError("Parquet vector rows are ragged or disagree with dtype") from error
    return row_ids, vectors


def _load_vector_shard(bundle: Path, descriptor: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    path = resolve_contained(bundle, descriptor["path"], field=f"shard {descriptor['shard_id']} path")
    if descriptor["format"] == "npz":
        return _load_npz_vectors(path, descriptor)
    if descriptor["format"] == "parquet":
        return _load_parquet_vectors(path, descriptor)
    raise ContractError("feature shard format must be npz or parquet")


def _validate_locator(value: Any, *, field: str) -> Mapping[str, Any]:
    locator = _require_mapping(value, field=field)
    _require_exact_keys(locator, {"kind", "value"}, field=field)
    kind = _require_nonempty_string(locator["kind"], field=f"{field}.kind")
    if kind not in {"json_pointer", "utf8_line_0based", "utf8_byte_range"}:
        raise ContractError(
            f"{field}.kind {kind!r} has no audited resolver; dataset-specific locators fail closed"
        )
    _require_nonempty_string(locator["value"], field=f"{field}.value")
    return locator


LocatorCache = MutableMapping[tuple[Path, str], Any]


def _resolve_locator(
    source_path: Path,
    locator: Mapping[str, Any],
    *,
    field: str,
    locator_cache: LocatorCache,
) -> Any:
    kind = str(locator["kind"])
    value = str(locator["value"])
    key = (source_path, kind)
    if kind == "json_pointer":
        if key not in locator_cache:
            locator_cache[key] = _decode_json(source_path.read_bytes(), field=str(source_path))
        current: Any = locator_cache[key]
        if value == "":
            return current
        if not value.startswith("/"):
            raise ContractError(f"{field} JSON Pointer must start with /")
        for raw_token in value[1:].split("/"):
            token = raw_token.replace("~1", "/").replace("~0", "~")
            if isinstance(current, Mapping):
                if token not in current:
                    raise ContractError(f"{field} JSON Pointer misses key {token!r}")
                current = current[token]
            elif isinstance(current, list):
                if not token.isdigit():
                    raise ContractError(f"{field} JSON Pointer list token is not an index")
                index = int(token)
                if index >= len(current):
                    raise ContractError(f"{field} JSON Pointer list index is out of range")
                current = current[index]
            else:
                raise ContractError(f"{field} JSON Pointer traverses through a scalar")
        return current
    if kind == "utf8_line_0based":
        if not value.isdigit():
            raise ContractError(f"{field} line locator must be a zero-based integer")
        if key not in locator_cache:
            try:
                locator_cache[key] = source_path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError as error:
                raise ContractError(f"{field} source is not UTF-8") from error
        lines = locator_cache[key]
        index = int(value)
        if index >= len(lines):
            raise ContractError(f"{field} line locator is out of range")
        return lines[index]
    if kind == "utf8_byte_range":
        match = re.fullmatch(r"(0|[1-9][0-9]*):(0|[1-9][0-9]*)", value)
        if match is None:
            raise ContractError(f"{field} byte-range locator must have form start:end")
        start, end = (int(match.group(1)), int(match.group(2)))
        if end < start:
            raise ContractError(f"{field} byte-range end precedes start")
        if key not in locator_cache:
            locator_cache[key] = source_path.read_bytes()
        payload = locator_cache[key]
        if end > len(payload):
            raise ContractError(f"{field} byte range is outside source")
        try:
            return payload[start:end].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ContractError(f"{field} byte range is not complete UTF-8") from error
    raise ContractError(f"{field} has no audited locator resolver")


def _validate_text_record(
    text: Mapping[str, Any],
    *,
    raw_root: Path,
    field: str,
    source_cache: SourceCache,
    locator_cache: LocatorCache,
) -> set[Path]:
    _require_exact_keys(
        text,
        {"normalization", "raw_empty", "source_items", "fallback", "model_inputs", "aggregation"},
        field=field,
    )
    if text["normalization"] != "unicode_nfc_lf_v1":
        raise ContractError(f"{field}.normalization is unsupported")
    raw_empty = text["raw_empty"]
    if not isinstance(raw_empty, bool):
        raise ContractError(f"{field}.raw_empty must be boolean")
    items = _require_list(text["source_items"], field=f"{field}.source_items")
    if not items:
        raise ContractError(f"{field}.source_items may not be empty; locate the empty raw record explicitly")
    external_paths: set[Path] = set()
    values: list[str] = []
    item_keys = {"source", "locator", "value", "utf8_sha256"}
    for index, raw in enumerate(items):
        item = _require_mapping(raw, field=f"{field}.source_items[{index}]")
        _require_exact_keys(item, item_keys, field=f"{field}.source_items[{index}]")
        source = _require_mapping(item["source"], field=f"{field}.source_items[{index}].source")
        path = _canonical_external_path(
            source,
            field=f"{field}.source_items[{index}].source",
            source_cache=source_cache,
        )
        if not is_within(path, raw_root):
            raise ContractError(f"{field} text source must be under raw data root")
        external_paths.add(path)
        locator = _validate_locator(
            item["locator"], field=f"{field}.source_items[{index}].locator"
        )
        value = item["value"]
        if not isinstance(value, str) or normalize_text(value) != value:
            raise ContractError(f"{field}.source_items[{index}].value is not canonical text")
        if item["utf8_sha256"] != text_utf8_sha256(value):
            raise ContractError(f"{field}.source_items[{index}] UTF-8 SHA-256 mismatch")
        resolved_value = _resolve_locator(
            path,
            locator,
            field=f"{field}.source_items[{index}].locator",
            locator_cache=locator_cache,
        )
        if not isinstance(resolved_value, str) or normalize_text(resolved_value) != value:
            raise ContractError(f"{field}.source_items[{index}] locator does not resolve to recorded text")
        values.append(value)
    expected_empty = all(value == "" for value in values)
    if raw_empty != expected_empty:
        raise ContractError(f"{field}.raw_empty disagrees with located raw text")

    fallback = text["fallback"]
    model_inputs = _require_list(text["model_inputs"], field=f"{field}.model_inputs")
    if raw_empty:
        fallback_record = _require_mapping(fallback, field=f"{field}.fallback")
        _require_exact_keys(
            fallback_record, {"policy_id", "value", "utf8_sha256"}, field=f"{field}.fallback"
        )
        _require_nonempty_string(fallback_record["policy_id"], field=f"{field}.fallback.policy_id")
        fallback_value = _require_nonempty_string(fallback_record["value"], field=f"{field}.fallback.value")
        if normalize_text(fallback_value) != fallback_value:
            raise ContractError(f"{field}.fallback.value is not canonical text")
        if fallback_record["utf8_sha256"] != text_utf8_sha256(fallback_value):
            raise ContractError(f"{field}.fallback UTF-8 SHA-256 mismatch")
        expected_inputs = [("fallback", None, fallback_value)]
    else:
        if fallback is not None:
            raise ContractError(f"{field}.fallback is allowed only for explicitly empty raw text")
        expected_inputs = [("raw", index, value) for index, value in enumerate(values) if value != ""]

    if len(model_inputs) != len(expected_inputs):
        raise ContractError(f"{field}.model_inputs does not exactly cover extraction inputs")
    input_keys = {"origin", "source_item_index", "value", "utf8_sha256"}
    for index, (raw, expected) in enumerate(zip(model_inputs, expected_inputs)):
        item = _require_mapping(raw, field=f"{field}.model_inputs[{index}]")
        _require_exact_keys(item, input_keys, field=f"{field}.model_inputs[{index}]")
        origin, source_index, value = expected
        if item["origin"] != origin or item["source_item_index"] != source_index or item["value"] != value:
            raise ContractError(f"{field}.model_inputs[{index}] is reordered or not source-exact")
        if item["utf8_sha256"] != text_utf8_sha256(value):
            raise ContractError(f"{field}.model_inputs[{index}] UTF-8 SHA-256 mismatch")

    aggregation = _require_mapping(text["aggregation"], field=f"{field}.aggregation")
    _require_exact_keys(
        aggregation, {"method", "input_count", "order_sensitive", "version"}, field=f"{field}.aggregation"
    )
    _require_nonempty_string(aggregation["method"], field=f"{field}.aggregation.method")
    _require_nonempty_string(aggregation["version"], field=f"{field}.aggregation.version")
    if aggregation["input_count"] != len(model_inputs):
        raise ContractError(f"{field}.aggregation.input_count mismatch")
    if not isinstance(aggregation["order_sensitive"], bool):
        raise ContractError(f"{field}.aggregation.order_sensitive must be boolean")
    if len(model_inputs) > 1 and aggregation["method"] == "single":
        raise ContractError(f"{field} has multiple texts but declares single aggregation")
    return external_paths


def _validate_row_structure(
    row: Mapping[str, Any],
    *,
    index: int,
    dataset: str,
    raw_root: Path,
    extractors: Mapping[str, Mapping[str, Any]],
    source_cache: SourceCache,
    locator_cache: LocatorCache,
    label_class_count: int,
) -> tuple[set[Path], np.ndarray]:
    _require_exact_keys(
        row,
        {
            "row_id",
            "global_row",
            "dataset",
            "sample_id",
            "identity",
            "raw_image",
            "text",
            "label",
            "vectors",
            "deterministic_split_positions",
        },
        field=f"rows[{index}]",
    )
    require_content_id(row["row_id"], field=f"rows[{index}].row_id")
    if row["global_row"] != index:
        raise ContractError(f"rows[{index}] global_row is missing or reordered")
    if row["dataset"] != dataset:
        raise ContractError(f"rows[{index}] dataset mismatch")
    _require_nonempty_string(row["sample_id"], field=f"rows[{index}].sample_id")

    identity = _require_mapping(row["identity"], field=f"rows[{index}].identity")
    _require_exact_keys(identity, {"method", "source", "locator"}, field=f"rows[{index}].identity")
    method = _require_nonempty_string(identity["method"], field=f"rows[{index}].identity.method")
    lowered = method.lower()
    if method not in ALLOWED_EXACT_IDENTITY_METHODS or any(token in lowered for token in FORBIDDEN_IDENTITY_TOKENS):
        raise ContractError(
            f"rows[{index}] identity method {method!r} is not exact; label-signature/similarity guessing is forbidden"
        )
    identity_source = _require_mapping(identity["source"], field=f"rows[{index}].identity.source")
    identity_path = _canonical_external_path(
        identity_source,
        field=f"rows[{index}].identity.source",
        source_cache=source_cache,
    )
    if not is_within(identity_path, raw_root):
        raise ContractError(f"rows[{index}] identity authority must come from the raw data root")
    identity_locator = _validate_locator(
        identity["locator"], field=f"rows[{index}].identity.locator"
    )
    resolved_sample_id = _resolve_locator(
        identity_path,
        identity_locator,
        field=f"rows[{index}].identity.locator",
        locator_cache=locator_cache,
    )
    if str(resolved_sample_id) != str(row["sample_id"]):
        raise ContractError(f"rows[{index}] identity locator does not resolve to sample_id")

    image = _require_mapping(row["raw_image"], field=f"rows[{index}].raw_image")
    image_path = _canonical_external_path(
        image, field=f"rows[{index}].raw_image", source_cache=source_cache
    )
    if not is_within(image_path, raw_root):
        raise ContractError(f"rows[{index}] raw image must be under raw data root")
    external_paths = {identity_path, image_path}
    external_paths.update(
        _validate_text_record(
            _require_mapping(row["text"], field=f"rows[{index}].text"),
            raw_root=raw_root,
            field=f"rows[{index}].text",
            source_cache=source_cache,
            locator_cache=locator_cache,
        )
    )

    label = _require_mapping(row["label"], field=f"rows[{index}].label")
    _require_exact_keys(
        label,
        {"source", "locator", "encoding", "value", "vector_sha256"},
        field=f"rows[{index}].label",
    )
    if label["encoding"] != "binary_multihot_uint8_v1":
        raise ContractError(f"rows[{index}] label encoding is unsupported")
    label_source = _require_mapping(label["source"], field=f"rows[{index}].label.source")
    label_path = _canonical_external_path(
        label_source,
        field=f"rows[{index}].label.source",
        source_cache=source_cache,
    )
    if not is_within(label_path, raw_root):
        raise ContractError(f"rows[{index}] label authority must come from the raw data root")
    external_paths.add(label_path)
    label_locator = _validate_locator(label["locator"], field=f"rows[{index}].label.locator")
    resolved_label = _resolve_locator(
        label_path,
        label_locator,
        field=f"rows[{index}].label.locator",
        locator_cache=locator_cache,
    )
    if not isinstance(label["value"], list) or resolved_label != label["value"]:
        raise ContractError(f"rows[{index}] label locator does not resolve to the recorded vector")
    label_vector = np.asarray(label["value"])
    if label_vector.shape != (label_class_count,) or label_vector.dtype.kind not in "biu":
        raise ContractError(f"rows[{index}] raw label vector shape/dtype mismatch")
    if not np.all(np.isin(label_vector, (0, 1))) or int(label_vector.sum()) <= 0:
        raise ContractError(f"rows[{index}] raw label vector must be non-empty binary multi-hot")
    label_vector = np.ascontiguousarray(label_vector, dtype=np.uint8)
    require_sha256(label["vector_sha256"], field=f"rows[{index}].label.vector_sha256")
    if label["vector_sha256"] != feature_row_sha256(label_vector):
        raise ContractError(f"rows[{index}] raw label vector SHA-256 mismatch")

    vectors = _require_mapping(row["vectors"], field=f"rows[{index}].vectors")
    if set(vectors) != set(SEMANTICS):
        raise ContractError(f"rows[{index}] must bind image, text, and multilabel vectors")
    vector_keys = {"shard_id", "shard_row", "dtype", "shape", "sha256", "extractor_id"}
    for semantic in SEMANTICS:
        vector = _require_mapping(vectors[semantic], field=f"rows[{index}].vectors.{semantic}")
        _require_exact_keys(vector, vector_keys, field=f"rows[{index}].vectors.{semantic}")
        _require_nonempty_string(vector["shard_id"], field="shard_id")
        _require_nonnegative_int(vector["shard_row"], field="shard_row")
        dtype = np.dtype(_require_nonempty_string(vector["dtype"], field="dtype"))
        shape = vector["shape"]
        if not isinstance(shape, list) or not shape or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape
        ):
            raise ContractError(f"rows[{index}].vectors.{semantic}.shape is invalid")
        require_sha256(vector["sha256"], field="vector sha256")
        if semantic in {"image", "text"}:
            extractor_id = vector["extractor_id"]
            if extractor_id not in extractors or extractors[extractor_id]["semantic"] != semantic:
                raise ContractError(f"rows[{index}] {semantic} vector references wrong extractor")
            extractor = extractors[extractor_id]
            if dtype.name != np.dtype(extractor["output_dtype"]).name or shape != extractor["output_shape"]:
                raise ContractError(f"rows[{index}] {semantic} dtype/shape differs from extractor")
        else:
            if vector["extractor_id"] != "raw_multilabel_locator" or dtype.name != "uint8":
                raise ContractError(f"rows[{index}] multilabel vector must be uint8 raw-locator data")
            if vector["sha256"] != label["vector_sha256"]:
                raise ContractError(f"rows[{index}] label/vector hashes disagree")

    positions = _require_mapping(
        row["deterministic_split_positions"], field=f"rows[{index}].deterministic_split_positions"
    )
    if set(positions) != set(SPLIT_NAMES):
        raise ContractError(f"rows[{index}] must explicitly record deterministic indT/indQ/indD positions")
    for split, value in positions.items():
        if value is not None:
            _require_nonnegative_int(value, field=f"rows[{index}].deterministic_split_positions.{split}")

    if row["row_id"] != derive_row_id(row):
        raise ContractError(f"rows[{index}] immutable row_id does not bind complete row provenance")
    return external_paths, label_vector


def _validate_labels_contract(labels: Mapping[str, Any], row_count: int, dataset: str) -> int:
    _require_exact_keys(
        labels,
        {
            "authority",
            "row_count",
            "class_count",
            "row_order",
            "selection_use",
        },
        field="labels",
    )
    if labels["authority"] != "raw_locator_per_sample_v1":
        raise ContractError("labels must be rebuilt from a raw locator on every sample")
    if labels["row_count"] != row_count:
        raise ContractError("label source row count differs from manifest")
    class_count = _require_nonnegative_int(labels["class_count"], field="labels.class_count")
    if class_count <= 0:
        raise ContractError("labels.class_count must be positive")
    if dataset not in RAW_REBUILT_LABEL_DIMS or class_count != RAW_REBUILT_LABEL_DIMS[dataset]:
        raise ContractError(f"{dataset} raw-rebuilt label dimension is not the frozen value")
    if labels["row_order"] != "canonical_raw_sample_order" or labels["selection_use"] != "provenance_only_never_model_selection":
        raise ContractError("labels may only be row-ordered provenance, never model-selection input")
    return class_count


def _validate_authority(
    authority: Mapping[str, Any],
    *,
    dataset: str,
    row_count: int,
    sample_ids: Sequence[str],
) -> None:
    _require_exact_keys(
        authority,
        {
            "kind",
            "canonical_order_algorithm",
            "ordered_sample_ids_sha256",
            "process_data_count_comparison",
        },
        field="authority",
    )
    if authority["kind"] != "raw_rebuilt_v1":
        raise ContractError("final visualization authority must be raw_rebuilt_v1")
    if authority["canonical_order_algorithm"] != "nfc_utf8_sample_id_ascending_v1":
        raise ContractError("raw sample canonical-order algorithm is not frozen")
    if dataset not in RAW_REBUILT_COUNTS:
        raise ContractError(f"dataset {dataset!r} has no frozen raw-rebuilt count registry")
    if RAW_REBUILT_COUNTS[dataset]["n_rows"] != row_count:
        raise ContractError("row_count differs from the frozen raw-rebuilt registry")
    if list(sample_ids) != canonical_sample_id_order(sample_ids):
        raise ContractError("raw samples are missing, duplicated, or not in canonical sample_id order")
    if authority["ordered_sample_ids_sha256"] != ordered_ids_sha256(list(sample_ids)):
        raise ContractError("ordered raw sample-ID digest mismatch")
    comparison = authority["process_data_count_comparison"]
    if comparison is not None:
        comparison = _require_mapping(comparison, field="authority.process_data_count_comparison")
        _require_exact_keys(comparison, {"role", "counts"}, field="authority.process_data_count_comparison")
        if comparison["role"] != "counts_only_non_authoritative":
            raise ContractError("ProcessData comparison may contain counts only and is never authoritative")
        counts = _require_mapping(comparison["counts"], field="authority.process_data_count_comparison.counts")
        if set(counts) != {"n_rows", "indT", "indQ", "indD"}:
            raise ContractError("ProcessData comparison contains more than frozen counts")
        if dict(counts) != RAW_REBUILT_COUNTS[dataset]:
            raise ContractError("optional ProcessData count comparison disagrees with frozen raw counts")


def _reject_processdata_authority_paths(
    value: Any, process_root: Path | None, *, field: str = "manifest"
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child_field = f"{field}.{key}"
            if key == "path" and isinstance(item, str) and Path(item).is_absolute():
                candidate = Path(item).resolve(strict=False)
                named_processdata = any(part.casefold() == "processdata" for part in candidate.parts)
                inside_explicit = process_root is not None and is_within(candidate, process_root)
                if named_processdata or inside_explicit:
                    raise ContractError(
                        f"{child_field} references ProcessData; features/IDs/labels/indices cannot be authority"
                    )
            _reject_processdata_authority_paths(item, process_root, field=child_field)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_processdata_authority_paths(item, process_root, field=f"{field}[{index}]")


def _validate_splits(
    bundle: Path,
    descriptor: Mapping[str, Any],
    *,
    inventory: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    dataset: str,
) -> None:
    _require_exact_keys(
        descriptor,
        {"algorithm", "seed", "counts", "assignment_sha256", "artifact", "arrays", "relations"},
        field="deterministic_splits",
    )
    if descriptor["algorithm"] != "kbs-content-hash-split-v1":
        raise ContractError("raw-rebuilt split algorithm is not the preregistered SHA-256 sorter")
    if descriptor["seed"] != RAW_REBUILT_SPLIT_SEED:
        raise ContractError("raw-rebuilt split seed must be exactly 20260822; seed search is forbidden")
    counts = _require_mapping(descriptor["counts"], field="deterministic_splits.counts")
    if dict(counts) != RAW_REBUILT_COUNTS[dataset]:
        raise ContractError("deterministic split counts differ from the frozen dataset registry")
    sample_ids = [str(row["sample_id"]) for row in rows]
    expected_arrays = raw_rebuilt_split_indices(dataset, sample_ids)
    expected_assignment_sha = raw_rebuilt_assignment_sha256(dataset, sample_ids, expected_arrays)
    if descriptor["assignment_sha256"] != expected_assignment_sha:
        raise ContractError("deterministic split assignment SHA-256 mismatch")
    artifact = _require_mapping(descriptor["artifact"], field="deterministic_splits.artifact")
    _require_exact_keys(artifact, {"path", "size", "sha256", "format"}, field="deterministic_splits.artifact")
    if artifact["format"] != "npz":
        raise ContractError("canonical deterministic split artifact must be NPZ")
    relative = _require_inventory_descriptor(artifact, inventory, field="deterministic_splits.artifact")
    path = resolve_contained(bundle, relative.as_posix(), field="deterministic_splits.artifact.path")
    expected_keys = set(SPLIT_NAMES) | {f"{name}_row_ids" for name in SPLIT_NAMES}
    try:
        with np.load(path, allow_pickle=False) as loaded:
            if set(loaded.files) != expected_keys:
                raise ContractError("deterministic split NPZ has missing or unbound arrays")
            canonical_arrays = {name: np.asarray(loaded[name]) for name in SPLIT_NAMES}
            split_row_ids = {name: np.asarray(loaded[f"{name}_row_ids"]) for name in SPLIT_NAMES}
    except (OSError, ValueError) as error:
        if isinstance(error, ContractError):
            raise
        raise ContractError(f"cannot load deterministic split artifact: {error}") from error

    arrays_descriptor = _require_mapping(descriptor["arrays"], field="deterministic_splits.arrays")
    if set(arrays_descriptor) != set(SPLIT_NAMES):
        raise ContractError("official split array descriptors must cover indT/indQ/indD")
    row_ids = [str(row["row_id"]) for row in rows]
    n_rows = len(rows)
    for name in SPLIT_NAMES:
        value = canonical_arrays[name]
        if value.ndim != 1 or value.dtype != np.dtype("int64"):
            raise ContractError(f"canonical {name} must be one-dimensional int64")
        if value.size == 0 or int(value.min()) < 0 or int(value.max()) >= n_rows:
            raise ContractError(f"canonical {name} is empty or outside global rows")
        if np.unique(value).size != value.size:
            raise ContractError(f"canonical {name} contains duplicate rows")
        if not np.array_equal(value, expected_arrays[name]):
            raise ContractError(f"canonical {name} is reordered or differs from the frozen SHA-256 assignment")
        ids = split_row_ids[name]
        if ids.ndim != 1 or ids.dtype.kind not in "US":
            raise ContractError(f"{name}_row_ids must be a string vector")
        expected_ids = np.asarray([row_ids[int(index)] for index in value], dtype=str)
        if not np.array_equal(ids.astype(str), expected_ids):
            raise ContractError(f"{name}_row_ids do not bind official row order")
        array_descriptor = _require_mapping(arrays_descriptor[name], field=f"deterministic_splits.arrays.{name}")
        _require_exact_keys(
            array_descriptor, {"count", "indices_sha256", "row_ids_sha256"}, field=f"deterministic_splits.arrays.{name}"
        )
        if array_descriptor["count"] != int(value.size):
            raise ContractError(f"{name} count mismatch")
        if array_descriptor["indices_sha256"] != int64_array_sha256(value):
            raise ContractError(f"{name} index digest mismatch")
        if array_descriptor["row_ids_sha256"] != ordered_ids_sha256(ids):
            raise ContractError(f"{name} row-ID digest mismatch")

        actual_positions: list[int | None] = [None] * n_rows
        for position, global_row in enumerate(value.tolist()):
            actual_positions[int(global_row)] = position
        declared_positions = [row["deterministic_split_positions"][name] for row in rows]
        if declared_positions != actual_positions:
            raise ContractError(f"per-row {name} positions do not exactly mirror official order")

    relations = _require_mapping(descriptor["relations"], field="deterministic_splits.relations")
    _require_exact_keys(
        relations,
        {"indT_subset_of_indD", "indQ_disjoint_indD", "indQ_disjoint_indT", "indQ_union_indD_full"},
        field="deterministic_splits.relations",
    )
    if any(relations[name] is not True for name in relations):
        raise ContractError("official split relation assertions must all be true")
    sets = {name: set(canonical_arrays[name].tolist()) for name in SPLIT_NAMES}
    if not sets["indT"].issubset(sets["indD"]):
        raise ContractError("official indT is not a subset of indD")
    if sets["indQ"] & sets["indD"] or sets["indQ"] & sets["indT"]:
        raise ContractError("deterministic query rows overlap training/database rows")
    if sets["indQ"] | sets["indD"] != set(range(len(rows))):
        raise ContractError("deterministic indQ and indD do not cover every raw sample exactly once")


def _validate_shards(
    bundle: Path,
    descriptors: Sequence[Any],
    *,
    inventory: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    raw_labels: np.ndarray,
) -> None:
    n_rows = len(rows)
    by_semantic: dict[str, list[tuple[Mapping[str, Any], np.ndarray, np.ndarray]]] = {
        semantic: [] for semantic in SEMANTICS
    }
    shard_ids: set[str] = set()
    expected_keys = {
        "shard_id",
        "semantic",
        "path",
        "format",
        "size",
        "sha256",
        "row_start",
        "row_count",
        "row_ids_key",
        "vector_key",
        "row_ids_sha256",
        "vectors_sha256",
        "dtype",
        "shape",
    }
    for index, raw in enumerate(descriptors):
        descriptor = _require_mapping(raw, field=f"shards[{index}]")
        _require_exact_keys(descriptor, expected_keys, field=f"shards[{index}]")
        shard_id = _require_nonempty_string(descriptor["shard_id"], field="shard_id")
        if shard_id in shard_ids:
            raise ContractError(f"duplicate shard_id {shard_id!r}")
        shard_ids.add(shard_id)
        semantic = descriptor["semantic"]
        if semantic not in SEMANTICS:
            raise ContractError("shard semantic must be image, text, or multilabel")
        _require_inventory_descriptor(descriptor, inventory, field=f"shards[{index}]")
        row_start = _require_nonnegative_int(descriptor["row_start"], field="row_start")
        row_count = _require_nonnegative_int(descriptor["row_count"], field="row_count")
        if row_count == 0:
            raise ContractError("empty feature shards are forbidden")
        _require_nonempty_string(descriptor["row_ids_key"], field="row_ids_key")
        _require_nonempty_string(descriptor["vector_key"], field="vector_key")
        require_sha256(descriptor["row_ids_sha256"], field="row_ids_sha256")
        require_sha256(descriptor["vectors_sha256"], field="vectors_sha256")
        dtype = np.dtype(_require_nonempty_string(descriptor["dtype"], field="dtype"))
        shape = descriptor["shape"]
        if not isinstance(shape, list) or not shape or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape
        ):
            raise ContractError("shard shape must be a non-empty positive row shape")
        row_ids, vectors = _load_vector_shard(bundle, descriptor)
        if row_ids.shape != (row_count,) or vectors.shape != (row_count, *shape):
            raise ContractError(f"shard {shard_id} row count/shape mismatch")
        if vectors.dtype.name != dtype.name:
            raise ContractError(f"shard {shard_id} dtype mismatch")
        if ordered_ids_sha256(row_ids) != descriptor["row_ids_sha256"]:
            raise ContractError(f"shard {shard_id} ordered row-ID digest mismatch")
        if numeric_array_sha256(vectors) != descriptor["vectors_sha256"]:
            raise ContractError(f"shard {shard_id} vector content digest mismatch")
        if semantic in {"image", "text"}:
            if dtype.kind != "f" or not np.all(np.isfinite(vectors)):
                raise ContractError(f"shard {shard_id} features must be finite floating point")
            flat = vectors.reshape(row_count, -1)
            if np.any(np.linalg.norm(flat.astype(np.float64), axis=1) <= 0.0):
                raise ContractError(f"shard {shard_id} contains an all-zero feature row")
        else:
            if dtype.name != "uint8" or not np.all(np.isin(vectors, (0, 1))):
                raise ContractError("multilabel shards must contain binary uint8 vectors")
        by_semantic[str(semantic)].append((descriptor, row_ids, vectors))

    expected_row_ids = [str(row["row_id"]) for row in rows]
    for semantic in SEMANTICS:
        items = sorted(by_semantic[semantic], key=lambda item: int(item[0]["row_start"]))
        if not items:
            raise ContractError(f"missing {semantic} feature shards")
        cursor = 0
        concatenated: list[np.ndarray] = []
        seen_refs: set[tuple[str, int]] = set()
        for descriptor, row_ids, vectors in items:
            if descriptor["row_start"] != cursor:
                raise ContractError(f"{semantic} shards have a missing, overlapping, or reordered range")
            stop = cursor + int(descriptor["row_count"])
            expected_ids = np.asarray(expected_row_ids[cursor:stop], dtype=str)
            if not np.array_equal(row_ids.astype(str), expected_ids):
                raise ContractError(f"{semantic} shard row_ids are reordered")
            for local_row in range(int(descriptor["row_count"])):
                global_row = cursor + local_row
                vector_ref = rows[global_row]["vectors"][semantic]
                if vector_ref["shard_id"] != descriptor["shard_id"] or vector_ref["shard_row"] != local_row:
                    raise ContractError(f"row {global_row} {semantic} shard reference is missing or reordered")
                reference = (str(descriptor["shard_id"]), local_row)
                if reference in seen_refs:
                    raise ContractError(f"duplicate {semantic} shard-row reference")
                seen_refs.add(reference)
                if vector_ref["dtype"] != descriptor["dtype"] or vector_ref["shape"] != descriptor["shape"]:
                    raise ContractError(f"row {global_row} {semantic} dtype/shape differs from shard")
                if vector_ref["sha256"] != feature_row_sha256(vectors[local_row]):
                    raise ContractError(f"row {global_row} {semantic} vector SHA-256 mismatch")
            concatenated.append(vectors)
            cursor = stop
        if cursor != n_rows or len(seen_refs) != n_rows:
            raise ContractError(f"{semantic} shards do not cover every row exactly once")
        full = np.concatenate(concatenated, axis=0)
        if semantic == "multilabel" and not np.array_equal(full, raw_labels):
            raise ContractError("multilabel vectors differ from raw locators or canonical sample order")


@dataclass
class _TraceResolverCache:
    """Parsed raw sources, cached once per path for full-dataset validation."""

    json_documents: dict[Path, Any] = field(default_factory=dict)
    mat_documents: dict[Path, Mapping[str, Any]] = field(default_factory=dict)
    text_lines: dict[Path, list[str]] = field(default_factory=dict)
    text_memberships: dict[Path, frozenset[str]] = field(default_factory=dict)
    coco_labels: dict[tuple[Path, tuple[int, ...]], dict[int, set[int]]] = field(
        default_factory=dict
    )


def _trace_file_fingerprint(
    path: Path,
    cache: MutableMapping[Path, tuple[int, int, str]],
) -> tuple[int, str]:
    """Hash each shared dependency exactly once per validation run."""

    cached = cache.get(path)
    if cached is not None:
        return cached[0], cached[2]
    stat = path.stat()
    digest = sha256_file(path)
    cache[path] = (stat.st_size, stat.st_mtime_ns, digest)
    return stat.st_size, digest


def _trace_resolve_hashed_source(
    item: Mapping[str, Any],
    *,
    field_name: str,
    oral_root: Path,
    process_root: Path | None,
    file_cache: MutableMapping[Path, tuple[int, int, str]],
) -> Path:
    raw_path = _require_nonempty_string(item.get("path"), field=f"{field_name}.path")
    expected_size = _require_nonnegative_int(item.get("bytes"), field=f"{field_name}.bytes")
    expected_sha = require_sha256(item.get("sha256"), field=f"{field_name}.sha256")
    path = Path(raw_path)
    if not path.is_absolute():
        raise ContractError(f"{field_name}.path must be absolute")
    cached = file_cache.get(path)
    if cached is not None:
        if cached[0] != expected_size:
            raise ContractError(f"trace raw source size mismatch: {path}")
        if cached[2] != expected_sha:
            raise ContractError(f"trace raw source SHA-256 mismatch: {path}")
        return path
    try:
        path = path.resolve(strict=True)
    except OSError as error:
        raise ContractError(f"{field_name}.path is missing: {raw_path}") from error
    if not path.is_file() or not is_within(path, oral_root):
        raise ContractError(f"{field_name} must be a regular file under OralData")
    if any(part.casefold() == "processdata" for part in path.parts) or (
        process_root is not None and is_within(path, process_root)
    ):
        raise ContractError(f"{field_name} references forbidden ProcessData")
    actual_size, actual_sha = _trace_file_fingerprint(path, file_cache)
    if actual_size != expected_size:
        raise ContractError(f"trace raw source size mismatch: {path}")
    if actual_sha != expected_sha:
        raise ContractError(f"trace raw source SHA-256 mismatch: {path}")
    return path


def _trace_json(path: Path, cache: _TraceResolverCache) -> Any:
    if path not in cache.json_documents:
        cache.json_documents[path] = _decode_json(path.read_bytes(), field=str(path))
    return cache.json_documents[path]


def _trace_mat(path: Path, cache: _TraceResolverCache) -> Mapping[str, Any]:
    if path not in cache.mat_documents:
        try:
            import scipy.io as sio
        except ImportError as error:
            raise ContractError("SciPy is required to execute trace MAT locators") from error
        try:
            cache.mat_documents[path] = sio.loadmat(str(path))
        except Exception as error:
            raise ContractError(f"cannot parse trace MAT source {path}: {error}") from error
    return cache.mat_documents[path]


def _trace_lines(path: Path, cache: _TraceResolverCache) -> list[str]:
    if path not in cache.text_lines:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                cache.text_lines[path] = [line.strip() for line in handle]
        except OSError as error:
            raise ContractError(f"cannot read trace text source {path}: {error}") from error
    return cache.text_lines[path]


def _trace_membership(path: Path, cache: _TraceResolverCache) -> frozenset[str]:
    if path not in cache.text_memberships:
        cache.text_memberships[path] = frozenset(
            value for value in _trace_lines(path, cache) if value
        )
    return cache.text_memberships[path]


def _trace_json_pointer(document: Any, pointer: str, *, field_name: str) -> Any:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ContractError(f"{field_name} must be an absolute JSON pointer")
    value = document
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not token.isdigit() or (token != "0" and token.startswith("0")):
                raise ContractError(f"{field_name} has a non-canonical array index")
            index = int(token)
            if index >= len(value):
                raise ContractError(f"{field_name} points outside its JSON array")
            value = value[index]
        elif isinstance(value, Mapping):
            if token not in value:
                raise ContractError(f"{field_name} points to a missing JSON key")
            value = value[token]
        else:
            raise ContractError(f"{field_name} traverses a scalar JSON value")
    return value


def _trace_dense_row(array: Any, row: int, *, field_name: str) -> np.ndarray:
    if row < 0 or not hasattr(array, "shape") or len(array.shape) != 2 or row >= array.shape[0]:
        raise ContractError(f"{field_name} row is outside its two-dimensional source")
    value = array[row]
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value).reshape(-1)


def _trace_nonnegative_int_list(value: Any, *, field_name: str) -> list[int]:
    values = _require_list(value, field=field_name)
    result = [_require_nonnegative_int(item, field=f"{field_name}[]") for item in values]
    if result != sorted(set(result)):
        raise ContractError(f"{field_name} must be strictly increasing and duplicate-free")
    return result


def _validate_trace_text_sources(
    record: Mapping[str, Any],
    *,
    row_index: int,
    dataset: str,
    oral_root: Path,
    process_root: Path | None,
    file_cache: MutableMapping[Path, tuple[int, int, str]],
    resolver_cache: _TraceResolverCache,
    dependency_descriptors: MutableMapping[Path, tuple[int, str]],
) -> set[Path]:
    sources = _require_list(record.get("raw_text_sources"), field=f"trace row {row_index} raw_text_sources")
    if not sources:
        raise ContractError(f"trace row {row_index} must bind at least one raw text source")
    raw_text = _require_mapping(record.get("raw_text"), field=f"trace row {row_index} raw_text")
    encoded_texts = _require_list(record.get("encoded_texts"), field=f"trace row {row_index} encoded_texts")
    resolved: list[dict[str, Any]] = []
    paths: set[Path] = set()
    for source_index, raw_source in enumerate(sources):
        field_name = f"trace row {row_index} raw_text_sources[{source_index}]"
        source = _require_mapping(raw_source, field=field_name)
        _require_exact_keys(
            source,
            {"path", "kind", "locator", "content_sha256", "bytes", "sha256"},
            field=field_name,
        )
        path = _trace_resolve_hashed_source(
            source,
            field_name=field_name,
            oral_root=oral_root,
            process_root=process_root,
            file_cache=file_cache,
        )
        paths.add(path)
        descriptor = (int(source["bytes"]), str(source["sha256"]))
        if path in dependency_descriptors and dependency_descriptors[path] != descriptor:
            raise ContractError(f"inconsistent trace dependency descriptor for {path}")
        dependency_descriptors[path] = descriptor
        kind = _require_nonempty_string(source.get("kind"), field=f"{field_name}.kind")
        locator = _require_mapping(source.get("locator"), field=f"{field_name}.locator")
        content_sha = require_sha256(source.get("content_sha256"), field=f"{field_name}.content_sha256")
        item: dict[str, Any] = {
            "kind": kind,
            "path": path,
            "locator": locator,
            "content_sha256": content_sha,
        }
        if kind == "json_annotation":
            _require_exact_keys(
                locator,
                {"json_pointer", "annotation_id", "image_id"},
                field=f"{field_name}.locator",
            )
            annotation = _require_mapping(
                _trace_json_pointer(
                    _trace_json(path, resolver_cache),
                    locator["json_pointer"],
                    field_name=f"{field_name}.locator.json_pointer",
                ),
                field=f"{field_name} annotation",
            )
            annotation_id = _require_nonnegative_int(locator["annotation_id"], field=f"{field_name}.annotation_id")
            image_id = _require_nonnegative_int(locator["image_id"], field=f"{field_name}.image_id")
            if annotation.get("id") != annotation_id or annotation.get("image_id") != image_id:
                raise ContractError(f"{field_name} annotation identity does not match its locator")
            caption = _require_nonempty_string(annotation.get("caption"), field=f"{field_name} caption")
            item["payload"] = {
                "annotation_id": annotation_id,
                "image_id": image_id,
                "caption": caption,
            }
        elif kind == "text_file_lines":
            _require_exact_keys(
                locator,
                {"mir_image_id", "filtered_line_indices", "index_base"},
                field=f"{field_name}.locator",
            )
            if locator["index_base"] != 0:
                raise ContractError(f"{field_name} must use zero-based line indices")
            image_id = _require_nonnegative_int(locator["mir_image_id"], field=f"{field_name}.mir_image_id")
            indices = _trace_nonnegative_int_list(
                locator["filtered_line_indices"], field_name=f"{field_name}.filtered_line_indices"
            )
            lines = _trace_lines(path, resolver_cache)
            if indices and indices[-1] >= len(lines):
                raise ContractError(f"{field_name} filtered line index is out of range")
            item["payload"] = {
                "raw_tags": [value for value in lines if value],
                "filtered_tags": [lines[index] for index in indices],
            }
            item["mir_image_id"] = image_id
        elif kind == "text_line_selection":
            _require_exact_keys(locator, {"line_indices", "index_base"}, field=f"{field_name}.locator")
            if locator["index_base"] != 0:
                raise ContractError(f"{field_name} must use zero-based line indices")
            indices = _trace_nonnegative_int_list(locator["line_indices"], field_name=f"{field_name}.line_indices")
            lines = _trace_lines(path, resolver_cache)
            if indices and indices[-1] >= len(lines):
                raise ContractError(f"{field_name} selected line index is out of range")
            item["payload"] = [lines[index] for index in indices]
            item["active_columns"] = indices
        elif kind == "text_line":
            _require_exact_keys(
                locator,
                {"line_index", "index_base", "photo_id"},
                field=f"{field_name}.locator",
            )
            if locator["index_base"] != 0:
                raise ContractError(f"{field_name} must use zero-based line indices")
            line_index = _require_nonnegative_int(locator["line_index"], field=f"{field_name}.line_index")
            photo_id = _require_nonempty_string(locator["photo_id"], field=f"{field_name}.photo_id")
            lines = _trace_lines(path, resolver_cache)
            if line_index >= len(lines):
                raise ContractError(f"{field_name} selected line index is out of range")
            fields = lines[line_index].split()
            if not fields or fields[0] != photo_id:
                raise ContractError(f"{field_name} photo ID does not match the raw line")
            item["payload"] = {"photo_id": photo_id, "full_user_tags": fields[1:]}
        elif kind == "mat_row":
            _require_exact_keys(
                locator,
                {"variable", "row", "index_base", "active_columns"},
                field=f"{field_name}.locator",
            )
            if locator["index_base"] != 0:
                raise ContractError(f"{field_name} must use a zero-based MAT row")
            variable = _require_nonempty_string(locator["variable"], field=f"{field_name}.variable")
            row = _require_nonnegative_int(locator["row"], field=f"{field_name}.row")
            document = _trace_mat(path, resolver_cache)
            if variable not in document:
                raise ContractError(f"{field_name} MAT variable is missing")
            values = _trace_dense_row(document[variable], row, field_name=field_name)
            if values.size and not np.all(np.isin(values, (0, 1))):
                raise ContractError(f"{field_name} MAT text row is not binary")
            actual_columns = np.flatnonzero(values).astype(int).tolist()
            declared_columns = _trace_nonnegative_int_list(
                locator["active_columns"], field_name=f"{field_name}.active_columns"
            )
            if actual_columns != declared_columns:
                raise ContractError(f"{field_name} active columns differ from the raw MAT row")
            item["active_columns"] = actual_columns
        else:
            raise ContractError(f"trace text locator kind {kind!r} has no audited resolver")
        resolved.append(item)

    kinds = [item["kind"] for item in resolved]
    if dataset == "nuswide":
        if sorted(kinds) != sorted(["mat_row", "text_line_selection", "text_line"]):
            raise ContractError("NUS trace row must bind exactly the MAT, TagList, and All_Tags sources")
        mat_item = next(item for item in resolved if item["kind"] == "mat_row")
        selection = next(item for item in resolved if item["kind"] == "text_line_selection")
        full = next(item for item in resolved if item["kind"] == "text_line")
        if mat_item["active_columns"] != selection["active_columns"]:
            raise ContractError("NUS MAT active columns and TagList line selection disagree")
        tags = selection["payload"]
        mat_item["payload"] = {"active_columns": mat_item["active_columns"], "tags": tags}
        if raw_text.get("baseline_top1k_tags") != tags or raw_text.get("full_user_tags") != full["payload"]["full_user_tags"]:
            raise ContractError("NUS raw text record differs from executed source locators")
        fallback = not tags
        if raw_text.get("baseline_fallback") is not fallback or raw_text.get("full_tags_not_encoded") is not True:
            raise ContractError("NUS empty-text fallback metadata is invalid")
        expected_encoded = tags if tags else ["a generic photo"]
        if encoded_texts != expected_encoded:
            raise ContractError("NUS encoded text inputs differ from the audited raw sources")
    elif dataset in {"mscoco", "synthetic"}:
        if not kinds or any(kind != "json_annotation" for kind in kinds):
            raise ContractError(f"{dataset} trace rows require audited JSON annotation text locators")
        captions = [item["payload"]["caption"] for item in resolved]
        raw_captions = raw_text.get("captions")
        if raw_captions is None and dataset == "synthetic":
            raw_captions = [raw_text.get("caption")]
        if raw_captions != captions or encoded_texts != captions:
            raise ContractError(f"{dataset} captions differ from executed JSON locators")
    elif dataset == "mirflickr":
        if kinds != ["text_file_lines"]:
            raise ContractError("MIR trace rows require exactly one raw tag-file locator")
        payload = resolved[0]["payload"]
        if str(resolved[0]["mir_image_id"]) != str(record.get("source_id")):
            raise ContractError("MIR tag locator image ID differs from canonical source_id")
        if raw_text.get("raw_tags") != payload["raw_tags"] or raw_text.get("filtered_tags") != payload["filtered_tags"]:
            raise ContractError("MIR raw/filtered tags differ from the executed tag-file locator")
        if raw_text.get("filter_rule") != "common_tags count >= 20":
            raise ContractError("MIR tag filter rule is not frozen")
        if encoded_texts != [f"a photo of {tag}" for tag in payload["filtered_tags"]]:
            raise ContractError("MIR encoded prompts differ from the audited filtered tags")

    for item in resolved:
        if "payload" not in item:
            raise ContractError("trace text locator could not be independently resolved")
        observed = sha256_bytes(canonical_json_bytes(item["payload"]))
        if observed != item["content_sha256"]:
            raise ContractError(f"trace raw text locator content SHA-256 mismatch: {item['path']}")
    return paths


def _trace_merkle_root(hex_leaves: Sequence[str]) -> str:
    if not hex_leaves:
        return sha256_bytes(b"")
    level = [bytes.fromhex(value) for value in hex_leaves]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            bytes.fromhex(sha256_bytes(level[index] + level[index + 1]))
            for index in range(0, len(level), 2)
        ]
    return level[0].hex()


def _validate_trace_dependency_summary(
    raw_summary: Any,
    descriptors: Mapping[Path, tuple[int, str]],
) -> None:
    summary = _require_mapping(raw_summary, field="trace raw_dependency_inventory")
    _require_exact_keys(
        summary,
        {"algorithm", "files", "total_bytes", "merkle_root_sha256", "summary_sha256"},
        field="trace raw_dependency_inventory",
    )
    if summary["algorithm"] != "sorted-path-sha256-merkle-v1":
        raise ContractError("trace raw dependency Merkle algorithm is not frozen")
    ordered = []
    for path in sorted(descriptors, key=lambda value: str(value)):
        size, digest = descriptors[path]
        ordered.append({"path": str(path), "bytes": size, "sha256": digest})
    leaves = [sha256_bytes(canonical_json_bytes(item)) for item in ordered]
    expected_body = {
        "algorithm": "sorted-path-sha256-merkle-v1",
        "files": len(ordered),
        "total_bytes": sum(int(item["bytes"]) for item in ordered),
        "merkle_root_sha256": _trace_merkle_root(leaves),
    }
    expected_sha = sha256_bytes(canonical_json_bytes(expected_body))
    if any(summary.get(key) != value for key, value in expected_body.items()):
        raise ContractError("trace raw dependency inventory does not match all row sources")
    if summary.get("summary_sha256") != expected_sha:
        raise ContractError("trace raw dependency inventory summary SHA-256 mismatch")


def _trace_binary_label(value: Any, *, dimension: int, field_name: str) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    array = np.asarray(value).reshape(-1)
    if array.size != dimension or not np.all(np.isin(array, (0, 1))):
        raise ContractError(f"{field_name} is not a {dimension}-dimensional binary label row")
    result = array.astype(np.uint8, copy=False)
    if int(result.sum()) == 0:
        raise ContractError(f"{field_name} is an empty multi-label row")
    return result


def _trace_coco_labels(
    path: Path,
    category_ids: tuple[int, ...],
    cache: _TraceResolverCache,
) -> dict[int, set[int]]:
    key = (path, category_ids)
    if key not in cache.coco_labels:
        document = _require_mapping(_trace_json(path, cache), field=f"COCO labels {path}")
        annotations = _require_list(document.get("annotations"), field=f"COCO annotations {path}")
        allowed = set(category_ids)
        by_image: dict[int, set[int]] = {}
        for index, raw in enumerate(annotations):
            annotation = _require_mapping(raw, field=f"COCO annotation {index}")
            image_id = _require_nonnegative_int(annotation.get("image_id"), field="COCO image_id")
            category_id = _require_nonnegative_int(annotation.get("category_id"), field="COCO category_id")
            if category_id not in allowed:
                raise ContractError(f"COCO annotation category {category_id} is outside the frozen axis")
            by_image.setdefault(image_id, set()).add(category_id)
        cache.coco_labels[key] = by_image
    return cache.coco_labels[key]


def _validate_trace_label_source(
    record: Mapping[str, Any],
    *,
    row_index: int,
    dataset: str,
    adapter: Mapping[str, Any],
    observed_label: np.ndarray,
    oral_root: Path,
    resolver_cache: _TraceResolverCache,
    dependency_descriptors: Mapping[Path, tuple[int, str]],
) -> None:
    metadata = _require_mapping(record.get("metadata"), field=f"trace row {row_index} metadata")
    source = _require_mapping(metadata.get("label_source"), field=f"trace row {row_index} label_source")
    dimension = RAW_REBUILT_LABEL_DIMS[dataset]
    expected: np.ndarray
    if dataset == "nuswide":
        path = Path(_require_nonempty_string(source.get("path"), field="NUS label_source.path")).resolve(strict=True)
        if path not in dependency_descriptors or not is_within(path, oral_root):
            raise ContractError("NUS label source is not a hashed OralData dependency")
        if path.name != "labels.nuswide-tc21.mat" or source.get("variable") != "labels":
            raise ContractError("NUS label source must be labels.nuswide-tc21.mat variable labels")
        if source.get("index_base") != 0 or source.get("protocol") != "TC21":
            raise ContractError("NUS label source must use the zero-based TC21 protocol")
        raw_row = _require_nonnegative_int(source.get("row"), field="NUS label row")
        if raw_row != record.get("source_index"):
            raise ContractError("NUS label row differs from canonical raw source_index")
        document = _trace_mat(path, resolver_cache)
        if "labels" not in document or tuple(document["labels"].shape) != (269_648, 21):
            raise ContractError("NUS TC21 label matrix must have frozen shape 269648x21")
        expected = _trace_binary_label(
            _trace_dense_row(document["labels"], raw_row, field_name="NUS TC21 labels"),
            dimension=dimension,
            field_name="NUS TC21 label row",
        )
    elif dataset == "mscoco":
        path = Path(_require_nonempty_string(source.get("path"), field="COCO label_source.path")).resolve(strict=True)
        if path not in dependency_descriptors or not is_within(path, oral_root):
            raise ContractError("COCO label source is not a hashed OralData dependency")
        image_id = _require_nonnegative_int(source.get("image_id"), field="COCO label image_id")
        if str(image_id) != str(record.get("source_id")):
            raise ContractError("COCO label image_id differs from canonical source_id")
        raw_categories = _require_list(adapter.get("category_ids"), field="COCO category_ids")
        category_ids = tuple(_require_nonnegative_int(value, field="COCO category_id") for value in raw_categories)
        if len(category_ids) != 80 or len(set(category_ids)) != 80:
            raise ContractError("COCO adapter must freeze exactly 80 unique official category IDs")
        memberships = _trace_coco_labels(path, category_ids, resolver_cache).get(image_id, set())
        expected = np.asarray([int(value in memberships) for value in category_ids], dtype=np.uint8)
        expected = _trace_binary_label(expected, dimension=dimension, field_name="COCO official label row")
    elif dataset == "mirflickr":
        root = Path(_require_nonempty_string(source.get("annotation_root"), field="MIR annotation_root")).resolve(strict=True)
        if not root.is_dir() or not is_within(root, oral_root):
            raise ContractError("MIR annotation root is not an OralData directory")
        raw_names = _require_list(adapter.get("label_names"), field="MIR label_names")
        if len(raw_names) != 24 or len(set(raw_names)) != 24 or not all(isinstance(value, str) and value for value in raw_names):
            raise ContractError("MIR adapter must freeze exactly 24 unique label names")
        source_id = _require_nonempty_string(record.get("source_id"), field="MIR source_id")
        values = []
        positive_names = []
        for name in raw_names:
            path = (root / f"{name}.txt").resolve(strict=True)
            if path not in dependency_descriptors:
                raise ContractError("MIR label file is not present in the hashed dependency inventory")
            present = source_id in _trace_membership(path, resolver_cache)
            values.append(int(present))
            if present:
                positive_names.append(f"{name}.txt")
        declared_positive = _require_list(source.get("positive_label_files"), field="MIR positive_label_files")
        if declared_positive != positive_names:
            raise ContractError("MIR declared positive labels differ from raw annotation membership")
        expected = _trace_binary_label(values, dimension=dimension, field_name="MIR official label row")
    elif dataset == "synthetic":
        path = Path(_require_nonempty_string(source.get("path"), field="synthetic label_source.path")).resolve(strict=True)
        if path not in dependency_descriptors or not is_within(path, oral_root):
            raise ContractError("synthetic label source is not a hashed OralData dependency")
        if source.get("kind") != "json_pointer":
            raise ContractError("synthetic label source requires the audited json_pointer kind")
        value = _trace_json_pointer(
            _trace_json(path, resolver_cache),
            source.get("json_pointer"),
            field_name="synthetic label_source.json_pointer",
        )
        expected = _trace_binary_label(value, dimension=dimension, field_name="synthetic raw label row")
    else:
        raise ContractError(f"trace dataset {dataset!r} has no audited label resolver")
    if not np.array_equal(expected, np.asarray(observed_label, dtype=np.uint8)):
        raise ContractError(f"trace row {row_index} label differs from its OralData locator")


def _validate_trace_nus_identity_sources(
    records: Sequence[Mapping[str, Any]],
    *,
    inventory_paths: set[Path],
    resolver_cache: _TraceResolverCache,
) -> None:
    by_name = {path.name: path for path in inventory_paths}
    clean_path = by_name.get("clean_id.nuswide.tc21.mat")
    image_list_path = by_name.get("Imagelist.txt")
    if clean_path is None or image_list_path is None:
        raise ContractError("NUS raw identity requires clean_id.nuswide.tc21.mat and Imagelist.txt")
    clean_document = _trace_mat(clean_path, resolver_cache)
    if "clean_id" not in clean_document:
        raise ContractError("NUS clean_id source has no clean_id variable")
    clean_raw = np.asarray(clean_document["clean_id"]).reshape(-1)
    if clean_raw.size != 195_834 or not np.all(np.equal(clean_raw, np.floor(clean_raw))):
        raise ContractError("NUS clean_id must contain exactly 195834 integral raw indices")
    clean = clean_raw.astype(np.int64)
    if clean.tolist() != sorted(set(clean.tolist())) or (clean.size and (clean.min() < 0 or clean.max() >= 269_648)):
        raise ContractError("NUS clean_id must be strictly ascending unique zero-based raw indices")
    source_indices = np.asarray([record.get("source_index", -1) for record in records], dtype=np.int64)
    if not np.array_equal(source_indices, clean):
        raise ContractError("NUS canonical rows differ from the TC21 clean_id source")
    image_lines = _trace_lines(image_list_path, resolver_cache)
    if len(image_lines) != 269_648:
        raise ContractError("NUS Imagelist.txt must contain exactly 269648 raw rows")
    for record, raw_index in zip(records, clean.tolist()):
        metadata = _require_mapping(record.get("metadata"), field="NUS row metadata")
        photo_id = _require_nonempty_string(metadata.get("photo_id"), field="NUS photo_id")
        expected_source_id = f"raw-index:{raw_index}|photo-id:{photo_id}"
        if record.get("source_id") != expected_source_id or metadata.get("clean_id") != raw_index:
            raise ContractError("NUS composite canonical source_id differs from raw clean_id/photo_id")
        listed_name = Path(image_lines[raw_index].replace("\\", "/").rsplit("/", 1)[-1]).name
        if metadata.get("image_list_name") != listed_name or Path(str(record.get("image_path"))).name != listed_name:
            raise ContractError("NUS raw image path differs from the authoritative ImageList row")
        if Path(listed_name).stem.rsplit("_", 1)[-1] != photo_id:
            raise ContractError("NUS photo_id differs from the authoritative ImageList row")


def _validate_trace_canonical_order(dataset: str, records: Sequence[Mapping[str, Any]]) -> None:
    source_ids = [str(record.get("source_id", "")) for record in records]
    source_indices = [int(record.get("source_index", -1)) for record in records]
    if len(set(source_ids)) != len(source_ids) or any(not value for value in source_ids):
        raise ContractError("trace canonical source IDs are empty or duplicated")
    if len(set(source_indices)) != len(source_indices) or any(value < 0 for value in source_indices):
        raise ContractError("trace canonical source indices are invalid or duplicated")
    if dataset in {"mirflickr", "mscoco"}:
        try:
            numeric = [int(value) for value in source_ids]
        except ValueError as error:
            raise ContractError(f"{dataset} canonical source IDs must be integers") from error
        if numeric != sorted(numeric):
            raise ContractError(f"{dataset} rows are not in canonical ascending raw ID order")
    elif dataset in {"nuswide", "synthetic"}:
        if source_indices != sorted(source_indices):
            raise ContractError(f"{dataset} rows are not in canonical ascending raw source-index order")
    else:
        raise ContractError(f"trace dataset {dataset!r} has no frozen canonical-order rule")


def validate_visualization_trace_bundle(
    run_dir: Path,
    forbidden_process_data_root: Path | None = None,
) -> ValidationReport:
    """Validate the actual ``visualization_trace`` extraction bundle format."""

    try:
        from visualization_trace.core import canonical_row_id as trace_canonical_row_id
        from visualization_trace.extraction import load_trace_bundle, verify_trace_bundle
    except ImportError as error:
        raise ContractError("visualization_trace package is unavailable") from error
    root = run_dir.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ContractError("visualization_trace run path must be a directory")
    process = (
        None
        if forbidden_process_data_root is None
        else forbidden_process_data_root.expanduser().resolve(strict=False)
    )
    if process is not None and (is_within(root, process) or is_within(process, root)):
        raise ContractError("visualization_trace output must be completely outside ProcessData")
    if any(part.casefold() == "processdata" for part in root.parts):
        raise ContractError("visualization_trace output path may not be under ProcessData")
    try:
        verification = verify_trace_bundle(root)
        image, text, labels, records, complete = load_trace_bundle(root)
    except Exception as error:
        raise ContractError(f"visualization_trace sealed-bundle verification failed: {error}") from error

    contract = _require_mapping(verification["contract"], field="trace contract")
    adapter = _require_mapping(contract.get("adapter"), field="trace contract.adapter")
    dataset = _require_nonempty_string(adapter.get("dataset"), field="trace adapter dataset")
    if dataset not in RAW_REBUILT_COUNTS:
        raise ContractError(f"trace dataset {dataset!r} has no frozen raw-rebuilt registry")
    row_count = _require_nonnegative_int(adapter.get("rows"), field="trace adapter rows")
    if row_count != RAW_REBUILT_COUNTS[dataset]["n_rows"] or int(complete.get("rows", -1)) != row_count:
        raise ContractError("trace row count differs from the frozen raw-rebuilt registry")
    if adapter.get("identity_chain") not in {"OralData only", "OralData official JSON only"}:
        raise ContractError("trace identity chain is not raw OralData authority")
    if not str(adapter.get("process_data_role", "")).startswith("none;"):
        raise ContractError("trace adapter assigns a forbidden role to ProcessData")
    data_root = Path(_require_nonempty_string(adapter.get("data_root"), field="trace adapter data_root"))
    oral_root = (data_root / "OralData").resolve(strict=True)
    if not oral_root.is_dir():
        raise ContractError("trace OralData root is missing")
    if is_within(root, oral_root) or is_within(oral_root, root):
        raise ContractError("trace output must be completely separate from OralData")

    source_artifacts_raw = adapter.get("source_artifacts")
    if not isinstance(source_artifacts_raw, list) or not all(
        isinstance(item, str) and item for item in source_artifacts_raw
    ):
        raise ContractError("trace adapter source_artifacts are incomplete")
    source_inventory = _require_list(contract.get("source_inventory"), field="trace source_inventory")
    inventory_paths: set[Path] = set()
    file_cache: dict[Path, tuple[int, int, str]] = {}
    resolver_cache = _TraceResolverCache()
    dependency_descriptors: dict[Path, tuple[int, str]] = {}
    for index, raw in enumerate(source_inventory):
        item = _require_mapping(raw, field=f"trace source_inventory[{index}]")
        _require_exact_keys(item, {"path", "bytes", "sha256"}, field=f"trace source_inventory[{index}]")
        path = _trace_resolve_hashed_source(
            item,
            field_name=f"trace source_inventory[{index}]",
            oral_root=oral_root,
            process_root=process,
            file_cache=file_cache,
        )
        if path in inventory_paths:
            raise ContractError("trace source inventory contains duplicate paths")
        inventory_paths.add(path)
        dependency_descriptors[path] = (int(item["bytes"]), str(item["sha256"]))
    declared_sources = {Path(value).resolve(strict=True) for value in source_artifacts_raw}
    if declared_sources != inventory_paths:
        raise ContractError("trace adapter source_artifacts and hashed inventory disagree")
    if dataset == "nuswide":
        basenames = {path.name for path in inventory_paths}
        if "labels.nuswide-tc21.mat" not in basenames or "clean_id.nuswide.tc21.mat" not in basenames:
            raise ContractError("NUS-WIDE raw rebuild requires TC21 labels and the TC21 clean_id source")
        if "labels.nuswide.mat" in basenames:
            raise ContractError("NUS-WIDE 81-hot labels.nuswide.mat is forbidden; TC21 is required")
        if adapter.get("label_dimension") != 21 or adapter.get("label_protocol") != "NUS-WIDE TC21":
            raise ContractError("NUS-WIDE adapter contract must freeze the TC21 21-hot protocol")

    if not (
        image.ndim == text.ndim == labels.ndim == 2
        and image.shape == text.shape
        and image.shape[0] == labels.shape[0] == row_count == len(records)
    ):
        raise ContractError("trace image/text/label arrays are not fully row-aligned")
    expected_label_dim = RAW_REBUILT_LABEL_DIMS[dataset]
    if labels.shape[1] != expected_label_dim or labels.dtype.name != "uint8":
        raise ContractError(f"{dataset} trace label dimension/dtype is not frozen raw-rebuilt data")
    if not np.all(np.isin(labels, (0, 1))) or np.any(labels.sum(axis=1) == 0):
        raise ContractError("trace labels must be non-empty binary multi-hot rows")
    if not np.all(np.isfinite(image)) or not np.all(np.isfinite(text)):
        raise ContractError("trace features contain NaN or infinity")

    records = [_require_mapping(record, field=f"trace record[{index}]") for index, record in enumerate(records)]
    _validate_trace_canonical_order(dataset, records)
    if dataset == "nuswide":
        _validate_trace_nus_identity_sources(
            records,
            inventory_paths=inventory_paths,
            resolver_cache=resolver_cache,
        )
    source_ids = [str(record["source_id"]) for record in records]
    expected_split = raw_rebuilt_split_indices(dataset, source_ids)
    split_path = root / "canonical_split.npz"
    with np.load(split_path, allow_pickle=False) as loaded:
        if set(loaded.files) != set(SPLIT_NAMES) | {"row_ids"}:
            raise ContractError("trace canonical split NPZ must contain exactly indT/indQ/indD/row_ids")
        observed_split = {name: np.asarray(loaded[name]) for name in SPLIT_NAMES}
        split_row_ids = np.asarray(loaded["row_ids"]).reshape(-1)
    expected_row_ids = np.asarray(
        [trace_canonical_row_id(dataset, source_id) for source_id in source_ids],
        dtype="S64",
    )
    if (
        split_row_ids.dtype != np.dtype("S64")
        or split_row_ids.shape != (row_count,)
        or not np.array_equal(split_row_ids, expected_row_ids)
    ):
        raise ContractError("trace canonical split row_ids differ from immutable raw identities")
    for name in SPLIT_NAMES:
        if observed_split[name].dtype != np.dtype("int64") or not np.array_equal(
            observed_split[name], expected_split[name]
        ):
            raise ContractError(f"trace {name} differs from the frozen content-hash assignment")
    summary = _require_mapping(verification["split"].get("summary"), field="trace split summary")
    if summary.get("algorithm") != "kbs-content-hash-split-v1" or summary.get("seed") != RAW_REBUILT_SPLIT_SEED:
        raise ContractError("trace split algorithm/seed is not frozen")
    expected_identity_sha = sha256_bytes(canonical_json_bytes(source_ids))
    expected_selection_sha = sha256_bytes(
        canonical_json_bytes({name: expected_split[name].tolist() for name in ("indQ", "indT", "indD")})
    )
    if summary.get("identity_order_sha256") != expected_identity_sha:
        raise ContractError("trace canonical identity-order SHA-256 mismatch")
    expected_row_ids_sha = sha256_bytes(
        canonical_json_bytes([value.decode("ascii") for value in expected_row_ids])
    )
    if summary.get("ordered_row_ids_sha256") != expected_row_ids_sha:
        raise ContractError("trace ordered canonical row-ID SHA-256 mismatch")
    if summary.get("selection_sha256") != expected_selection_sha:
        raise ContractError("trace split assignment SHA-256 mismatch")
    expected_counts = RAW_REBUILT_COUNTS[dataset]
    for name in SPLIT_NAMES:
        if summary.get(f"{name}_rows") != expected_counts[name]:
            raise ContractError(f"trace {name} count differs from frozen registry")

    split_sets = {name: set(expected_split[name].tolist()) for name in SPLIT_NAMES}
    row_contracts: set[str] = set()
    for index, record in enumerate(records):
        if record.get("row_index") != index or record.get("dataset") != dataset:
            raise ContractError("trace record rows are missing, duplicated, or reordered")
        canonical_row = expected_row_ids[index].decode("ascii")
        if record.get("canonical_row_id") != canonical_row:
            raise ContractError(f"trace row {index} canonical_row_id differs from raw identity")
        row_contract = require_sha256(record.get("row_contract_sha256"), field="trace row contract")
        if row_contract in row_contracts:
            raise ContractError("trace immutable row contracts are duplicated")
        row_contracts.add(row_contract)
        image_source = _require_mapping(record.get("raw_image_source"), field=f"trace row {index} raw_image_source")
        _require_exact_keys(
            image_source,
            {"path", "kind", "bytes", "sha256"},
            field=f"trace row {index} raw_image_source",
        )
        if image_source.get("kind") != "raw_image":
            raise ContractError(f"trace row {index} raw image source kind is invalid")
        image_path = _trace_resolve_hashed_source(
            image_source,
            field_name=f"trace row {index} raw_image_source",
            oral_root=oral_root,
            process_root=process,
            file_cache=file_cache,
        )
        declared_image_path = Path(
            _require_nonempty_string(record.get("image_path"), field="trace image path")
        ).resolve(strict=True)
        if declared_image_path != image_path or record.get("image_sha256") != image_source["sha256"]:
            raise ContractError(f"trace row {index} raw image fields are not one immutable source")
        image_descriptor = (int(image_source["bytes"]), str(image_source["sha256"]))
        if image_path in dependency_descriptors and dependency_descriptors[image_path] != image_descriptor:
            raise ContractError(f"inconsistent trace image dependency descriptor: {image_path}")
        dependency_descriptors[image_path] = image_descriptor
        if dataset == "mirflickr" and image_path.name != f"im{record['source_id']}.jpg":
            raise ContractError("MIR raw image filename differs from canonical source_id")
        if dataset == "mscoco":
            metadata = _require_mapping(record.get("metadata"), field=f"trace row {index} metadata")
            if str(metadata.get("coco_image_id")) != str(record.get("source_id")):
                raise ContractError("COCO metadata image ID differs from canonical source_id")
        encoded_texts = record.get("encoded_texts")
        if not isinstance(encoded_texts, list) or not encoded_texts or not all(
            isinstance(value, str) and value for value in encoded_texts
        ):
            raise ContractError(f"trace row {index} has no exact encoded text inputs")
        raw_text = _require_mapping(record.get("raw_text"), field=f"trace row {index} raw_text")
        if sha256_bytes(canonical_json_bytes(raw_text)) != record.get("raw_text_sha256"):
            raise ContractError(f"trace row {index} raw text digest mismatch")
        if sha256_bytes(canonical_json_bytes(encoded_texts)) != record.get("encoded_text_sha256"):
            raise ContractError(f"trace row {index} encoded text digest mismatch")
        _validate_trace_text_sources(
            record,
            row_index=index,
            dataset=dataset,
            oral_root=oral_root,
            process_root=process,
            file_cache=file_cache,
            resolver_cache=resolver_cache,
            dependency_descriptors=dependency_descriptors,
        )
        expected_flags = {
            name: index in split_sets[name] for name in SPLIT_NAMES
        }
        if record.get("split") != expected_flags:
            raise ContractError(f"trace row {index} split flags differ from frozen assignment")
        if list(labels[index].astype(int)) != record.get("label_hot"):
            raise ContractError(f"trace row {index} label vector is not bound to its shard")
        if sha256_bytes(np.asarray(labels[index], dtype=np.uint8).tobytes()) != record.get("label_sha256"):
            raise ContractError(f"trace row {index} label SHA-256 mismatch")
        _validate_trace_label_source(
            record,
            row_index=index,
            dataset=dataset,
            adapter=adapter,
            observed_label=labels[index],
            oral_root=oral_root,
            resolver_cache=resolver_cache,
            dependency_descriptors=dependency_descriptors,
        )

    _validate_trace_dependency_summary(
        contract.get("raw_dependency_inventory"), dependency_descriptors
    )

    expected_files = {
        "contract.json",
        "canonical_split.npz",
        "canonical_split.json",
        "complete.json",
    }
    receipt_paths = sorted((root / "receipts").glob("part-*.json"))
    if len(receipt_paths) != int(complete.get("shards", -1)):
        raise ContractError("trace receipt count differs from complete marker")
    shard_row_id_parts: list[np.ndarray] = []
    for receipt_path in receipt_paths:
        receipt = _require_mapping(_decode_json(receipt_path.read_bytes(), field=str(receipt_path)), field="receipt")
        expected_files.add(receipt_path.relative_to(root).as_posix())
        npz_path = resolve_contained(root, receipt.get("npz_path"), field="trace receipt npz_path")
        manifest_path = resolve_contained(root, receipt.get("manifest_path"), field="trace receipt manifest_path")
        npz_relative = npz_path.relative_to(root).as_posix()
        manifest_relative = manifest_path.relative_to(root).as_posix()
        if receipt.get("npz_path") != npz_relative or receipt.get("manifest_path") != manifest_relative:
            raise ContractError("trace receipt paths must be canonical bundle-relative POSIX paths")
        expected_files.add(npz_relative)
        expected_files.add(manifest_relative)
        with np.load(npz_path, allow_pickle=False) as loaded:
            if set(loaded.files) != {
                "row_index",
                "row_ids",
                "image_features",
                "text_features",
                "labels",
            }:
                raise ContractError(f"trace shard arrays are missing or unbound: {npz_path}")
            shard_ids = np.asarray(loaded["row_ids"]).reshape(-1)
            if shard_ids.dtype != np.dtype("S64"):
                raise ContractError(f"trace shard row_ids must be fixed S64: {npz_path}")
            shard_row_id_parts.append(shard_ids.copy())
    concatenated_shard_row_ids = (
        np.concatenate(shard_row_id_parts)
        if shard_row_id_parts
        else np.empty((0,), dtype="S64")
    )
    if not np.array_equal(concatenated_shard_row_ids, split_row_ids):
        raise ContractError("trace split row_ids differ byte-for-byte from concatenated shard row_ids")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if actual_files != expected_files:
        raise ContractError("trace bundle contains missing or unbound files")

    return ValidationReport(
        bundle_id="sha256:" + str(contract["contract_sha256"]),
        dataset=dataset,
        row_count=row_count,
        shard_count=int(complete["shards"]),
        source_file_count=len(dependency_descriptors),
        checks=(
            "visualization_trace_contract_receipts_complete_chain",
            "OralData_only_no_ProcessData_read_or_authority",
            "raw_rebuilt_v1_canonical_identity_order",
            "raw_image_text_label_and_feature_row_binding",
            "seed_20260822_kbs_content_hash_split_and_fixed_counts",
            "complete_trace_inventory_no_unbound_files",
        ),
    )


def validate_bundle(
    bundle_root: Path,
    process_data_root: Path | None = None,
) -> ValidationReport:
    """Validate every source, row, split, and vector before visualization use.

    Labels, identity, and text are resolved only from raw sources.  ProcessData
    is never a row/label/index authority.  This function contains no metric,
    training, selection, ranking, or model-choice operation.
    """

    try:
        bundle = bundle_root.resolve(strict=True)
    except OSError as error:
        raise ContractError(f"bundle root does not exist: {bundle_root}") from error
    if not bundle.is_dir():
        raise ContractError("bundle root must be a directory")
    if (bundle / "contract.json").is_file() and (bundle / "complete.json").is_file():
        return validate_visualization_trace_bundle(bundle, process_data_root)
    manifest_path = bundle / MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ContractError(f"bundle must contain a regular {MANIFEST_NAME}")
    manifest = _load_canonical_manifest(manifest_path)
    _require_exact_keys(manifest, TOP_LEVEL_KEYS, field="manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ContractError("unsupported manifest schema version")
    if manifest["bundle_id"] != derive_bundle_id(manifest):
        raise ContractError("bundle_id does not bind the complete manifest")
    require_content_id(manifest["bundle_id"], field="bundle_id")
    dataset = _require_nonempty_string(manifest["dataset"], field="dataset")
    _require_nonempty_string(manifest["created_utc"], field="created_utc")
    row_count = _require_nonnegative_int(manifest["row_count"], field="row_count")
    if row_count == 0:
        raise ContractError("row_count must be positive")
    process_root, raw_root = _validate_boundaries(bundle, manifest, process_data_root)
    _reject_processdata_authority_paths(manifest, process_root)
    inventory = _validate_inventory(bundle, manifest)
    extractors = _validate_extractors(manifest)
    label_class_count = _validate_labels_contract(
        _require_mapping(manifest["labels"], field="labels"), row_count, dataset
    )
    source_cache: dict[Path, tuple[int, int, str]] = {}
    locator_cache: dict[tuple[Path, str], Any] = {}

    rows_descriptor = _require_mapping(manifest["rows"], field="rows")
    _require_exact_keys(rows_descriptor, {"path", "size", "sha256", "count", "format"}, field="rows")
    if rows_descriptor["format"] != "canonical_jsonl_v1" or rows_descriptor["count"] != row_count:
        raise ContractError("rows descriptor count/format mismatch")
    rows_relative = _require_inventory_descriptor(rows_descriptor, inventory, field="rows")
    rows = _load_canonical_jsonl(resolve_contained(bundle, rows_relative.as_posix(), field="rows.path"))
    if len(rows) != row_count:
        raise ContractError("rows JSONL count mismatch")

    external_paths: set[Path] = set()
    sample_ids: list[str] = []
    sample_id_seen: set[str] = set()
    row_ids: set[str] = set()
    raw_label_rows: list[np.ndarray] = []
    for index, row in enumerate(rows):
        row_paths, raw_label = _validate_row_structure(
            row,
            index=index,
            dataset=dataset,
            raw_root=raw_root,
            extractors=extractors,
            source_cache=source_cache,
            locator_cache=locator_cache,
            label_class_count=label_class_count,
        )
        external_paths.update(row_paths)
        raw_label_rows.append(raw_label)
        sample_id = str(row["sample_id"])
        row_id = str(row["row_id"])
        if sample_id in sample_id_seen:
            raise ContractError(f"duplicate dataset/sample_id {dataset}/{sample_id}")
        if row_id in row_ids:
            raise ContractError(f"duplicate immutable row_id {row_id}")
        sample_ids.append(sample_id)
        sample_id_seen.add(sample_id)
        row_ids.add(row_id)

    _validate_authority(
        _require_mapping(manifest["authority"], field="authority"),
        dataset=dataset,
        row_count=row_count,
        sample_ids=sample_ids,
    )
    raw_labels = np.stack(raw_label_rows, axis=0)
    _validate_splits(
        bundle,
        _require_mapping(manifest["deterministic_splits"], field="deterministic_splits"),
        inventory=inventory,
        rows=rows,
        dataset=dataset,
    )
    shard_descriptors = _require_list(manifest["shards"], field="shards")
    _validate_shards(
        bundle,
        shard_descriptors,
        inventory=inventory,
        rows=rows,
        raw_labels=raw_labels,
    )
    referenced_bundle_paths = {
        str(rows_descriptor["path"]),
        str(manifest["deterministic_splits"]["artifact"]["path"]),
    }
    referenced_bundle_paths.update(str(item["path"]) for item in shard_descriptors)
    if referenced_bundle_paths != set(inventory):
        raise ContractError("inventory contains unreferenced or missing content")
    _reverify_source_cache(source_cache)

    return ValidationReport(
        bundle_id=str(manifest["bundle_id"]),
        dataset=dataset,
        row_count=row_count,
        shard_count=len(shard_descriptors),
        source_file_count=len(external_paths),
        checks=(
            "output_outside_ProcessData_and_raw_data",
            "raw_rebuilt_v1_identity_text_and_labels_only",
            "ProcessData_never_row_label_index_or_feature_authority",
            "canonical_NFC_UTF8_sample_id_order",
            "canonical_manifest_rows_and_complete_inventory",
            "image_text_multilabel_same_n_rows_and_row_id_order",
            "raw_image_and_multitext_source_SHA256",
            "explicit_empty_text_fallback_and_aggregation",
            "raw_multilabel_locator_exact_row_binding_provenance_only",
            "seed_20260822_SHA256_split_counts_and_assignment",
            "NPZ_or_Parquet_file_and_content_binding",
            "per_row_vector_SHA256_and_no_missing_duplicate_reorder",
        ),
    )
