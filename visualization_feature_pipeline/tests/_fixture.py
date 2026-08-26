from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from visualization_feature_pipeline.contract import (
    SCHEMA_VERSION,
    canonical_json_bytes,
    derive_bundle_id,
    derive_extractor_id,
    derive_row_id,
    feature_row_sha256,
    int64_array_sha256,
    numeric_array_sha256,
    ordered_ids_sha256,
    raw_rebuilt_assignment_sha256,
    raw_rebuilt_split_indices,
    sha256_bytes,
    sha256_file,
    text_utf8_sha256,
)


@dataclass(frozen=True)
class SyntheticBundle:
    root: Path
    process_root: Path
    raw_root: Path


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def bundle_file_record(bundle: Path, relative: str) -> dict[str, Any]:
    path = bundle / relative
    return {"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)}


def write_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def read_manifest(bundle: Path) -> dict[str, Any]:
    return json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))


def write_manifest(bundle: Path, manifest: dict[str, Any]) -> None:
    manifest["bundle_id"] = derive_bundle_id(manifest)
    write_canonical_json(bundle / "bundle_manifest.json", manifest)


def read_rows(bundle: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (bundle / "rows.jsonl").read_text(encoding="utf-8").splitlines()]


def write_rows_and_reseal(bundle: Path, rows: list[dict[str, Any]]) -> None:
    rows_path = bundle / "rows.jsonl"
    rows_path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    manifest = read_manifest(bundle)
    record = bundle_file_record(bundle, "rows.jsonl")
    manifest["rows"].update(record)
    for item in manifest["inventory"]:
        if item["path"] == "rows.jsonl":
            item.update(record)
    write_manifest(bundle, manifest)


def update_bound_file_and_reseal(bundle: Path, relative: str) -> None:
    manifest = read_manifest(bundle)
    record = bundle_file_record(bundle, relative)
    for item in manifest["inventory"]:
        if item["path"] == relative:
            item.update(record)
    if manifest["rows"]["path"] == relative:
        manifest["rows"].update(record)
    if manifest["deterministic_splits"]["artifact"]["path"] == relative:
        manifest["deterministic_splits"]["artifact"].update(record)
    for shard in manifest["shards"]:
        if shard["path"] == relative:
            shard.update(record)
    write_manifest(bundle, manifest)


def _extractor(semantic: str, shape: list[int]) -> dict[str, Any]:
    preprocess = (
        {"resize": 224, "crop": "center", "normalize": "clip_reference"}
        if semantic == "image"
        else {"tokenizer": "clip_bpe", "context_length": 77, "truncate": True}
    )
    value: dict[str, Any] = {
        "extractor_id": "",
        "semantic": semantic,
        "model_name": "synthetic-clip",
        "model_revision": "fixture-revision-1",
        "model_artifact_sha256": sha256_bytes(b"synthetic-checkpoint"),
        "library": "synthetic-runtime",
        "library_version": "1.0",
        "code_version": "fixture-code-v1",
        "preprocess": preprocess,
        "preprocess_sha256": sha256_bytes(canonical_json_bytes(preprocess)),
        "output_dtype": "float32",
        "output_shape": shape,
    }
    value["extractor_id"] = derive_extractor_id(value)
    return value


def _source_item(source: Mapping[str, Any], locator: str, value: str) -> dict[str, Any]:
    return {
        "source": dict(source),
        "locator": {"kind": "json_pointer", "value": locator},
        "value": value,
        "utf8_sha256": text_utf8_sha256(value),
    }


def _text_record(source: Mapping[str, Any], row: int, values: list[str]) -> dict[str, Any]:
    items = [_source_item(source, f"/samples/{row}/captions/{index}", value) for index, value in enumerate(values)]
    raw_empty = all(value == "" for value in values)
    if raw_empty:
        fallback_value = "a generic photo"
        fallback: dict[str, Any] | None = {
            "policy_id": "explicit_empty_text_fallback_v1",
            "value": fallback_value,
            "utf8_sha256": text_utf8_sha256(fallback_value),
        }
        model_values = [("fallback", None, fallback_value)]
    else:
        fallback = None
        model_values = [("raw", index, value) for index, value in enumerate(values) if value]
    model_inputs = [
        {
            "origin": origin,
            "source_item_index": source_index,
            "value": value,
            "utf8_sha256": text_utf8_sha256(value),
        }
        for origin, source_index, value in model_values
    ]
    return {
        "normalization": "unicode_nfc_lf_v1",
        "raw_empty": raw_empty,
        "source_items": items,
        "fallback": fallback,
        "model_inputs": model_inputs,
        "aggregation": {
            "method": "mean_l2_normalized" if len(model_inputs) > 1 else "single",
            "input_count": len(model_inputs),
            "order_sensitive": False,
            "version": "synthetic-aggregation-v1",
        },
    }


def build_valid_bundle(parent: Path, *, bundle_inside_process: bool = False) -> SyntheticBundle:
    process_root = parent / "ProcessData"
    raw_root = parent / "OralData"
    process_root.mkdir(parents=True)
    raw_root.mkdir(parents=True)
    bundle = (process_root / "KBS_Visualization_Features_BAD") if bundle_inside_process else (parent / "KBS_Visualization_Features_SYNTHETIC")
    bundle.mkdir(parents=True)
    (bundle / "shards").mkdir()

    n_rows = 4
    labels = np.asarray([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.uint8)
    sample_ids = [f"sample-{index}" for index in range(n_rows)]
    official = raw_rebuilt_split_indices("synthetic", sample_ids)

    identity_payload = {"samples": [{"sample_id": f"sample-{index}"} for index in range(n_rows)]}
    write_canonical_json(raw_root / "identity_map.json", identity_payload)
    captions = [
        ["red cat"],
        ["blue sky", "small cloud"],
        [""],
        ["green tree"],
    ]
    write_canonical_json(raw_root / "captions.json", {"samples": [{"captions": row} for row in captions]})
    write_canonical_json(raw_root / "labels.json", {"samples": [{"labels": row.tolist()} for row in labels]})
    for index in range(n_rows):
        (raw_root / f"image_{index}.bin").write_bytes(b"synthetic-image-" + bytes([index]) * 13)

    image_values = np.asarray(
        [[1.0, 0.1, 0.2], [0.2, 1.0, 0.3], [0.4, 0.5, 1.0], [0.9, 0.8, 0.7]],
        dtype=np.float32,
    )
    text_values = np.asarray(
        [[0.8, 0.1, 0.2], [0.1, 0.9, 0.3], [0.3, 0.2, 0.7], [0.7, 0.6, 0.5]],
        dtype=np.float32,
    )
    image_extractor = _extractor("image", [3])
    text_extractor = _extractor("text", [3])
    identity_source = file_record(raw_root / "identity_map.json")
    text_source = file_record(raw_root / "captions.json")
    label_source = file_record(raw_root / "labels.json")

    split_positions: dict[str, dict[int, int]] = {
        name: {int(global_row): position for position, global_row in enumerate(values.tolist())}
        for name, values in official.items()
    }
    rows: list[dict[str, Any]] = []
    for index in range(n_rows):
        row: dict[str, Any] = {
            "row_id": "",
            "global_row": index,
            "dataset": "synthetic",
            "sample_id": f"sample-{index}",
            "identity": {
                "method": "official_id_map",
                "source": identity_source,
                "locator": {"kind": "json_pointer", "value": f"/samples/{index}/sample_id"},
            },
            "raw_image": file_record(raw_root / f"image_{index}.bin"),
            "text": _text_record(text_source, index, captions[index]),
            "label": {
                "source": label_source,
                "locator": {"kind": "json_pointer", "value": f"/samples/{index}/labels"},
                "encoding": "binary_multihot_uint8_v1",
                "value": labels[index].tolist(),
                "vector_sha256": feature_row_sha256(labels[index]),
            },
            "vectors": {
                "image": {
                    "shard_id": "image-00000",
                    "shard_row": index,
                    "dtype": "float32",
                    "shape": [3],
                    "sha256": feature_row_sha256(image_values[index]),
                    "extractor_id": image_extractor["extractor_id"],
                },
                "text": {
                    "shard_id": "text-00000",
                    "shard_row": index,
                    "dtype": "float32",
                    "shape": [3],
                    "sha256": feature_row_sha256(text_values[index]),
                    "extractor_id": text_extractor["extractor_id"],
                },
                "multilabel": {
                    "shard_id": "multilabel-00000",
                    "shard_row": index,
                    "dtype": "uint8",
                    "shape": [3],
                    "sha256": feature_row_sha256(labels[index]),
                    "extractor_id": "raw_multilabel_locator",
                },
            },
            "deterministic_split_positions": {
                name: split_positions[name].get(index) for name in ("indT", "indQ", "indD")
            },
        }
        row["row_id"] = derive_row_id(row)
        rows.append(row)
    row_ids = np.asarray([row["row_id"] for row in rows], dtype=str)

    np.savez(bundle / "shards" / "image.npz", row_ids=row_ids, vectors=image_values)
    np.savez(bundle / "shards" / "text.npz", row_ids=row_ids, vectors=text_values)
    np.savez(bundle / "shards" / "multilabel.npz", row_ids=row_ids, vectors=labels)
    split_payload: dict[str, np.ndarray] = {}
    for name, values in official.items():
        split_payload[name] = values
        split_payload[f"{name}_row_ids"] = row_ids[values]
    np.savez(bundle / "deterministic_splits.npz", **split_payload)
    (bundle / "rows.jsonl").write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    )

    shard_specs = [
        ("image-00000", "image", "shards/image.npz", image_values),
        ("multilabel-00000", "multilabel", "shards/multilabel.npz", labels),
        ("text-00000", "text", "shards/text.npz", text_values),
    ]
    shards: list[dict[str, Any]] = []
    for shard_id, semantic, relative, vectors in shard_specs:
        record = bundle_file_record(bundle, relative)
        shards.append(
            {
                "shard_id": shard_id,
                "semantic": semantic,
                **record,
                "format": "npz",
                "row_start": 0,
                "row_count": n_rows,
                "row_ids_key": "row_ids",
                "vector_key": "vectors",
                "row_ids_sha256": ordered_ids_sha256(row_ids),
                "vectors_sha256": numeric_array_sha256(vectors),
                "dtype": vectors.dtype.name,
                "shape": list(vectors.shape[1:]),
            }
        )

    rows_record = bundle_file_record(bundle, "rows.jsonl")
    split_record = bundle_file_record(bundle, "deterministic_splits.npz")
    inventory_paths = ["deterministic_splits.npz", "rows.jsonl"] + [item[2] for item in shard_specs]
    inventory = [bundle_file_record(bundle, relative) for relative in sorted(inventory_paths)]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "bundle_id": "",
        "dataset": "synthetic",
        "created_utc": "2026-08-22T00:00:00Z",
        "row_count": n_rows,
        "boundaries": {
            "raw_data_root": str(raw_root.resolve(strict=True)),
            "output_kind": "independent_visualization_features",
            "process_data_policy": "forbidden_as_input_or_authority",
            "raw_data_access": "read_only",
        },
        "authority": {
            "kind": "raw_rebuilt_v1",
            "canonical_order_algorithm": "nfc_utf8_sample_id_ascending_v1",
            "ordered_sample_ids_sha256": ordered_ids_sha256(sample_ids),
            "process_data_count_comparison": None,
        },
        "extractors": [image_extractor, text_extractor],
        "rows": {**rows_record, "count": n_rows, "format": "canonical_jsonl_v1"},
        "deterministic_splits": {
            "algorithm": "kbs-content-hash-split-v1",
            "seed": 20260822,
            "counts": {"n_rows": 4, "indQ": 1, "indT": 2, "indD": 3},
            "assignment_sha256": raw_rebuilt_assignment_sha256("synthetic", sample_ids, official),
            "artifact": {**split_record, "format": "npz"},
            "arrays": {
                name: {
                    "count": int(values.size),
                    "indices_sha256": int64_array_sha256(values),
                    "row_ids_sha256": ordered_ids_sha256(row_ids[values]),
                }
                for name, values in official.items()
            },
            "relations": {
                "indT_subset_of_indD": True,
                "indQ_disjoint_indD": True,
                "indQ_disjoint_indT": True,
                "indQ_union_indD_full": True,
            },
        },
        "labels": {
            "authority": "raw_locator_per_sample_v1",
            "row_count": n_rows,
            "class_count": labels.shape[1],
            "row_order": "canonical_raw_sample_order",
            "selection_use": "provenance_only_never_model_selection",
        },
        "shards": shards,
        "inventory": inventory,
    }
    write_manifest(bundle, manifest)
    return SyntheticBundle(bundle, process_root, raw_root)


def clone_manifest(bundle: Path) -> dict[str, Any]:
    return copy.deepcopy(read_manifest(bundle))
