from __future__ import annotations

import io
import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from visualization_feature_pipeline.cli import main as cli_main
from visualization_feature_pipeline.contract import (
    ContractError,
    canonical_json_bytes,
    derive_row_id,
    int64_array_sha256,
    numeric_array_sha256,
    ordered_ids_sha256,
    text_utf8_sha256,
)
from visualization_feature_pipeline.validator import validate_bundle

from ._fixture import (
    build_valid_bundle,
    bundle_file_record,
    file_record,
    read_manifest,
    read_rows,
    update_bound_file_and_reseal,
    write_manifest,
    write_rows_and_reseal,
)


class _TraceSyntheticEncoder:
    model_id = "synthetic-trace-encoder-v1"

    @staticmethod
    def _vector(value: str) -> np.ndarray:
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        return np.frombuffer(digest[:16], dtype=np.uint32).astype(np.float32)

    def encode_images(self, image_paths: Sequence[str]) -> np.ndarray:
        return np.stack([self._vector(Path(value).name) for value in image_paths])

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vector(value) for value in texts])

    def contract(self) -> Mapping[str, Any]:
        return {"model_id": self.model_id, "output_dtype": "float32", "dim": 4}


class _TraceSyntheticAdapter:
    dataset = "synthetic"

    def __init__(self, data_root: Path) -> None:
        from visualization_trace.core import ContentHashSplit

        self.data_root = data_root.resolve()
        self.oral_root = self.data_root / "OralData" / "SYNTHETIC"
        self.oral_root.mkdir(parents=True)
        self.rows = 4
        self.source_ids = tuple(f"sample-{index}" for index in range(self.rows))
        self.labels = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0))
        self.source_path = self.oral_root / "source.json"
        self.source_path.write_text(
            json.dumps(
                {
                    "annotations": [
                        {
                            "id": 1000 + index,
                            "image_id": index,
                            "caption": f"caption {index}",
                            "label": list(self.labels[index]),
                        }
                        for index in range(self.rows)
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.image_paths = []
        for index in range(self.rows):
            path = self.oral_root / f"image-{index}.bin"
            path.write_bytes(f"raw-image-{index}".encode("utf-8"))
            self.image_paths.append(path)
        self.source_artifacts = (self.source_path,)
        self.split = ContentHashSplit.build("synthetic", self.source_ids, 1, 2, seed=20260822)

    def dependency_paths(self) -> tuple[Path, ...]:
        paths = tuple(self.source_artifacts) + tuple(self.image_paths)
        return tuple(sorted(paths, key=lambda path: str(path.resolve())))

    def contract(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "rows": self.rows,
            "data_root": str(self.data_root),
            "source_artifacts": [str(path.resolve()) for path in self.source_artifacts],
            "split": self.split.summary(),
            "canonical_order": "ascending zero-based raw source_index",
            "identity_chain": "OralData only",
            "process_data_role": "none; forbidden as input or authority",
        }

    def iter_rows(self, start: int = 0) -> Iterator[Any]:
        from visualization_trace.core import TraceRow, sha256_json

        for index in range(start, self.rows):
            caption = f"caption {index}"
            yield TraceRow(
                dataset=self.dataset,
                row_index=index,
                source_index=index,
                source_id=self.source_ids[index],
                image_path=str(self.image_paths[index]),
                encoded_texts=(caption,),
                raw_text={"captions": [caption]},
                label_hot=self.labels[index],
                split=self.split.flags(index),
                metadata={
                    "canonical_identity_binding": "raw source_index",
                    "label_source": {
                        "path": str(self.source_path.resolve()),
                        "kind": "json_pointer",
                        "json_pointer": f"/annotations/{index}/label",
                    },
                },
                raw_text_locators=(
                    {
                        "path": str(self.source_path.resolve()),
                        "kind": "json_annotation",
                        "locator": {
                            "json_pointer": f"/annotations/{index}",
                            "annotation_id": 1000 + index,
                            "image_id": index,
                        },
                        "content_sha256": sha256_json(
                            {
                                "annotation_id": 1000 + index,
                                "image_id": index,
                                "caption": caption,
                            }
                        ),
                    },
                ),
            )


def _build_visualization_trace_bundle(root: Path) -> tuple[Path, _TraceSyntheticAdapter]:
    from visualization_trace.extraction import ExtractionConfig, extract_trace_bundle

    adapter = _TraceSyntheticAdapter(root / "Data")
    output = root / "trace-output"
    config = ExtractionConfig(
        output_root=output,
        run_name="raw-rebuilt-e2e",
        batch_size=2,
        text_batch_size=2,
        shard_rows=2,
    )
    extract_trace_bundle(adapter, _TraceSyntheticEncoder(), config)
    return output / "synthetic" / "raw-rebuilt-e2e", adapter


class VisualizationFeatureContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_valid_npz_bundle_covers_all_three_vectors_and_splits(self) -> None:
        fixture = build_valid_bundle(self.base)
        report = validate_bundle(fixture.root, fixture.process_root)
        self.assertEqual(report.row_count, 4)
        self.assertEqual(report.shard_count, 3)
        self.assertIn("image_text_multilabel_same_n_rows_and_row_id_order", report.checks)
        self.assertIn("seed_20260822_SHA256_split_counts_and_assignment", report.checks)

    def test_output_inside_process_data_is_rejected(self) -> None:
        fixture = build_valid_bundle(self.base, bundle_inside_process=True)
        with self.assertRaisesRegex(ContractError, "outside ProcessData"):
            validate_bundle(fixture.root, fixture.process_root)

    def test_duplicate_dataset_sample_id_is_rejected_before_shards(self) -> None:
        fixture = build_valid_bundle(self.base)
        rows = read_rows(fixture.root)
        rows[1]["sample_id"] = rows[0]["sample_id"]
        rows[1]["identity"]["locator"] = dict(rows[0]["identity"]["locator"])
        rows[1]["row_id"] = derive_row_id(rows[1])
        write_rows_and_reseal(fixture.root, rows)
        with self.assertRaisesRegex(ContractError, "duplicate dataset/sample_id"):
            validate_bundle(fixture.root, fixture.process_root)

    def test_reordered_row_ledger_is_rejected(self) -> None:
        fixture = build_valid_bundle(self.base)
        rows = read_rows(fixture.root)
        rows[0], rows[1] = rows[1], rows[0]
        write_rows_and_reseal(fixture.root, rows)
        with self.assertRaisesRegex(ContractError, "global_row is missing or reordered"):
            validate_bundle(fixture.root, fixture.process_root)

    def test_missing_modality_is_rejected(self) -> None:
        fixture = build_valid_bundle(self.base)
        rows = read_rows(fixture.root)
        del rows[0]["vectors"]["text"]
        write_rows_and_reseal(fixture.root, rows)
        with self.assertRaisesRegex(ContractError, "must bind image, text, and multilabel"):
            validate_bundle(fixture.root, fixture.process_root)

    def test_mir_label_signature_or_similarity_identity_is_rejected(self) -> None:
        fixture = build_valid_bundle(self.base)
        rows = read_rows(fixture.root)
        rows[0]["identity"]["method"] = "label_signature_plus_cosine_guess"
        write_rows_and_reseal(fixture.root, rows)
        with self.assertRaisesRegex(ContractError, "label-signature/similarity guessing is forbidden"):
            validate_bundle(fixture.root, fixture.process_root)

    def test_explicit_empty_text_requires_a_bound_fallback(self) -> None:
        fixture = build_valid_bundle(self.base)
        rows = read_rows(fixture.root)
        rows[2]["text"]["fallback"] = None
        write_rows_and_reseal(fixture.root, rows)
        with self.assertRaisesRegex(ContractError, "fallback"):
            validate_bundle(fixture.root, fixture.process_root)

    def test_multitext_aggregation_count_is_bound(self) -> None:
        fixture = build_valid_bundle(self.base)
        rows = read_rows(fixture.root)
        rows[1]["text"]["aggregation"]["input_count"] = 1
        write_rows_and_reseal(fixture.root, rows)
        with self.assertRaisesRegex(ContractError, "aggregation.input_count mismatch"):
            validate_bundle(fixture.root, fixture.process_root)

    def test_text_locator_is_executed_against_the_raw_source(self) -> None:
        fixture = build_valid_bundle(self.base)
        rows = read_rows(fixture.root)
        wrong = "not the raw caption"
        rows[0]["text"]["source_items"][0]["value"] = wrong
        rows[0]["text"]["source_items"][0]["utf8_sha256"] = text_utf8_sha256(wrong)
        rows[0]["text"]["model_inputs"][0]["value"] = wrong
        rows[0]["text"]["model_inputs"][0]["utf8_sha256"] = text_utf8_sha256(wrong)
        rows[0]["row_id"] = derive_row_id(rows[0])
        write_rows_and_reseal(fixture.root, rows)
        with self.assertRaisesRegex(ContractError, "locator does not resolve to recorded text"):
            validate_bundle(fixture.root, fixture.process_root)

    def test_unknown_dataset_specific_locator_fails_closed(self) -> None:
        fixture = build_valid_bundle(self.base)
        rows = read_rows(fixture.root)
        rows[0]["text"]["source_items"][0]["locator"] = {
            "kind": "unaudited_matlab_cell_guess",
            "value": "0",
        }
        write_rows_and_reseal(fixture.root, rows)
        with self.assertRaisesRegex(ContractError, "has no audited resolver"):
            validate_bundle(fixture.root, fixture.process_root)

    def test_same_size_raw_image_poison_is_rejected(self) -> None:
        fixture = build_valid_bundle(self.base)
        image = fixture.raw_root / "image_0.bin"
        before = image.stat().st_size
        payload = bytearray(image.read_bytes())
        payload[-1] ^= 0x01
        image.write_bytes(payload)
        self.assertEqual(image.stat().st_size, before)
        with self.assertRaisesRegex(ContractError, "SHA-256 mismatch"):
            validate_bundle(fixture.root, fixture.process_root)

    def test_same_size_feature_shard_poison_is_rejected(self) -> None:
        fixture = build_valid_bundle(self.base)
        shard = fixture.root / "shards" / "image.npz"
        before = shard.stat().st_size
        payload = bytearray(shard.read_bytes())
        payload[len(payload) // 2] ^= 0x01
        shard.write_bytes(payload)
        self.assertEqual(shard.stat().st_size, before)
        with self.assertRaisesRegex(ContractError, "inventory SHA-256 mismatch"):
            validate_bundle(fixture.root, fixture.process_root)

    def test_deterministic_split_reorder_is_rejected_even_after_bundle_reseal(self) -> None:
        fixture = build_valid_bundle(self.base)
        path = fixture.root / "deterministic_splits.npz"
        with np.load(path, allow_pickle=False) as loaded:
            payload = {name: np.asarray(loaded[name]) for name in loaded.files}
        payload["indD"] = payload["indD"][::-1].copy()
        payload["indD_row_ids"] = payload["indD_row_ids"][::-1].copy()
        np.savez(path, **payload)
        manifest = read_manifest(fixture.root)
        record = bundle_file_record(fixture.root, "deterministic_splits.npz")
        manifest["deterministic_splits"]["artifact"].update(record)
        manifest["deterministic_splits"]["arrays"]["indD"].update(
            {
                "indices_sha256": int64_array_sha256(payload["indD"]),
                "row_ids_sha256": ordered_ids_sha256(payload["indD_row_ids"]),
            }
        )
        for item in manifest["inventory"]:
            if item["path"] == "deterministic_splits.npz":
                item.update(record)
        write_manifest(fixture.root, manifest)
        with self.assertRaisesRegex(ContractError, "differs from the frozen SHA-256 assignment"):
            validate_bundle(fixture.root, fixture.process_root)

    def test_multilabel_row_reorder_is_rejected_even_after_shard_reseal(self) -> None:
        fixture = build_valid_bundle(self.base)
        path = fixture.root / "shards" / "multilabel.npz"
        with np.load(path, allow_pickle=False) as loaded:
            row_ids = np.asarray(loaded["row_ids"])
            vectors = np.asarray(loaded["vectors"])
        vectors[[0, 1]] = vectors[[1, 0]]
        np.savez(path, row_ids=row_ids, vectors=vectors)
        manifest = read_manifest(fixture.root)
        record = bundle_file_record(fixture.root, "shards/multilabel.npz")
        for shard in manifest["shards"]:
            if shard["semantic"] == "multilabel":
                shard.update(record)
                shard["vectors_sha256"] = numeric_array_sha256(vectors)
        for item in manifest["inventory"]:
            if item["path"] == "shards/multilabel.npz":
                item.update(record)
        write_manifest(fixture.root, manifest)
        with self.assertRaisesRegex(ContractError, "multilabel vector SHA-256 mismatch"):
            validate_bundle(fixture.root, fixture.process_root)

    def test_raw_label_source_change_cannot_silently_rebind_rows(self) -> None:
        fixture = build_valid_bundle(self.base)
        label_path = fixture.raw_root / "labels.json"
        before = label_path.stat().st_size
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        payload["samples"][0]["labels"] = [0, 1, 0]
        label_path.write_bytes(canonical_json_bytes(payload) + b"\n")
        self.assertEqual(label_path.stat().st_size, before)
        with self.assertRaisesRegex(ContractError, "SHA-256 mismatch"):
            validate_bundle(fixture.root, fixture.process_root)

    def test_split_seed_cannot_be_tried_or_loosened(self) -> None:
        fixture = build_valid_bundle(self.base)
        manifest = read_manifest(fixture.root)
        manifest["deterministic_splits"]["seed"] = 20260823
        write_manifest(fixture.root, manifest)
        with self.assertRaisesRegex(ContractError, "exactly 20260822"):
            validate_bundle(fixture.root, fixture.process_root)

    def test_split_counts_are_frozen_by_dataset_registry(self) -> None:
        fixture = build_valid_bundle(self.base)
        manifest = read_manifest(fixture.root)
        manifest["deterministic_splits"]["counts"]["indT"] = 3
        write_manifest(fixture.root, manifest)
        with self.assertRaisesRegex(ContractError, "frozen dataset registry"):
            validate_bundle(fixture.root, fixture.process_root)

    def test_missing_raw_text_source_is_rejected(self) -> None:
        fixture = build_valid_bundle(self.base)
        (fixture.raw_root / "captions.json").unlink()
        with self.assertRaisesRegex(ContractError, "missing"):
            validate_bundle(fixture.root, fixture.process_root)

    def test_unavailable_or_invalid_parquet_fails_closed(self) -> None:
        fixture = build_valid_bundle(self.base)
        manifest = read_manifest(fixture.root)
        manifest["shards"][0]["format"] = "parquet"
        write_manifest(fixture.root, manifest)
        with self.assertRaises(ContractError):
            validate_bundle(fixture.root, fixture.process_root)

    def test_unreferenced_file_is_rejected(self) -> None:
        fixture = build_valid_bundle(self.base)
        (fixture.root / "unbound.tmp").write_bytes(b"unbound")
        with self.assertRaisesRegex(ContractError, "inventory mismatch"):
            validate_bundle(fixture.root, fixture.process_root)

    def test_visualization_trace_synthetic_bundle_is_directly_compatible(self) -> None:
        run_dir, _ = _build_visualization_trace_bundle(self.base)
        report = validate_bundle(run_dir)
        self.assertEqual(report.dataset, "synthetic")
        self.assertEqual(report.row_count, 4)
        self.assertEqual(report.shard_count, 2)
        self.assertIn("raw_image_text_label_and_feature_row_binding", report.checks)

    def test_visualization_trace_same_size_raw_text_poison_is_rejected(self) -> None:
        run_dir, adapter = _build_visualization_trace_bundle(self.base)
        before = adapter.source_path.stat().st_size
        payload = adapter.source_path.read_bytes()
        self.assertIn(b"caption 0", payload)
        adapter.source_path.write_bytes(payload.replace(b"caption 0", b"Caption 0", 1))
        self.assertEqual(adapter.source_path.stat().st_size, before)
        with self.assertRaisesRegex(ContractError, "raw source SHA-256 mismatch"):
            validate_bundle(run_dir)

    def test_visualization_trace_split_row_id_poison_is_rejected(self) -> None:
        run_dir, _ = _build_visualization_trace_bundle(self.base)
        split_path = run_dir / "canonical_split.npz"
        with np.load(split_path, allow_pickle=False) as loaded:
            values = {name: np.asarray(loaded[name]).copy() for name in loaded.files}
        values["row_ids"][[0, 1]] = values["row_ids"][[1, 0]]
        np.savez_compressed(split_path, **values)
        with self.assertRaisesRegex(ContractError, "sealed-bundle verification failed"):
            validate_bundle(run_dir)

    def test_visualization_trace_missing_text_modality_is_rejected(self) -> None:
        run_dir, _ = _build_visualization_trace_bundle(self.base)
        shard_path = sorted((run_dir / "shards").glob("part-*.npz"))[0]
        with np.load(shard_path, allow_pickle=False) as loaded:
            values = {
                name: np.asarray(loaded[name]).copy()
                for name in loaded.files
                if name != "text_features"
            }
        np.savez_compressed(shard_path, **values)
        with self.assertRaisesRegex(ContractError, "sealed-bundle verification failed"):
            validate_bundle(run_dir)

    def test_cli_validate_emits_pass_json(self) -> None:
        fixture = build_valid_bundle(self.base)
        output = io.StringIO()
        with redirect_stdout(output):
            code = cli_main(
                [
                    "validate",
                    "--bundle",
                    str(fixture.root),
                    "--process-data-root",
                    str(fixture.process_root),
                    "--compact",
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "PASS")

    def test_json_schemas_are_valid_draft_2020_12(self) -> None:
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema is optional")
        schema_root = Path(__file__).parents[1] / "schema"
        for path in (schema_root / "bundle.schema.json", schema_root / "row.schema.json"):
            schema = json.loads(path.read_text(encoding="utf-8"))
            jsonschema.Draft202012Validator.check_schema(schema)

    def test_valid_fixture_conforms_to_both_json_schemas(self) -> None:
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema is optional")
        fixture = build_valid_bundle(self.base)
        schema_root = Path(__file__).parents[1] / "schema"
        bundle_schema = json.loads((schema_root / "bundle.schema.json").read_text(encoding="utf-8"))
        row_schema = json.loads((schema_root / "row.schema.json").read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator(bundle_schema).validate(read_manifest(fixture.root))
        validator = jsonschema.Draft202012Validator(row_schema)
        for row in read_rows(fixture.root):
            validator.validate(row)


if __name__ == "__main__":
    unittest.main()
