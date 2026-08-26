from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from unittest import mock

import numpy as np

from raw_rebuilt_runtime import (
    RuntimeBridgeError,
    load_indt_training_inputs,
    load_label_free_rank_inputs,
    load_metric_labels,
    materialize_runtime,
    verify_runtime_directory,
)
from raw_rebuilt_runtime.contract import require_geometry
from raw_rebuilt_runtime.metric_loader import load_frozen_metric_labels
from visualization_trace.core import ContentHashSplit, TraceRow, sha256_json
from visualization_trace.extraction import ExtractionConfig, extract_trace_bundle


class SyntheticAdapter:
    dataset = "synthetic"

    def __init__(self, root: Path) -> None:
        self.data_root = (root / "Data").resolve()
        oral = self.data_root / "OralData" / "Synthetic"
        oral.mkdir(parents=True)
        annotations = []
        labels = []
        self.image_paths = []
        for index in range(4):
            image = (oral / f"image-{index}.bin").resolve()
            image.write_bytes(f"raw-image-{index}".encode("ascii"))
            self.image_paths.append(image)
            annotations.append(
                {"id": 100 + index, "image_id": index, "caption": f"caption {index}"}
            )
            label = [0, 0, 0]
            label[index % 3] = 1
            labels.append(label)
        self.annotation_path = (oral / "raw.json").resolve()
        self.annotation_path.write_text(
            json.dumps({"annotations": annotations, "labels": labels}),
            encoding="utf-8",
        )
        self.source_artifacts = tuple([self.annotation_path, *self.image_paths])
        self.rows = 4
        self.source_ids = tuple(f"sample-{index}" for index in range(self.rows))
        self.split = ContentHashSplit.build(
            self.dataset, self.source_ids, query_rows=1, train_rows=2
        )

    def dependency_paths(self) -> tuple[Path, ...]:
        return tuple(sorted(self.source_artifacts, key=lambda value: str(value)))

    def contract(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "rows": self.rows,
            "data_root": str(self.data_root),
            "source_artifacts": [str(path) for path in self.source_artifacts],
            "split": self.split.summary(),
            "identity_chain": "OralData only",
            "process_data_role": "none; synthetic test only",
            "canonical_order": "ascending zero-based raw source-index",
        }

    def iter_rows(self, start: int = 0) -> Iterator[TraceRow]:
        for index in range(start, self.rows):
            caption = f"caption {index}"
            annotation_payload = {
                "annotation_id": 100 + index,
                "image_id": index,
                "caption": caption,
            }
            label = [0, 0, 0]
            label[index % 3] = 1
            yield TraceRow(
                dataset=self.dataset,
                row_index=index,
                source_index=index,
                source_id=self.source_ids[index],
                image_path=str(self.image_paths[index]),
                encoded_texts=(caption,),
                raw_text={"caption": caption},
                label_hot=tuple(label),
                split=self.split.flags(index),
                metadata={
                    "label_source": {
                        "path": str(self.annotation_path),
                        "kind": "json_pointer",
                        "json_pointer": f"/labels/{index}",
                    }
                },
                raw_text_locators=(
                    {
                        "path": str(self.annotation_path),
                        "kind": "json_annotation",
                        "locator": {
                            "json_pointer": f"/annotations/{index}",
                            "annotation_id": 100 + index,
                            "image_id": index,
                        },
                        "content_sha256": sha256_json(annotation_payload),
                    },
                ),
            )


class FakeClip512:
    model_id = "synthetic:clip512"

    @staticmethod
    def _vector(value: str) -> np.ndarray:
        seed = hashlib.sha256(value.encode("utf-8")).digest()
        base = np.frombuffer(seed, dtype=np.uint8).astype(np.float32) + 1.0
        return np.tile(base, 16)[:512]

    def encode_images(self, image_paths: Sequence[str]) -> np.ndarray:
        return np.stack([self._vector(Path(value).name) for value in image_paths])

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vector(value) for value in texts])

    def contract(self) -> Mapping[str, Any]:
        return {"model_id": self.model_id, "output_dtype": "float32", "dim": 512}


def build_source(root: Path) -> tuple[Path, Path]:
    adapter = SyntheticAdapter(root)
    output = root / "sealed-trace"
    config = ExtractionConfig(
        output_root=output,
        run_name="raw_rebuilt_v1",
        batch_size=2,
        text_batch_size=2,
        shard_rows=2,
    )
    extract_trace_bundle(adapter, FakeClip512(), config)
    return output / "synthetic" / "raw_rebuilt_v1", adapter.data_root


class RuntimeBridgeTests(unittest.TestCase):
    def test_synthetic_e2e_resume_and_runner_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, data_root = build_source(root)
            runtime = root / "runtime"
            first = materialize_runtime(
                source,
                runtime,
                process_data_root=data_root / "ProcessData",
                max_new_parts=1,
                _test_allow_synthetic=True,
            )
            self.assertEqual(first["status"], "IN_PROGRESS")
            self.assertEqual(first["parts_committed"], 1)
            completed = materialize_runtime(
                source,
                runtime,
                process_data_root=data_root / "ProcessData",
                _test_allow_synthetic=True,
            )
            self.assertEqual(completed["status"], "COMPLETE")
            self.assertTrue(completed["resumed"])
            manifest = verify_runtime_directory(
                runtime,
                process_data_root=data_root / "ProcessData",
                _test_allow_synthetic=True,
            )
            self.assertEqual(manifest["rows"], 4)
            rank = load_label_free_rank_inputs(
                runtime,
                process_data_root=data_root / "ProcessData",
                _test_allow_synthetic=True,
            )
            self.assertEqual(rank.image.shape, (4, 512))
            self.assertFalse(hasattr(rank, "labels"))
            train = load_indt_training_inputs(
                runtime,
                process_data_root=data_root / "ProcessData",
                _test_allow_synthetic=True,
            )
            self.assertEqual(train.image.shape, (2, 512))
            self.assertEqual(train.labels.shape, (2, 3))
            metric = load_metric_labels(
                runtime,
                rank_contract={
                    "status": "rank_state_frozen",
                    "labels_loaded_during_freeze": False,
                    "source_seal_sha256": rank.source_seal_sha256,
                },
                process_data_root=data_root / "ProcessData",
                _test_allow_synthetic=True,
            )
            self.assertEqual(metric.query.shape, (1, 3))
            self.assertEqual(metric.database.shape, (3, 3))
            rank.close()

    def test_resume_rejects_same_shape_partial_poison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, data_root = build_source(root)
            runtime = root / "runtime"
            materialize_runtime(
                source,
                runtime,
                max_new_parts=1,
                process_data_root=data_root / "ProcessData",
                _test_allow_synthetic=True,
            )
            image = np.load(
                runtime / "arrays" / "image_features_clip512.npy",
                mmap_mode="r+",
                allow_pickle=False,
            )
            image[0, 0] += np.float32(1.0)
            image.flush()
            image._mmap.close()
            with self.assertRaisesRegex(RuntimeBridgeError, "same-shape poison"):
                materialize_runtime(
                    source,
                    runtime,
                    process_data_root=data_root / "ProcessData",
                    _test_allow_synthetic=True,
                )

    def test_completed_rejects_row_reorder_and_extra_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, data_root = build_source(root)
            runtime = root / "runtime"
            materialize_runtime(
                source,
                runtime,
                process_data_root=data_root / "ProcessData",
                _test_allow_synthetic=True,
            )
            row_ids = np.load(
                runtime / "arrays" / "row_ids.npy", mmap_mode="r+", allow_pickle=False
            )
            first = row_ids[0].copy()
            row_ids[0] = row_ids[1]
            row_ids[1] = first
            row_ids.flush()
            row_ids._mmap.close()
            with self.assertRaises(RuntimeBridgeError):
                verify_runtime_directory(
                    runtime,
                    process_data_root=data_root / "ProcessData",
                    _test_allow_synthetic=True,
                )

            # Rebuild a clean runtime and prove the inventory is exact.
            clean = root / "clean-runtime"
            materialize_runtime(
                source,
                clean,
                process_data_root=data_root / "ProcessData",
                _test_allow_synthetic=True,
            )
            (clean / "unbound.txt").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeBridgeError, "extra files"):
                verify_runtime_directory(
                    clean,
                    process_data_root=data_root / "ProcessData",
                    _test_allow_synthetic=True,
                )

    def test_source_same_shape_poison_is_rejected_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, data_root = build_source(root)
            shard = source / "shards" / "part-000000.npz"
            with np.load(shard, allow_pickle=False) as loaded:
                values = {name: np.asarray(loaded[name]).copy() for name in loaded.files}
            values["image_features"][0, 0] += np.float32(1.0)
            with shard.open("wb") as handle:
                np.savez_compressed(handle, **values)
            runtime = root / "runtime"
            with self.assertRaisesRegex(RuntimeBridgeError, "verify_trace_bundle rejected"):
                materialize_runtime(
                    source,
                    runtime,
                    process_data_root=data_root / "ProcessData",
                    _test_allow_synthetic=True,
                )
            self.assertFalse((runtime / "runtime_contract.json").exists())

    def test_provenance_pass_is_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, data_root = build_source(root)
            runtime = root / "runtime"
            with mock.patch(
                "visualization_feature_pipeline.validate_bundle",
                side_effect=ValueError("synthetic provenance rejection"),
            ):
                with self.assertRaisesRegex(RuntimeBridgeError, "provenance validator rejected"):
                    materialize_runtime(
                        source,
                        runtime,
                        process_data_root=data_root / "ProcessData",
                        _test_allow_synthetic=True,
                    )
            self.assertFalse((runtime / "runtime_contract.json").exists())

    def test_output_boundaries_and_nus_81_hot_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, data_root = build_source(root)
            with self.assertRaises(RuntimeBridgeError):
                materialize_runtime(
                    source,
                    data_root / "ProcessData" / "runtime",
                    process_data_root=data_root / "ProcessData",
                    _test_allow_synthetic=True,
                )
            with self.assertRaisesRegex(RuntimeBridgeError, "81-hot legacy labels"):
                require_geometry(
                    "nuswide", rows=195_834, feature_dim=512, label_dim=81
                )

    def test_metric_labels_fail_before_rank_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, data_root = build_source(root)
            runtime = root / "runtime"
            materialize_runtime(
                source,
                runtime,
                process_data_root=data_root / "ProcessData",
                _test_allow_synthetic=True,
            )
            with self.assertRaisesRegex(RuntimeBridgeError, "rank_state_frozen"):
                load_metric_labels(
                    runtime,
                    rank_contract={"status": "training"},
                    process_data_root=data_root / "ProcessData",
                    _test_allow_synthetic=True,
                )

    def test_metric_gate_reopens_frozen_source_without_raw_rematerialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, data_root = build_source(root)
            runtime = root / "runtime"
            materialize_runtime(
                source,
                runtime,
                process_data_root=data_root / "ProcessData",
                _test_allow_synthetic=True,
            )
            rank = load_label_free_rank_inputs(
                runtime,
                process_data_root=data_root / "ProcessData",
                _test_allow_synthetic=True,
            )
            # Materialization already recorded the independent raw provenance
            # PASS.  The post-freeze metric gate must verify that seal and all
            # scoring bytes without invoking the full raw-row validator again.
            with mock.patch(
                "visualization_feature_pipeline.validate_bundle",
                side_effect=AssertionError("raw validator was rematerialized"),
            ):
                metric = load_frozen_metric_labels(
                    runtime,
                    rank_contract={
                        "status": "rank_state_frozen",
                        "labels_loaded_during_freeze": False,
                        "source_seal_sha256": rank.source_seal_sha256,
                    },
                    process_data_root=data_root / "ProcessData",
                    _test_allow_synthetic=True,
                )
            self.assertEqual(metric.query.shape, (1, 3))
            self.assertEqual(metric.database.shape, (3, 3))
            rank.close()

    def test_rank_freeze_does_not_decode_labels_but_metric_gate_does(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, data_root = build_source(root)
            runtime = root / "runtime"
            materialize_runtime(
                source,
                runtime,
                process_data_root=data_root / "ProcessData",
                _test_allow_synthetic=True,
            )
            labels = np.load(
                runtime / "arrays" / "labels.npy", mmap_mode="r+", allow_pickle=False
            )
            labels[0] = np.roll(labels[0], 1)
            labels.flush()
            labels._mmap.close()

            # Rank input verification reads only the label NPY header/size.  It
            # neither decodes nor hashes label rows during the freeze stage.
            rank = load_label_free_rank_inputs(
                runtime,
                process_data_root=data_root / "ProcessData",
                _test_allow_synthetic=True,
            )
            self.assertFalse(rank.labels_loaded_during_freeze)
            seal = rank.source_seal_sha256
            rank.close()

            # The post-freeze metric boundary performs full verification and
            # therefore rejects the poisoned label content before scoring.
            with self.assertRaises(RuntimeBridgeError):
                load_frozen_metric_labels(
                    runtime,
                    rank_contract={
                        "status": "rank_state_frozen",
                        "labels_loaded_during_freeze": False,
                        "source_seal_sha256": seal,
                    },
                    process_data_root=data_root / "ProcessData",
                    _test_allow_synthetic=True,
                )


if __name__ == "__main__":
    unittest.main()
