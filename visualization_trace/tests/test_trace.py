from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import scipy.io as sio
from scipy import sparse

from visualization_trace.adapters import (
    MIR_LABEL_NAMES,
    MIRFlickrCanonicalAdapter,
    MSCOCOAdapter,
    NUSWideAdapter,
)
from visualization_trace.core import ContentHashSplit, TraceContractError
from visualization_trace.extraction import (
    ExtractionConfig,
    canonicalize_runtime_repr,
    extract_trace_bundle,
    load_trace_bundle,
    preflight_adapter,
    verify_trace_bundle,
)


class RuntimeReprTests(unittest.TestCase):
    def test_process_address_is_removed_without_changing_structure(self) -> None:
        first = "<function _convert_image_to_rgb at 0x7F1234AB>"
        second = "<function _convert_image_to_rgb at 0x10>"
        expected = "<function _convert_image_to_rgb>"
        self.assertEqual(canonicalize_runtime_repr(first), expected)
        self.assertEqual(canonicalize_runtime_repr(second), expected)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _touch(path: Path, payload: bytes = b"synthetic-image") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _nus_fixture(root: Path) -> Path:
    data = root / "Data"
    oral = data / "OralData" / "NUS-WIDE"
    (oral / "ImageList").mkdir(parents=True)
    (oral / "images").mkdir(parents=True)
    names = ["0001_100", "0002_200", "0003_300", "0004_400"]
    (oral / "ImageList" / "Imagelist.txt").write_text(
        "".join(f"class\\{name}.jpg\n" for name in names), encoding="utf-8"
    )
    for index, name in enumerate(names):
        _touch(oral / "images" / f"{name}.jpg", f"image-{index}".encode())
    tags = [f"tag{index}" for index in range(1000)]
    (oral / "TagList1k.txt").write_text("\n".join(tags) + "\n", encoding="utf-8")
    (oral / "All_Tags.txt").write_text(
        "100 alpha beta\n200 gamma\n300 rare-only\n400 delta\n",
        encoding="utf-8",
    )
    clean = np.asarray([[3, 0, 2]], dtype=np.int64)
    sio.savemat(oral / "clean_id.nuswide.tc21.mat", {"clean_id": clean})
    labels = np.zeros((4, 21), dtype=np.uint8)
    labels[0, 1] = 1
    labels[2, 2] = 1
    labels[3, 3] = 1
    sio.savemat(
        oral / "labels.nuswide-tc21.mat", {"labels": sparse.csr_matrix(labels)}
    )
    texts = np.zeros((4, 1000), dtype=np.uint8)
    texts[0, 0] = 1
    texts[3, 3] = 1
    sio.savemat(
        oral / "texts.nuswide.AllTags1k.mat", {"texts": sparse.csr_matrix(texts)}
    )
    return data


def _coco_fixture(root: Path) -> Path:
    data = root / "Data"
    oral = data / "OralData" / "MSCOCO"
    ann = oral / "annotations_trainval2017"
    train_images = [
        {"id": 10, "file_name": "000000000010.jpg"},
        {"id": 5, "file_name": "000000000005.jpg"},
    ]
    val_images = [{"id": 2, "file_name": "000000000002.jpg"}]
    train_captions = [
        {"id": 1001, "image_id": 10, "caption": "ten first"},
        {"id": 1002, "image_id": 10, "caption": "ten second"},
        {"id": 1003, "image_id": 5, "caption": "excluded no instance"},
    ]
    val_captions = [{"id": 2001, "image_id": 2, "caption": "two caption"}]
    categories = [{"id": 1, "name": "alpha"}, {"id": 3, "name": "beta"}]
    _write_json(
        ann / "captions_train2017.json",
        {"images": train_images, "annotations": train_captions},
    )
    _write_json(
        ann / "captions_val2017.json",
        {"images": val_images, "annotations": val_captions},
    )
    _write_json(
        ann / "instances_train2017.json",
        {
            "categories": categories,
            "annotations": [{"image_id": 10, "category_id": 3}],
        },
    )
    _write_json(
        ann / "instances_val2017.json",
        {
            "categories": categories,
            "annotations": [{"image_id": 2, "category_id": 1}],
        },
    )
    for image in train_images:
        _touch(oral / "train2017" / image["file_name"], str(image["id"]).encode())
    for image in val_images:
        _touch(oral / "val2017" / image["file_name"], str(image["id"]).encode())
    return data


def _mir_fixture(root: Path) -> Path:
    data = root / "Data"
    oral = data / "OralData" / "MIRFLICK"
    raw = oral / "mirflickr25k" / "mirflickr"
    annotations = oral / "mirflickr25k_annotations_v080"
    annotations.mkdir(parents=True)
    (raw / "doc").mkdir(parents=True)
    (raw / "meta" / "tags").mkdir(parents=True)
    (raw / "doc" / "common_tags.txt").write_text(
        "valid 20\nsecond 25\nrare 19\n", encoding="utf-8"
    )
    for label in MIR_LABEL_NAMES:
        values = "3\n1\n2\n" if label == "animals" else ""
        (annotations / f"{label}.txt").write_text(values, encoding="utf-8")
    for image_id, text in ((1, "valid\nrare\n"), (2, "second\n"), (3, "valid\nsecond\n")):
        _touch(raw / f"im{image_id}.jpg", f"mir-{image_id}".encode())
        (raw / "meta" / "tags" / f"tags{image_id}.txt").write_text(
            text, encoding="utf-8"
        )
    return data


class FakeEncoder:
    model_id = "fake:deterministic-v1"

    def __init__(self, fail_on_image_call: int = 0) -> None:
        self.image_calls = 0
        self.text_calls = 0
        self.fail_on_image_call = fail_on_image_call

    @staticmethod
    def _vector(value: str) -> np.ndarray:
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        return np.frombuffer(digest[:16], dtype=np.uint32).astype(np.float32)

    def encode_images(self, image_paths: Sequence[str]) -> np.ndarray:
        self.image_calls += 1
        if self.fail_on_image_call and self.image_calls == self.fail_on_image_call:
            raise RuntimeError("synthetic interrupted encoder")
        return np.stack([self._vector(Path(value).name) for value in image_paths])

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        self.text_calls += 1
        return np.stack([self._vector(value) for value in texts])

    def contract(self) -> Dict[str, object]:
        return {"model_id": self.model_id, "output_dtype": "float32", "dim": 4}


class ContentHashSplitTests(unittest.TestCase):
    def test_deterministic_sizes_and_relations(self) -> None:
        identities = [f"sample-{index}" for index in range(20)]
        first = ContentHashSplit.build("demo", identities, 4, 7)
        second = ContentHashSplit.build("demo", identities, 4, 7)
        self.assertEqual(first.summary(), second.summary())
        self.assertEqual(len(first.ind_q), 4)
        self.assertEqual(len(first.ind_t), 7)
        self.assertEqual(len(first.ind_d), 16)
        self.assertFalse(first.ind_q & first.ind_d)
        self.assertEqual(first.ind_q | first.ind_d, frozenset(range(20)))
        self.assertTrue(first.ind_t <= first.ind_d)
        self.assertEqual(first.seed, 20_260_822)

    def test_rejects_duplicate_identity(self) -> None:
        with self.assertRaises(TraceContractError):
            ContentHashSplit.build("demo", ["same", "same"], 1, 1)


class AdapterTests(unittest.TestCase):
    def test_preflight_checks_canonical_contract_without_clip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = _nus_fixture(Path(temporary))
            adapter = NUSWideAdapter(
                data, expected_rows=3, query_rows=1, train_rows=1
            )
            report = preflight_adapter(adapter, limit=2)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["canonical_rows"], 3)
            self.assertEqual(report["label_dimension"], 21)
            self.assertEqual(report["split_counts"], {"indQ": 1, "indT": 1, "indD": 2})
            self.assertEqual(len(report["samples"]), 2)
            self.assertFalse(report["clip_loaded"])

    def test_nus_raw_only_mapping_and_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = _nus_fixture(Path(temporary))
            adapter = NUSWideAdapter(
                data, expected_rows=3, query_rows=1, train_rows=1
            )
            rows = list(adapter.iter_rows())
            self.assertEqual([row.source_index for row in rows], [0, 2, 3])
            self.assertEqual(rows[0].source_id, "raw-index:0|photo-id:100")
            self.assertEqual(rows[0].metadata["photo_id"], "100")
            self.assertEqual(rows[0].encoded_texts, ("tag0",))
            self.assertEqual(rows[1].encoded_texts, ("a generic photo",))
            self.assertTrue(rows[1].raw_text["baseline_fallback"])
            self.assertEqual(rows[1].raw_text["full_user_tags"], ["rare-only"])
            self.assertEqual(sum(rows[1].label_hot), 1)
            self.assertEqual(len(rows[1].label_hot), 21)
            self.assertFalse(
                any("ProcessData" in str(path) for path in adapter.source_artifacts)
            )
            self.assertEqual(sum(row.split["indQ"] for row in rows), 1)
            self.assertEqual(sum(row.split["indT"] for row in rows), 1)

    def test_coco_official_id_order_and_raw_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = _coco_fixture(Path(temporary))
            adapter = MSCOCOAdapter(
                data,
                expected_rows=2,
                classes=2,
                query_rows=1,
                train_rows=1,
            )
            rows = list(adapter.iter_rows())
            self.assertEqual([row.source_id for row in rows], ["2", "10"])
            self.assertEqual(rows[0].encoded_texts, ("two caption",))
            self.assertEqual(rows[1].encoded_texts, ("ten first", "ten second"))
            self.assertEqual(rows[0].label_hot, (1, 0))
            self.assertEqual(rows[1].label_hot, (0, 1))
            self.assertEqual(adapter.category_ids, (1, 3))
            self.assertEqual(adapter.contract()["category_ids"], [1, 3])
            self.assertFalse(
                any("ProcessData" in str(path) for path in adapter.source_artifacts)
            )

    def test_mir_numeric_order_prompts_labels_and_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = _mir_fixture(Path(temporary))
            adapter = MIRFlickrCanonicalAdapter(
                data, expected_rows=3, query_rows=1, train_rows=1
            )
            rows = list(adapter.iter_rows())
            self.assertEqual([row.source_id for row in rows], ["1", "2", "3"])
            self.assertEqual(rows[0].encoded_texts, ("a photo of valid",))
            self.assertEqual(
                rows[2].encoded_texts,
                ("a photo of valid", "a photo of second"),
            )
            self.assertEqual(rows[0].label_hot[0], 1)
            self.assertEqual(rows[0].source_id, str(int(rows[0].metadata["raw_image_id_token"])))
            self.assertEqual(sum(row.split["indQ"] for row in rows), 1)
            self.assertEqual(sum(row.split["indD"] for row in rows), 2)
            self.assertEqual(sum(row.split["indT"] for row in rows), 1)


class ExtractionTests(unittest.TestCase):
    def test_resume_after_interrupted_second_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = _mir_fixture(root)
            adapter = MIRFlickrCanonicalAdapter(
                data, expected_rows=3, query_rows=1, train_rows=1
            )
            output = root / "trace-output"
            config = ExtractionConfig(
                output_root=output,
                run_name="interrupted",
                batch_size=2,
                text_batch_size=2,
                shard_rows=2,
            )
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                extract_trace_bundle(adapter, FakeEncoder(fail_on_image_call=2), config)
            run_dir = output / "mirflickr" / "interrupted"
            self.assertEqual(
                len(list((run_dir / "receipts").glob("part-*.json"))), 1
            )
            resumed_encoder = FakeEncoder()
            complete = extract_trace_bundle(adapter, resumed_encoder, config)
            self.assertEqual(complete["rows"], 3)
            self.assertEqual(resumed_encoder.image_calls, 1)

    def test_shards_manifest_resume_and_poison_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = _mir_fixture(root)
            adapter = MIRFlickrCanonicalAdapter(
                data, expected_rows=3, query_rows=1, train_rows=1
            )
            output = root / "trace-output"
            encoder = FakeEncoder()
            config = ExtractionConfig(
                output_root=output,
                run_name="synthetic",
                batch_size=2,
                text_batch_size=2,
                shard_rows=2,
            )
            complete = extract_trace_bundle(adapter, encoder, config)
            self.assertEqual(complete["rows"], 3)
            self.assertEqual(complete["shards"], 2)
            self.assertGreater(encoder.image_calls, 0)
            run_dir = output / "mirflickr" / "synthetic"
            image, text, labels, records, loaded_complete = load_trace_bundle(run_dir)
            self.assertEqual(image.shape, (3, 4))
            self.assertEqual(text.shape, (3, 4))
            self.assertEqual(labels.shape, (3, 24))
            self.assertEqual(len(records), 3)
            self.assertEqual(loaded_complete, complete)
            self.assertTrue((run_dir / "canonical_split.npz").is_file())
            with np.load(run_dir / "canonical_split.npz", allow_pickle=False) as split:
                self.assertEqual(split["row_ids"].dtype, np.dtype("S64"))
                self.assertEqual(split["row_ids"].shape, (3,))
                split_row_ids = split["row_ids"].copy()
            shard_row_ids = []
            for shard_path in sorted((run_dir / "shards").glob("part-*.npz")):
                with np.load(shard_path, allow_pickle=False) as shard:
                    shard_row_ids.append(shard["row_ids"].copy())
            self.assertTrue(np.array_equal(split_row_ids, np.concatenate(shard_row_ids)))
            self.assertTrue(all(record["image_sha256"] for record in records))
            self.assertTrue(all(len(record["canonical_row_id"]) == 64 for record in records))
            self.assertTrue(
                all(record["raw_text_sources"][0]["sha256"] for record in records)
            )
            verification = verify_trace_bundle(run_dir)
            semantics = verification["contract"]["feature_semantics"]
            self.assertFalse(
                semantics["per_text_embedding"]["l2_normalized_by_extractor"]
            )
            self.assertFalse(
                semantics["text_row_aggregation"]["input_l2_normalized"]
            )
            self.assertFalse(
                semantics["text_row_aggregation"]["output_l2_normalized"]
            )
            expected_first_text = FakeEncoder._vector("a photo of valid")
            self.assertTrue(np.array_equal(text[0], expected_first_text))
            expected_third_text = np.asarray(
                np.stack(
                    [
                        FakeEncoder._vector("a photo of valid"),
                        FakeEncoder._vector("a photo of second"),
                    ]
                ).astype(np.float64).mean(axis=0),
                dtype=np.float32,
            )
            self.assertTrue(np.array_equal(text[2], expected_third_text))

            resumed_encoder = FakeEncoder()
            resumed = extract_trace_bundle(adapter, resumed_encoder, config)
            self.assertEqual(resumed, complete)
            self.assertEqual(resumed_encoder.image_calls, 0)
            self.assertEqual(resumed_encoder.text_calls, 0)

            manifest = run_dir / "manifests" / "part-000000.jsonl"
            manifest.write_bytes(manifest.read_bytes() + b"poison\n")
            with self.assertRaises(TraceContractError):
                extract_trace_bundle(adapter, FakeEncoder(), config)

    def test_forbids_output_inside_raw_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = _mir_fixture(root)
            adapter = MIRFlickrCanonicalAdapter(
                data, expected_rows=3, query_rows=1, train_rows=1
            )
            config = ExtractionConfig(
                output_root=data / "OralData" / "illegal",
                run_name="synthetic",
                shard_rows=2,
            )
            with self.assertRaises(TraceContractError):
                extract_trace_bundle(adapter, FakeEncoder(), config)


if __name__ == "__main__":
    unittest.main()
