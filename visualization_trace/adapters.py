from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from .core import (
    ContentHashSplit,
    TraceContractError,
    TraceRow,
    dense_row,
    load_mat_variable,
    padded_ids_to_hot,
    require_file,
    sha256_json,
    validate_expected_rows,
)


MIR_LABEL_NAMES: Tuple[str, ...] = (
    "animals",
    "baby",
    "bird",
    "car",
    "clouds",
    "dog",
    "female",
    "flower",
    "food",
    "indoor",
    "lake",
    "male",
    "night",
    "people",
    "plant_life",
    "portrait",
    "river",
    "sea",
    "sky",
    "structures",
    "sunset",
    "transport",
    "tree",
    "water",
)

NUS_TC21_LABEL_NAMES: Tuple[str, ...] = (
    "animal",
    "beach",
    "buildings",
    "clouds",
    "flowers",
    "grass",
    "lake",
    "mountain",
    "ocean",
    "person",
    "plants",
    "reflection",
    "road",
    "rocks",
    "sky",
    "snow",
    "sunset",
    "tree",
    "vehicle",
    "water",
    "window",
)


class DatasetAdapter(ABC):
    dataset: str
    rows: int
    data_root: Path
    source_artifacts: Tuple[Path, ...]
    split: Optional[Any]

    @abstractmethod
    def iter_rows(self, start: int = 0) -> Iterator[TraceRow]:
        raise NotImplementedError

    def contract(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "rows": self.rows,
            "data_root": str(self.data_root),
            "source_artifacts": [str(path) for path in self.source_artifacts],
            "split": self.split.summary() if self.split is not None else None,
        }

    def dependency_paths(self) -> Tuple[Path, ...]:
        paths = list(self.source_artifacts)
        paths.extend(getattr(self, "image_paths", ()))
        paths.extend(getattr(self, "tag_paths", ()))
        unique: Dict[str, Path] = {}
        for raw in paths:
            path = Path(raw).expanduser().resolve()
            unique[str(path)] = path
        return tuple(unique[key] for key in sorted(unique))


def _read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return [line.strip() for line in handle]


class NUSWideAdapter(DatasetAdapter):
    dataset = "nuswide"

    def __init__(
        self,
        data_root: os.PathLike[str],
        expected_rows: Optional[int] = 195_834,
        require_images: bool = True,
        query_rows: int = 2_085,
        train_rows: int = 21_000,
        split_seed: int = 20_260_822,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        oral = self.data_root / "OralData" / "NUS-WIDE"
        self.oral_root = oral

        self.clean_path = require_file(
            oral / "clean_id.nuswide.tc21.mat", "NUS clean_id"
        )
        self.raw_labels_path = require_file(
            oral / "labels.nuswide-tc21.mat", "NUS TC21 label matrix"
        )
        self.raw_text_path = require_file(
            oral / "texts.nuswide.AllTags1k.mat", "NUS top-1k text matrix"
        )
        self.tag_list_path = require_file(oral / "TagList1k.txt", "NUS tag list")
        self.all_tags_path = require_file(oral / "All_Tags.txt", "NUS full raw tags")
        self.image_list_path = require_file(
            oral / "ImageList" / "Imagelist.txt", "NUS image list"
        )
        clean_members = np.asarray(
            load_mat_variable(self.clean_path, "clean_id")
        ).reshape(-1).astype(np.int64)
        self.clean_id = np.sort(clean_members)
        self.rows = int(self.clean_id.size)
        validate_expected_rows(self.rows, expected_rows, self.dataset)
        if np.unique(self.clean_id).size != self.rows:
            raise TraceContractError("NUS clean_id contains duplicates")

        self.raw_labels = load_mat_variable(self.raw_labels_path, "labels")
        self.raw_texts = load_mat_variable(self.raw_text_path, "texts")
        if sparse.issparse(self.raw_texts):
            self.raw_texts = self.raw_texts.tocsr()
        if self.raw_labels.shape[0] != self.raw_texts.shape[0]:
            raise TraceContractError("NUS raw label/text row counts disagree")
        raw_rows = int(self.raw_labels.shape[0])
        if self.clean_id.size and (
            int(self.clean_id.min()) < 0 or int(self.clean_id.max()) >= raw_rows
        ):
            raise TraceContractError("NUS clean_id is not a zero-based raw-row index")
        if int(self.raw_labels.shape[1]) != 21:
            raise TraceContractError(
                f"NUS labels must be TC21 21-hot; shape={self.raw_labels.shape}"
            )
        if int(self.raw_texts.shape[1]) != 1000:
            raise TraceContractError(
                f"NUS baseline text must have 1000 columns; shape={self.raw_texts.shape}"
            )
        text_values = (
            self.raw_texts.data
            if sparse.issparse(self.raw_texts)
            else np.asarray(self.raw_texts).reshape(-1)
        )
        if text_values.size and not np.all(np.isin(text_values, (0, 1))):
            raise TraceContractError("NUS AllTags1k matrix must be binary")

        self.tags = _read_lines(self.tag_list_path)
        if len(self.tags) != 1000:
            raise TraceContractError(f"NUS TagList1k must contain 1000 rows")
        listed = _read_lines(self.image_list_path)
        if len(listed) != raw_rows:
            raise TraceContractError(
                f"NUS image list rows={len(listed)} but raw rows={raw_rows}"
            )
        self.image_names = [
            Path(line.replace("\\", "/").rsplit("/", 1)[-1]).stem for line in listed
        ]
        if len(set(self.image_names)) != len(self.image_names):
            raise TraceContractError("NUS ImageList contains duplicate basenames")

        selected_raw = set(int(v) for v in self.clean_id)
        self.full_tags: Dict[int, Tuple[str, ...]] = {}
        all_tag_rows = 0
        with self.all_tags_path.open(
            "r", encoding="utf-8", errors="replace"
        ) as handle:
            for raw_index, line in enumerate(handle):
                all_tag_rows += 1
                if raw_index not in selected_raw:
                    continue
                fields = line.strip().split()
                expected_photo_id = self.image_names[raw_index].rsplit("_", 1)[-1]
                if not fields or fields[0] != expected_photo_id:
                    raise TraceContractError(
                        "NUS All_Tags/ImageList photo ID mismatch at raw row "
                        f"{raw_index}: tags={fields[:1]}, image={self.image_names[raw_index]}"
                    )
                self.full_tags[raw_index] = tuple(fields[1:])
        if all_tag_rows != raw_rows:
            raise TraceContractError(
                f"NUS All_Tags rows={all_tag_rows}, expected={raw_rows}"
            )
        if len(self.full_tags) != self.rows:
            raise TraceContractError("NUS full raw tags did not cover every clean row")

        selected_labels = self.raw_labels[self.clean_id]
        if sparse.issparse(selected_labels):
            selected_labels = selected_labels.toarray()
        selected_labels = np.asarray(selected_labels)
        if selected_labels.size and not np.all(np.isin(selected_labels, (0, 1))):
            raise TraceContractError("NUS raw labels must be binary")
        self.labels = selected_labels.astype(np.uint8, copy=False)

        self.image_paths: List[Path] = []
        for row_index, raw_index in enumerate(self.clean_id):
            raw_index = int(raw_index)
            image_path = (
                oral / "images" / (self.image_names[raw_index] + ".jpg")
            ).resolve()
            if require_images and not image_path.is_file():
                raise TraceContractError(f"missing NUS raw image: {image_path}")
            self.image_paths.append(image_path)

        split_ids = [
            f"raw-index:{int(raw_index)}|photo-id:{self.image_names[int(raw_index)].rsplit('_', 1)[-1]}"
            for raw_index in self.clean_id
        ]
        self.split = ContentHashSplit.build(
            self.dataset,
            split_ids,
            query_rows=query_rows,
            train_rows=train_rows,
            seed=split_seed,
        )
        self.source_artifacts = (
            self.clean_path,
            self.raw_labels_path,
            self.raw_text_path,
            self.tag_list_path,
            self.all_tags_path,
            self.image_list_path,
        )

    def contract(self) -> Dict[str, Any]:
        contract = super().contract()
        contract.update(
            {
                "canonical_order": "ascending zero-based raw clean_id",
                "identity_chain": "OralData only",
                "process_data_role": "none; may be compared outside this adapter only",
                "label_protocol": "NUS-WIDE TC21",
                "label_dimension": 21,
                "label_names": list(NUS_TC21_LABEL_NAMES),
            }
        )
        return contract

    def iter_rows(self, start: int = 0) -> Iterator[TraceRow]:
        if not 0 <= start <= self.rows:
            raise TraceContractError(f"invalid NUS start row {start}")
        for row_index in range(start, self.rows):
            raw_index = int(self.clean_id[row_index])
            text_row = dense_row(self.raw_texts, raw_index)
            active = np.flatnonzero(text_row)
            top1k = tuple(self.tags[int(index)] for index in active)
            fallback = not top1k
            encoded = top1k if top1k else ("a generic photo",)
            photo_id = self.image_names[raw_index].rsplit("_", 1)[-1]
            canonical_source_id = f"raw-index:{raw_index}|photo-id:{photo_id}"
            yield TraceRow(
                dataset=self.dataset,
                row_index=row_index,
                source_index=raw_index,
                source_id=canonical_source_id,
                image_path=str(self.image_paths[row_index]),
                encoded_texts=encoded,
                raw_text={
                    "baseline_top1k_tags": list(top1k),
                    "full_user_tags": list(self.full_tags[raw_index]),
                    "baseline_fallback": fallback,
                    "full_tags_not_encoded": True,
                },
                label_hot=tuple(int(v) for v in self.labels[row_index]),
                split=self.split.flags(row_index),
                metadata={
                    "clean_id": raw_index,
                    "photo_id": photo_id,
                    "image_list_name": self.image_names[raw_index] + ".jpg",
                    "baseline_text_source": "AllTags1k",
                    "raw_text_sources": [
                        str(self.raw_text_path),
                        str(self.tag_list_path),
                        str(self.all_tags_path),
                    ],
                    "label_source": {
                        "path": str(self.raw_labels_path),
                        "variable": "labels",
                        "row": raw_index,
                        "index_base": 0,
                        "protocol": "TC21",
                    },
                },
                raw_text_locators=(
                    {
                        "path": str(self.raw_text_path),
                        "kind": "mat_row",
                        "locator": {
                            "variable": "texts",
                            "row": raw_index,
                            "index_base": 0,
                            "active_columns": [int(value) for value in active],
                        },
                        "content_sha256": sha256_json(
                            {
                                "active_columns": [int(value) for value in active],
                                "tags": list(top1k),
                            }
                        ),
                    },
                    {
                        "path": str(self.tag_list_path),
                        "kind": "text_line_selection",
                        "locator": {
                            "line_indices": [int(value) for value in active],
                            "index_base": 0,
                        },
                        "content_sha256": sha256_json(list(top1k)),
                    },
                    {
                        "path": str(self.all_tags_path),
                        "kind": "text_line",
                        "locator": {
                            "line_index": raw_index,
                            "index_base": 0,
                            "photo_id": photo_id,
                        },
                        "content_sha256": sha256_json(
                            {
                                "photo_id": photo_id,
                                "full_user_tags": list(self.full_tags[raw_index]),
                            }
                        ),
                    },
                ),
            )


class MSCOCOAdapter(DatasetAdapter):
    dataset = "mscoco"

    def __init__(
        self,
        data_root: os.PathLike[str],
        expected_rows: Optional[int] = 122_218,
        require_images: bool = True,
        classes: int = 80,
        query_rows: int = 5_000,
        train_rows: int = 10_500,
        split_seed: int = 20_260_822,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        oral = self.data_root / "OralData" / "MSCOCO"
        ann = oral / "annotations_trainval2017"
        self.oral_root = oral
        self.classes = int(classes)

        self.caption_train_path = require_file(
            ann / "captions_train2017.json", "COCO train captions"
        )
        self.caption_val_path = require_file(
            ann / "captions_val2017.json", "COCO val captions"
        )
        self.instance_train_path = require_file(
            ann / "instances_train2017.json", "COCO train instances"
        )
        self.instance_val_path = require_file(
            ann / "instances_val2017.json", "COCO val instances"
        )
        def load_json(path: Path) -> Mapping[str, Any]:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        caption_train = load_json(self.caption_train_path)
        caption_val = load_json(self.caption_val_path)
        instance_train = load_json(self.instance_train_path)
        instance_val = load_json(self.instance_val_path)

        caption_records: Dict[int, List[Dict[str, Any]]] = {}
        for source_path, source_data in (
            (self.caption_train_path, caption_train),
            (self.caption_val_path, caption_val),
        ):
            for annotation_index, annotation in enumerate(source_data["annotations"]):
                if "id" not in annotation:
                    raise TraceContractError(
                        f"COCO caption annotation lacks official ID at {source_path}:"
                        f"/annotations/{annotation_index}"
                    )
                image_id = int(annotation["image_id"])
                caption_records.setdefault(image_id, []).append(
                    {
                        "caption": str(annotation["caption"]),
                        "annotation_id": int(annotation["id"]),
                        "annotation_index": annotation_index,
                        "source_path": source_path,
                    }
                )
        captions = {
            image_id: [record["caption"] for record in records]
            for image_id, records in caption_records.items()
        }
        category_ids: Dict[int, List[int]] = {}
        for annotation in instance_train["annotations"] + instance_val["annotations"]:
            category_ids.setdefault(int(annotation["image_id"]), []).append(
                int(annotation["category_id"])
            )

        images = caption_train["images"] + caption_val["images"]
        valid_images = sorted(
            [
            image
            for image in images
            if int(image["id"]) in captions and int(image["id"]) in category_ids
            ],
            key=lambda image: int(image["id"]),
        )
        self.rows = len(valid_images)
        validate_expected_rows(self.rows, expected_rows, self.dataset)
        self.ids = np.asarray(
            [int(image["id"]) for image in valid_images], dtype=np.int64
        )
        if np.unique(self.ids).size != self.rows:
            raise TraceContractError("COCO image IDs are not unique")

        train_categories = tuple(
            (int(category["id"]), str(category["name"]))
            for category in instance_train["categories"]
        )
        val_categories = tuple(
            (int(category["id"]), str(category["name"]))
            for category in instance_val["categories"]
        )
        if train_categories != val_categories:
            raise TraceContractError(
                "COCO train/val official category axes differ in ID, name, or order"
            )
        if len(train_categories) < self.classes:
            raise TraceContractError(
                f"COCO requested {self.classes} classes but only "
                f"{len(train_categories)} exist"
            )
        selected_categories = train_categories[: self.classes]
        self.category_ids = tuple(category_id for category_id, _ in selected_categories)
        self.category_names = tuple(name for _, name in selected_categories)
        if len(set(self.category_ids)) != self.classes:
            raise TraceContractError("COCO official category IDs are not unique")
        if len(set(self.category_names)) != self.classes:
            raise TraceContractError("COCO official category names are not unique")
        category_id_to_axis = {
            category_id: axis for axis, category_id in enumerate(self.category_ids)
        }

        labels_hot: List[np.ndarray] = []
        for image_id in self.ids:
            hot = np.zeros(self.classes, dtype=np.uint8)
            for category_id in category_ids[int(image_id)]:
                if category_id not in category_id_to_axis:
                    raise TraceContractError(
                        f"COCO image {int(image_id)} references category ID "
                        f"{category_id} outside the frozen category axis"
                    )
                hot[category_id_to_axis[category_id]] = 1
            labels_hot.append(hot)
        self.labels_hot = np.stack(labels_hot)

        train_names = {str(image["file_name"]) for image in caption_train["images"]}
        image_by_id = {int(image["id"]): image for image in valid_images}
        self.image_paths: List[Path] = []
        self.captions: List[Tuple[str, ...]] = []
        self.caption_sources: List[Path] = []
        self.caption_records: List[Tuple[Mapping[str, Any], ...]] = []
        for row_index, image_id in enumerate(self.ids):
            image = image_by_id[int(image_id)]
            file_name = str(image["file_name"])
            split_dir = "train2017" if file_name in train_names else "val2017"
            expected_captions = tuple(captions[int(image_id)])
            image_path = (oral / split_dir / file_name).resolve()
            if require_images and not image_path.is_file():
                raise TraceContractError(f"missing COCO raw image: {image_path}")
            self.image_paths.append(image_path)
            self.captions.append(expected_captions)
            self.caption_records.append(tuple(caption_records[int(image_id)]))
            self.caption_sources.append(
                self.caption_train_path
                if split_dir == "train2017"
                else self.caption_val_path
            )

        self.split = ContentHashSplit.build(
            self.dataset,
            [str(int(image_id)) for image_id in self.ids],
            query_rows=query_rows,
            train_rows=train_rows,
            seed=split_seed,
        )
        self.source_artifacts = (
            self.caption_train_path,
            self.caption_val_path,
            self.instance_train_path,
            self.instance_val_path,
        )

    def contract(self) -> Dict[str, Any]:
        contract = super().contract()
        contract.update(
            {
                "canonical_order": "ascending official COCO image ID",
                "identity_chain": "OralData official JSON only",
                "process_data_role": "none; may be compared outside this adapter only",
                "category_ids": list(self.category_ids),
                "category_names": list(self.category_names),
                "category_axis": [
                    {"axis": axis, "official_category_id": category_id, "name": name}
                    for axis, (category_id, name) in enumerate(
                        zip(self.category_ids, self.category_names)
                    )
                ],
            }
        )
        return contract

    def iter_rows(self, start: int = 0) -> Iterator[TraceRow]:
        if not 0 <= start <= self.rows:
            raise TraceContractError(f"invalid COCO start row {start}")
        for row_index in range(start, self.rows):
            image_id = int(self.ids[row_index])
            captions = self.captions[row_index]
            caption_records = self.caption_records[row_index]
            yield TraceRow(
                dataset=self.dataset,
                row_index=row_index,
                source_index=row_index,
                source_id=str(image_id),
                image_path=str(self.image_paths[row_index]),
                encoded_texts=captions,
                raw_text={"captions": list(captions)},
                label_hot=tuple(int(v) for v in self.labels_hot[row_index]),
                split=self.split.flags(row_index),
                metadata={
                    "coco_image_id": image_id,
                    "category_ids": list(self.category_ids),
                    "category_names": list(self.category_names),
                    "text_aggregation": "mean_of_all_caption_embeddings",
                    "raw_text_source": str(self.caption_sources[row_index]),
                    "label_source": {
                        "path": str(
                            self.instance_train_path
                            if "train2017" in str(self.image_paths[row_index])
                            else self.instance_val_path
                        ),
                        "image_id": image_id,
                        "protocol": "official COCO instance categories mapped to 80-hot",
                    },
                },
                raw_text_locators=tuple(
                    {
                        "path": str(record["source_path"]),
                        "kind": "json_annotation",
                        "locator": {
                            "json_pointer": f"/annotations/{int(record['annotation_index'])}",
                            "annotation_id": int(record["annotation_id"]),
                            "image_id": image_id,
                        },
                        "content_sha256": sha256_json(
                            {
                                "annotation_id": int(record["annotation_id"]),
                                "image_id": image_id,
                                "caption": str(record["caption"]),
                            }
                        ),
                    }
                    for record in caption_records
                ),
            )


class MIRFlickrCanonicalAdapter(DatasetAdapter):
    """Deterministic raw member set; it is not the lost baseline row order."""

    dataset = "mirflickr"

    def __init__(
        self,
        data_root: os.PathLike[str],
        expected_rows: Optional[int] = 20_015,
        require_images: bool = True,
        query_rows: int = 2_243,
        train_rows: int = 5_000,
        split_seed: int = 20_260_822,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        oral = self.data_root / "OralData" / "MIRFLICK"
        raw = oral / "mirflickr25k" / "mirflickr"
        annotations = oral / "mirflickr25k_annotations_v080"
        self.oral_root = oral
        self.raw_root = raw
        self.annotation_root = annotations
        self.common_tags_path = require_file(
            raw / "doc" / "common_tags.txt", "MIR common tags"
        )

        valid_tags = set()
        with self.common_tags_path.open(
            "r", encoding="utf-8", errors="replace"
        ) as handle:
            for line in handle:
                fields = line.strip().split()
                if len(fields) != 2:
                    raise TraceContractError(f"malformed MIR common_tags line: {line!r}")
                if int(fields[1]) >= 20:
                    valid_tags.add(fields[0])
        self.valid_tags = frozenset(valid_tags)

        annotation_paths = sorted(annotations.glob("*.txt"), key=lambda p: p.name)
        if not annotation_paths:
            raise TraceContractError("no MIR annotation files found")
        labeled = set()
        for path in annotation_paths:
            labeled.update(value for value in _read_lines(path) if value)

        self.label_members: Dict[str, frozenset] = {}
        label_paths: List[Path] = []
        for label_name in MIR_LABEL_NAMES:
            path = require_file(annotations / f"{label_name}.txt", f"MIR {label_name} label")
            label_paths.append(path)
            self.label_members[label_name] = frozenset(
                value for value in _read_lines(path) if value
            )

        selected: List[
            Tuple[str, Tuple[str, ...], Tuple[str, ...], Tuple[int, ...]]
        ] = []
        tag_paths: List[Path] = []
        for image_id in labeled:
            tag_path = raw / "meta" / "tags" / f"tags{image_id}.txt"
            if not tag_path.is_file():
                continue
            raw_lines = _read_lines(tag_path)
            raw_tags = tuple(value for value in raw_lines if value)
            filtered_indices = tuple(
                index
                for index, value in enumerate(raw_lines)
                if value and value in self.valid_tags
            )
            filtered = tuple(raw_lines[index] for index in filtered_indices)
            if filtered:
                selected.append((image_id, raw_tags, filtered, filtered_indices))
                tag_paths.append(tag_path)
        selected.sort(key=lambda value: int(value[0]))
        self.rows = len(selected)
        validate_expected_rows(self.rows, expected_rows, self.dataset)
        self.samples = tuple(selected)

        self.image_paths: List[Path] = []
        self.tag_paths: List[Path] = []
        self.labels_hot: List[np.ndarray] = []
        for image_id, _, _, _ in self.samples:
            image_path = (raw / f"im{image_id}.jpg").resolve()
            if require_images and not image_path.is_file():
                raise TraceContractError(f"missing MIR raw image: {image_path}")
            self.image_paths.append(image_path)
            self.tag_paths.append(
                (raw / "meta" / "tags" / f"tags{image_id}.txt").resolve()
            )
            self.labels_hot.append(
                np.asarray(
                    [
                        int(image_id in self.label_members[label])
                        for label in MIR_LABEL_NAMES
                    ],
                    dtype=np.uint8,
                )
            )
        self.split = ContentHashSplit.build(
            self.dataset,
            [str(int(sample[0])) for sample in self.samples],
            query_rows=query_rows,
            train_rows=train_rows,
            seed=split_seed,
        )
        self.source_artifacts = tuple(
            [self.common_tags_path] + annotation_paths
        )

    def contract(self) -> Dict[str, Any]:
        contract = super().contract()
        contract.update(
            {
                "canonical_order": "ascending integer MIR image ID",
                "identity_chain": "OralData only",
                "process_data_role": "none; no legacy row recovery is attempted",
                "valid_tag_rule": "common_tags count >= 20",
                "text_prompt": "a photo of {tag}",
                "label_names": list(MIR_LABEL_NAMES),
            }
        )
        return contract

    def iter_rows(self, start: int = 0) -> Iterator[TraceRow]:
        if not 0 <= start <= self.rows:
            raise TraceContractError(f"invalid MIR start row {start}")
        for canonical_index in range(start, self.rows):
            image_id, raw_tags, filtered, filtered_indices = self.samples[canonical_index]
            canonical_source_id = str(int(image_id))
            prompts = tuple(f"a photo of {tag}" for tag in filtered)
            yield TraceRow(
                dataset=self.dataset,
                row_index=canonical_index,
                source_index=int(image_id) - 1,
                source_id=canonical_source_id,
                image_path=str(self.image_paths[canonical_index]),
                encoded_texts=prompts,
                raw_text={
                    "raw_tags": list(raw_tags),
                    "filtered_tags": list(filtered),
                    "filter_rule": "common_tags count >= 20",
                },
                label_hot=tuple(int(v) for v in self.labels_hot[canonical_index]),
                split=self.split.flags(canonical_index),
                metadata={
                    "mir_image_id": int(image_id),
                    "raw_image_id_token": image_id,
                    "canonical_index": canonical_index,
                    "canonical_identity_binding": "numeric_raw_image_id",
                    "label_names": list(MIR_LABEL_NAMES),
                    "raw_text_source": str(self.tag_paths[canonical_index]),
                    "label_source": {
                        "annotation_root": str(self.annotation_root.resolve()),
                        "positive_label_files": [
                            f"{label}.txt"
                            for label, value in zip(
                                MIR_LABEL_NAMES, self.labels_hot[canonical_index]
                            )
                            if int(value) == 1
                        ],
                    },
                },
                raw_text_locators=(
                    {
                        "path": str(self.tag_paths[canonical_index]),
                        "kind": "text_file_lines",
                        "locator": {
                            "mir_image_id": int(image_id),
                            "filtered_line_indices": list(filtered_indices),
                            "index_base": 0,
                        },
                        "content_sha256": sha256_json(
                            {
                                "raw_tags": list(raw_tags),
                                "filtered_tags": list(filtered),
                            }
                        ),
                    },
                ),
            )


def make_adapter(
    dataset: str,
    data_root: os.PathLike[str],
    expected_rows: Optional[int] = None,
    require_images: bool = True,
) -> DatasetAdapter:
    normalized = dataset.strip().lower().replace("-", "")
    if normalized in ("nus", "nuswide"):
        expected = 195_834 if expected_rows is None else expected_rows
        return NUSWideAdapter(data_root, expected, require_images)
    if normalized in ("coco", "mscoco"):
        expected = 122_218 if expected_rows is None else expected_rows
        return MSCOCOAdapter(data_root, expected, require_images)
    if normalized in ("mir", "mirflickr", "mirflickr25k"):
        expected = 20_015 if expected_rows is None else expected_rows
        return MIRFlickrCanonicalAdapter(data_root, expected, require_images)
    raise TraceContractError(f"unsupported trace dataset: {dataset}")
