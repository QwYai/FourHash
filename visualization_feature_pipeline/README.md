# Raw-rebuilt visualization provenance validator

This directory is an independent, fail-closed validator for visualization
features rebuilt from `Data/OralData`. The primary accepted input is the bundle
emitted by the sibling `visualization_trace` extractor:

```text
<run>/
  contract.json
  canonical_split.npz
  canonical_split.json
  shards/part-*.npz
  manifests/part-*.jsonl
  receipts/part-*.json
  complete.json
```

It does not import the training/evaluation runner, compute retrieval metrics,
select a model, search a split seed, or use labels for model choice. The output
root must be separate from both `OralData` and `ProcessData`.

## Authority and hard boundary

`raw_rebuilt_v1` is the final authority. Sample identity, image, text, labels,
and split assignment are rebuilt from raw `OralData` locators. `ProcessData` is
never required or opened and may not supply feature rows, IDs, metadata, labels,
or indices. `--process-data-root` is optional and is used only as an additional
forbidden-path boundary.

The validator independently enforces:

1. Every row has one immutable `canonical_row_id` derived from the exact UTF-8
   `(dataset, source_id)` identity. Manifest rows, split `row_ids`, and
   concatenated shard `row_ids` must be byte-identical in one canonical order.
2. Image features, text features, and binary multi-label vectors have exactly
   the same row count and order. Missing modalities, duplicate rows, shard
   gaps/overlaps, and reorderings fail closed.
3. Every raw image and raw text source is an absolute file below `OralData`,
   bound by byte count and SHA-256. The complete sorted dependency set is
   rebound to the contract's Merkle summary.
4. Raw text locators are executed, not merely inspected. The audited kinds are
   NUS `mat_row`, `text_line_selection`, and `text_line`; COCO
   `json_annotation`; and MIR `text_file_lines`. JSON/MAT/text sources are
   parsed once per path and cached, so full NUS/COCO validation is linear rather
   than repeatedly loading shared files.
5. Multi-text order and aggregation inputs are preserved. NUS empty top-1k
   text may use only its explicit `a generic photo` fallback; the raw empty state
   remains recorded and verified.
6. Labels are independently replayed from raw authorities: MIR annotation
   membership, COCO official instance JSON/category axis, or NUS TC21 MAT rows.
   They must equal both the row record and label shard.
7. The only split is `kbs-content-hash-split-v1`, seed `20260822`, using exact
   source-ID UTF-8 bytes with separate query/train hash domains. No seed search
   or legacy prepared-index authority is accepted. `Q` and `D` are disjoint and
   cover all rows; `T` is a subset of `D`.
8. Contract, split, receipts, manifests, NPZ contents, row feature digests, and
   the complete on-disk file inventory are all reverified.
9. Same-size source poison, same-shape feature poison, unbound files, unknown
   locators, and changed labels all fail closed.

Ordinary cosine/nearest-neighbour/label-signature identity matching is
forbidden. MIRFlickr is rebuilt directly from raw numeric image IDs, tag files,
and annotation membership; no legacy feature-to-row recovery is an authority.

## Frozen raw-rebuilt registry

| dataset | rows | indQ | indT | indD | labels |
|---|---:|---:|---:|---:|---:|
| `mirflickr` | 20,015 | 2,243 | 5,000 | 17,772 | 24 |
| `nuswide` | 195,834 | 2,085 | 21,000 | 193,749 | 21 |
| `mscoco` | 122,218 | 5,000 | 10,500 | 117,218 | 80 |

NUS-WIDE is locked to
`OralData/NUS-WIDE/labels.nuswide-tc21.mat` (`labels`, `269648 x 21`) and
`clean_id.nuswide.tc21.mat` (195,834 sorted unique zero-based raw indices).
The 81-hot `labels.nuswide.mat` is rejected.

## Validate

From `mixed_gallery`:

```powershell
F:\python\Python39\python.exe -m visualization_feature_pipeline.cli validate `
  --bundle D:\path\to\visualization_trace\nuswide\run-name
```

Optional extra boundary:

```powershell
F:\python\Python39\python.exe -m visualization_feature_pipeline.cli validate `
  --bundle D:\path\to\visualization_trace\nuswide\run-name `
  --process-data-root D:\path\to\Data\ProcessData
```

A pass emits JSON with status, dataset, row/shard/source counts, bundle ID, and
the completed checks. A contract failure returns exit code 2.

The package also retains its standalone canonical JSONL/NPZ/Parquet contract
and schemas for independently authored bundles. Direct `visualization_trace`
bundles are detected by `contract.json` plus `complete.json` and validated
against the producer's actual schema.

## Synthetic regression

All tests use temporary synthetic raw files. They do not open real datasets or
run a real model:

```powershell
F:\python\Python39\python.exe -m pytest -q `
  visualization_trace\tests\test_trace.py `
  visualization_feature_pipeline\tests\test_validator.py
```

The suite includes a producer-to-validator end-to-end bundle and direct trace
poison cases for raw text, split row IDs, and a missing text modality.
