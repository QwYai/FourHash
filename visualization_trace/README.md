# OralData-only visualization trace

This package extracts a separate qualitative-experiment feature bundle. It does
not read legacy `ProcessData` features, IDs, metadata mappings, or split files
to decide sample identity. It never writes under `OralData` or `ProcessData`.

Canonical identities are rebuilt from raw data:

- MIRFlickr: ascending numeric `im<ID>.jpg`, restricted to the reproducible
  20,015-member annotation/common-tag subset. Text is the mean of CLIP encodings
  of `a photo of {filtered_tag}`. Labels are rebuilt from the 24 annotation files.
- NUS-WIDE: ascending zero-based raw `clean_id`, mapped through official
  `ImageList/Imagelist.txt`. Model text is exactly the active AllTags1k tags, or
  the literal `a generic photo` for an empty top-1k row. Full `All_Tags.txt`
  tokens are retained as raw metadata and are explicitly marked not encoded.
  Multi-label supervision is rebuilt from `labels.nuswide-tc21.mat` and is
  fail-closed at exactly 21 dimensions.
- MSCOCO: ascending official image ID among images having both captions and
  instance categories. Every caption is encoded and averaged; 80-hot labels are
  rebuilt from official instance JSON.

## Frozen split

All datasets use `kbs-content-hash-split-v1`, seed `20260822`. SHA-256 keys are
domain-separated by dataset, role (`query` or `train`), and canonical source ID.
No random-number library and no seed search is involved. Query/database are
disjoint and cover all rows; train is a database subset. Sizes are:

| dataset | Q | T | D |
|---|---:|---:|---:|
| MIRFlickr | 2,243 | 5,000 | 17,772 |
| NUS-WIDE | 2,085 | 21,000 | 193,749 |
| MSCOCO | 5,000 | 10,500 | 117,218 |

## CLI

```text
python -m visualization_trace preflight \
  --dataset mirflickr \
  --data-root /path/to/Data \
  --limit 16

python -m visualization_trace extract \
  --dataset mirflickr \
  --data-root /path/to/Data \
  --output-root /path/to/work/visualization_trace
```

`preflight` does not load CLIP. It validates every canonical row, every raw
image path, model-text cardinality, label dimension, and the exact fixed split;
`--limit` only controls how many evenly spaced example records are printed.

The output contains immutable `contract.json`, `canonical_split.npz/json`,
aligned `shards/*.npz`, `manifests/*.jsonl`, chained receipts, and a final
`complete.json`. Resume verifies every existing digest before encoding the next
row. Each manifest row binds the feature pair to the raw image hash, exact model
texts, raw text metadata, multi-hot label, canonical source ID, and split flags.
The extraction contract also seals the actual ViT-B/32 checkpoint SHA-256,
checkpoint byte size/source URL, CLIP module hash, Torch/Pillow/NumPy versions,
preprocessing representation, tokenizer rule, and this package's code digest.
The feature contract deliberately stores raw, **unnormalized** CLIP image and
per-text outputs. Multi-text rows use a float64 arithmetic mean and are cast to
float32 without normalization before or after the mean. Any later experiment
must declare and apply one identical normalization policy to every compared
method.
