# raw_rebuilt_v1 runtime bridge

This package is the only supported handoff from a sealed
`visualization_trace` bundle to the CLIP512, V2, and 24-method experiment
runners. It never reads or infers identity from `ProcessData`.

## Admission and materialization

Every invocation first requires both independent gates to pass:

1. `visualization_trace.extraction.verify_trace_bundle` validates the
   contract, source receipts, per-row feature hashes, final chain, completion
   marker, and the exact `indQ/indT/indD/row_ids` split ledger.
2. `visualization_feature_pipeline.validate_bundle` independently validates
   `raw_rebuilt_v1` OralData identity, raw image/text/label locators, TC21 for
   NUS-WIDE, fixed split counts/seed, canonical order, and the complete source
   inventory.

If either check fails, no runtime contract is written. There is no legacy MAT,
`ids.mat`, `ProcessData`, 81-hot NUS-WIDE, inferred mapping, or unverified
fallback path.

From `mixed_gallery`:

```bash
python -m raw_rebuilt_runtime materialize \
  --bundle /path/to/work/raw_rebuilt_clip512_v1_sealed/mirflickr/clip_vit_b32_raw_rebuilt_v1 \
  --output /path/to/work/runtime/raw_rebuilt_v1/mirflickr \
  --process-data-root /path/to/Data/ProcessData
```

The sealed extraction run binds `visualization_trace/extraction.py` as
`6b32825576caf112eb24e5acb5f80b99ce07245c98fbf54921fb1d84c7bc15d3`.
Each dataset contract must independently carry that exact inventory entry;
the runtime still relies on the full contract and receipt chain rather than
trusting this documentation value alone.

Materialization writes independent, read-only-on-consumption NPY arrays:

```text
arrays/image_features_clip512.npy  float32 [N,512]
arrays/text_features_clip512.npy   float32 [N,512]
arrays/labels.npy                  uint8   [N,C]  (NUS-WIDE C=21)
arrays/row_ids.npy                 S64     [N]
arrays/indQ.npy                    int64
arrays/indT.npy                    int64
arrays/indD.npy                    int64
```

Source shards are copied one at a time into NPY memmaps. Each committed range
has a chained receipt binding the exact source receipt, NPZ/manifest hashes,
and decoded image/text/label/row-ID slice hashes. `--max-new-parts K` may be
used for a bounded checkpoint; the next identical command revalidates all
committed slices against the source and resumes. A completed runtime has an
exact inventory, manifest, and completion seal. Missing, reordered, poisoned,
or extra files fail closed.

Verify before a run:

```bash
python -m raw_rebuilt_runtime verify \
  --runtime /path/to/work/runtime/raw_rebuilt_v1/mirflickr \
  --process-data-root /path/to/Data/ProcessData
```

## Minimal runner integration

Do not point existing runners at a legacy dataset directory. Replace their
prepared-MAT loading boundary with these calls:

```python
from raw_rebuilt_runtime import (
    load_indt_training_inputs,
    load_label_free_rank_inputs,
    load_metric_labels,
)

fit = load_indt_training_inputs(runtime_dir)
rank = load_label_free_rank_inputs(runtime_dir)  # intentionally has no labels

# Only after the rank order is frozen:
metric = load_metric_labels(runtime_dir, rank_contract=rank_contract)
```

The returned field names match the current CLIP512 runtime concepts:
`image`, `text`, `identity_ids`, `train_idx`, `query_idx`, `database_idx`,
`query`, and `database`. Bind `source_seal_sha256` into every V2 fold/model,
rank token, 24-method cell, and metric ledger. This is the minimal patch point;
the model/ranking implementations themselves do not need broad changes.

`load_label_free_rank_inputs` uses a dedicated temporal gate: it verifies and
opens only image/text, split, and row-ID content. It reads the label NPY header
and size but never decodes or hashes label rows, and returns
`labels_loaded_during_freeze=False`. Full dual source verification already ran
during materialization; `load_metric_labels` repeats full verification after
the rank state is frozen and rejects any intervening label poison before a
metric is computed.

Run the synthetic contract and poisoning suite with:

```bash
python -m pytest -q raw_rebuilt_runtime/tests
```

The synthetic-only four-row registry is inaccessible from the CLI and cannot
be used as a real experiment dataset.
