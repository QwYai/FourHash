# FourHash / ShellGuard

This repository contains the code-only implementation of **ShellGuard**, a
deep semantic bridge for collision-aware cross-modal hashing. The public
tree intentionally excludes datasets, extracted features, checkpoints,
experiment outputs, audit receipts, and manuscript files.

## Method in one paragraph

ShellGuard trains two compact residual experts on paired 512-D CLIP image and
text features plus training labels. The primary expert defines global Hamming
shells. A separately trained, modality-normalized semantic expert averages five
neural posterior heads, predicts a nonempty concept set using one threshold
selected on training identities only, and compresses that set with deterministic
16-bit one-bit MinHash. The knowledge code is allowed to reorder candidates
only inside one equal primary-distance shell. Ranking uses the exact mixed-radix
distance

```text
D(q, x) = 17 * d_primary(q, x) + d_semantic(q, x),  0 <= d_semantic <= 16,
```

so no strict primary-shell comparison can be reversed. Dense posteriors are
transient at encoding time; serving stores only the primary code and the 16-bit
semantic sketch, with no float database cache.

## Repository map

- `visualization_trace/`: rebuilds image, text, label, split, and immutable row
  identity directly from raw `OralData`, then extracts separate CLIP ViT-B/32
  features.
- `visualization_feature_pipeline/`: independently validates every raw-source
  locator and one-to-one feature/label/identity binding.
- `raw_rebuilt_runtime/`: creates the sealed label-separated runtime.
- `raw_rebuilt_neural/`: compact primary/semantic networks, five-head posterior
  learning, training-only calibration, one-bit MinHash, and lexicographic rank
  freeze.
- `raw_rebuilt_baselines/` and `encoders/`: registered fixed-feature UCCH-F,
  DCMH-F-SemInit, and CIRH-F implementations.
- `raw_rebuilt_streaming/`: complete-gallery expected-tie evaluator with
  separate label-free rank and label-side metric processes.
- `raw_rebuilt_visuals/`: trace-backed qualitative evidence generation.
- `tools/`: frozen experiment, aggregation, ablation, audit, and visualization
  drivers.
- `src/rz_merge.py`: the earlier RZ-Merge serving component retained for
  backward compatibility.

## Environment

Python 3.9+ is supported by the synthetic test suite. A CUDA build of PyTorch
is recommended for full NUS-WIDE and MS COCO evaluation.

```bash
python -m pip install -r requirements.txt
```

The extractor uses the official OpenAI CLIP package and ViT-B/32 checkpoint.
Its exact module, checkpoint, preprocessing, and execution environment are
content-hashed into each extraction contract.

## Reproduction boundary

Keep raw and generated trees separate:

```text
Data/
  OralData/       # read-only raw images, text/annotations, labels
  ProcessData/    # never used as identity or feature authority
work/             # all newly extracted features and experiment artifacts
```

Each canonical sample contains one image vector, one text vector, one
multi-label vector, one immutable row ID, and locators for the exact raw image
and raw text. The three frozen datasets are MIRFlickr-25K (24 labels),
NUS-WIDE-TC21 (21 labels), and MS COCO (80 labels). Split seed `20260822` is
content-hash based; query and database are disjoint, and training is a database
subset.

Full raw-source validation is performed when a runtime seal is created. After
the label-free rank plan freezes that seal, metric workers reopen it in bounded
memory and rehash every scoring array and source shard without materializing
the complete raw-row manifest again.

Start with raw-data preflight and extraction:

```bash
python -m visualization_trace preflight \
  --dataset mirflickr --data-root /path/to/Data --limit 16

python -m visualization_trace extract \
  --dataset mirflickr --data-root /path/to/Data \
  --output-root /path/to/work/raw_clip512
```

Validate and materialize the sealed runtime:

```bash
python -m visualization_feature_pipeline.cli validate \
  --bundle /path/to/work/raw_clip512/mirflickr/clip_vit_b32_v1 \
  --process-data-root /path/to/Data/ProcessData

python -m raw_rebuilt_runtime materialize \
  --bundle /path/to/work/raw_clip512/mirflickr/clip_vit_b32_v1 \
  --output /path/to/work/runtime/mirflickr \
  --process-data-root /path/to/Data/ProcessData
```

The detailed primary/semantic training, calibration, rank-freeze, baseline,
and streaming-evaluation commands are documented in the README files inside
their respective packages. Run each information-boundary stage as a separate
process. In particular, query/database labels must not be opened until the
binary codes and rank plan are frozen.

The registered semantic-bridge pipeline is implemented by:

- `raw_rebuilt_neural/semantic_bridge.py` for threshold calibration, one-bit
  MinHash construction, and invariant mixed-radix distances;
- `tools/formal_semantic_bridge_streaming_eval.py` for label-separated formal
  evaluation;
- `tools/audit_semantic_bridge_formal.py` for receipt and invariant checks;
- `tools/dev_semantic_bridge_budget_ablation.py` for the training-only bit
  budget study; and
- the `analyze_semantic_bridge_*` and trace-materialization tools for post-hoc
  diagnostics that cannot feed back into a frozen rank plan.

The full preregistration freeze is an experiment artifact and is deliberately
not committed. Formal workers verify its registered hashes from
`raw_rebuilt_neural.ccde_contract`; set `SHELLGUARD_CCDE_FREEZE` to the local
`freeze.json` when running artifact-bound integration tests.

## Tests

The tests create synthetic temporary data and do not require the real datasets:

```bash
python -m pytest -q
```

For a fast core check:

```bash
python -m pytest -q \
  raw_rebuilt_runtime/tests \
  raw_rebuilt_neural/tests \
  raw_rebuilt_baselines/tests \
  raw_rebuilt_streaming/tests \
  visualization_trace/tests \
  visualization_feature_pipeline/tests
```

## Artifact policy

Only source code, documentation, and synthetic tests belong in this
repository. Paper sources/PDFs, raw data, feature arrays, model weights,
metrics, logs, and audit artifacts remain outside the Git repository.
