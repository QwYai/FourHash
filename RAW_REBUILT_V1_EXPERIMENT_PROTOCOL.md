# `raw_rebuilt_v1` experiment protocol

Status: frozen identity and feature-extraction protocol, 2026-08-22.

This protocol supersedes every experiment that inferred identity or row order
from `ProcessData`, legacy `ids.mat`, or a previously extracted CLIP feature
file. Those assets may be used only as diagnostics and never as a source of
features, labels, splits, ranks, metrics, tables, figures, or manuscript
claims.

## 1. Canonical sample contract

Every canonical row binds all of the following under one `S64` row ID:

- the source dataset's canonical identity;
- the exact raw image path and SHA-256;
- the exact raw text construction and replayable source locator;
- the ordered multi-label vector and label-axis definition;
- one raw OpenAI CLIP ViT-B/32 image vector and one raw text vector;
- deterministic `indQ`, `indT`, and `indD` membership;
- shard, manifest, receipt, source-inventory, and completion hashes.

Feature extraction reads only `/path/to/Data/OralData`. Outputs are
written below
`/path/to/work/raw_rebuilt_clip512_v1_sealed` and
are completely separate from both `OralData` and `ProcessData`.

## 2. Frozen extraction

- Encoder: OpenAI CLIP `ViT-B/32`.
- Checkpoint SHA-256:
  `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.
- Extractor `visualization_trace/extraction.py` SHA-256:
  `6b32825576caf112eb24e5acb5f80b99ce07245c98fbf54921fb1d84c7bc15d3`.
- Stored image and per-text vectors: unnormalized `float32` encoder outputs.
- Multi-text row: unnormalized per-text outputs, arithmetic mean accumulated
  in `float64`, stored as unnormalized `float32`.
- Text encoding has no embedding cache or string deduplication. This is
  intentional because encoder outputs may depend on batch context.
- Shard size: 1,024 canonical rows; image batch: 128; flattened text batch:
  1,024. Shard boundaries and batching are part of the sealed contract.
- The contract removes process-local addresses from the transform `repr` and
  records model dtype, GPU identity, driver, CUDA/cuDNN build, cuBLAS workspace,
  TF32 switches, deterministic-algorithm switches, and PyTorch build hash.

## 3. Dataset identities and split

The split algorithm is `kbs-content-hash-split-v1` with seed `20260822`.
SHA-256 is evaluated over dataset, role, and canonical source ID. `indQ` and
`indD` are disjoint and cover every canonical row; `indT` is a subset of
`indD`. No random retry, class-aware seed search, or metric-selected split is
allowed.

| Dataset | Canonical rows | Classes | Q | T | D |
|---|---:|---:|---:|---:|---:|
| MIRFlickr | 20,015 | 24 | 2,243 | 5,000 | 17,772 |
| NUS-WIDE-TC21 | 195,834 | 21 | 2,085 | 21,000 | 193,749 |
| MS COCO | 122,218 | 80 | 5,000 | 10,500 | 117,218 |

MIRFlickr uses ascending numeric image ID, the 24 official labels, and valid
common tags with count at least 20. Each valid tag is encoded as
`a photo of {tag}` in source order and then averaged.

NUS-WIDE uses the ascending, unique, zero-based `clean_id.nuswide.tc21.mat`
selection and the 21-column `labels.nuswide-tc21.mat`; 81-column legacy labels
are forbidden. The source identity is
`raw-index:{index}|photo-id:{photo_id}`. Active top-1,000 tags are encoded in
`TagList` order; rows with no such tag use the exact text `a generic photo`.

MS COCO uses ascending official image ID, the official 80-category ID/name
axis, and every caption in JSON annotation order. All captions for an image
are encoded and averaged.

## 4. Runtime and label boundary

`raw_rebuilt_runtime` is the only supported handoff to training and evaluation.
It admits a bundle only after the trace validator and the independent raw
provenance validator both pass. Materialization streams the sealed shards into
separate read-only NPY arrays and emits its own chained receipts and completion
seal.

Training may decode labels only for `indT`. The ranking process receives
features, row IDs, and Q/T/D indices but no labels. Only after it writes a
content-bound `rank_state_frozen` contract may a separate metric boundary open
the aligned `indQ` and `indD` labels. Every model, ranking, metric, table, and
figure receipt must carry the runtime `source_seal_sha256`.

## 5. Method hypothesis

The proposed method is a neural cross-modal hashing framework with a strict
four-stage retrieval hierarchy:

1. **Geometry proposes:** raw Hamming distance defines the candidate shell.
2. **RZ aligns:** reference-compiled standardization compares independently
   learned image and text indexes only where raw geometry is ambiguous.
3. **Relevance confirms:** a train-only multi-label neural posterior decides
   whether a candidate is semantically relevant; labels are not inference
   inputs.
4. **Grade orders:** soft-Jaccard posterior evidence orders only candidates
   that remain ambiguous after relevance evidence.

The causal ablation must remove or reverse exactly one stage at a time:
without the raw guard, without RZ, Jaccard-first, without soft Jaccard,
binary-only training, no posterior-head ensemble, no expert routing, and a
capacity-matched MLP control. The intended unique advantage is a large gain in
collision-heavy short-code graded retrieval and mixed-gallery robustness while
preserving standard mAP and the resolved suffix.

## 6. Evidence and reporting rules

- All baselines and the proposed method use these exact features, identities,
  splits, label axes, metric code, and rank-freeze boundary.
- Literature-reported numbers are shown only in a separately labelled context
  table; they are never mixed with rerun values.
- Main neural results use at least three training seeds and report mean and
  standard deviation. Small claimed advantages require paired uncertainty or
  a predeclared equivalence/significance rule.
- Main scores use three decimals unless uncertainty requires four. Percentage
  point deltas and interval endpoints use two decimals. Internal hashes and
  numerical residual checks retain full precision outside performance tables.
- SOTA language is prohibited until the same fixed model/profile is strictly
  supported on all three datasets under this protocol. A failed metric remains
  visible; it is not replaced, rounded away, or selected by dataset.
- Quantitative figures, qualitative retrieval cases, and embedding plots must
  resolve every displayed row ID back to its sealed raw image, exact raw text,
  and labels.
