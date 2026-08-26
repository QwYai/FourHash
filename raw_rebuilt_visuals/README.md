# Trace-backed qualitative evidence

This package turns canonical row IDs from a **frozen, label-free rank token**
into exact raw image/text/label evidence. It accepts only the sealed
`raw_rebuilt_v1` trace bundle. It never reads `ProcessData`, `ids.mat`, a legacy
feature file, or a nearest-feature identity match.

Before copying a selected image it runs both trace validators and the complete
materialized-runtime validator, resolves the row
through receipt-ordered JSONL manifests, and checks the raw `OralData` image
bytes against the row SHA-256. The evidence JSON retains the exact raw-text
value and locator, label vector, feature binding, row contract, rank token, and
source seal. This is the required input for qualitative retrieval figures.

The selection JSON has schema `raw_rebuilt_rank_visual_selection_v1` and must
contain `dataset`, `rank_token_sha256`, `source_seal_sha256`, nonempty `cases`,
and a `selection_sha256` over the canonical JSON body without that final field.
Each case contains one `query_row_id` and an ordered `candidate_row_ids` list.

```bash
python -m raw_rebuilt_visuals \
  --bundle /path/to/raw_rebuilt_trace/dataset/run \
  --runtime /path/to/materialized/raw_rebuilt_v1/dataset \
  --selection /path/to/frozen-rank-visual-selection.json \
  --output /path/to/new/evidence-directory \
  --process-data-root /path/to/Data/ProcessData
```
