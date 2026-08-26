# raw_rebuilt_neural

This package is the neural experiment boundary for the three sealed
`raw_rebuilt_v1` datasets. It uses the existing `RZCSD512` neural encoder but
does not reuse any legacy feature identity, label axis, split, model result, or
prepared experiment artifact.

No score in this package is a paper claim. A SOTA statement is permitted only
after all registered seeds and same-protocol baselines have completed.

## Four independent processes

Run each command as a separate process. There is intentionally no combined
"run everything" command.

1. Admission verifies the complete raw-rebuilt runtime and emits a
   content-addressed training artifact containing only `indT` image features,
   text features, labels, canonical row IDs, canonical split indices, and
   collision-checked identity digests.

   ```bash
   python -m raw_rebuilt_neural prepare-fit \
     --runtime /path/to/work/runtime/raw_rebuilt_v1/mirflickr \
     --output-parent /path/to/work/neural/fit
   ```

2. Training accepts the fit artifact only. Its command line has no runtime or
   full-label argument. Epoch permutations and dropout seeds are counter-based,
   and every checkpoint receipt binds the source seal, indT split hash, fit
   artifact, model config, training config, and code inventory. Resuming a
   different binding fails closed.

   The frozen curriculum uses 40 base epochs followed by five epochs at
   `5e-5`.  The final stage adds a width-specific label decoder
   (`code_bce_weight=0.035`) and balanced label-set-Jaccard regression
   (`graded_weight=0.07`).  These small decoder heads are training-only and are
   excluded from serving.  The schedule was selected on a deterministic
   NUS-WIDE indT-internal split without opening any formal query/database
   labels, then frozen unchanged for all datasets and registered seeds.

   ```bash
   python -m raw_rebuilt_neural train-three-seeds \
     --fit /root/.../fit-0123456789abcdef \
     --output-parent /root/.../neural/train/mirflickr
   ```

   The frozen seeds are `20260822`, `20260823`, and `20260824`. A single seed
   can be run or resumed with `train`.

3. Rank freezing uses `load_label_free_rank_inputs`; no labels are returned to
   the worker. Image-to-text uses image queries and text database codes, while
   text-to-image uses text queries and image database codes. The default is
   packed bipolar Hamming geometry for 16/32/64 bits. `rz_csd_local` is an
   explicit, bounded ablation: it closes the raw Hamming boundary shell, maps
   it to the paired-indT reference-Z coordinate, and applies relevance-first
   then soft-Jaccard evidence without crossing a raw shell. It aborts if the
   tie-closed shell exceeds the preregistered cap.

   ```bash
   python -m raw_rebuilt_neural freeze-ranks \
     --runtime /root/.../runtime/raw_rebuilt_v1/mirflickr \
     --checkpoint /root/.../seed-20260822/checkpoints/epoch-0039.pt \
     --output-parent /root/.../neural/ranks/mirflickr/seed-20260822 \
     --bits 16,32,64 --directions i2t,t2i --modes hamming
   ```

   Rank matrices are query-chunked, receipt-chained, and resumable. Exact
   evidence groups are stored alongside a deterministic canonical permutation;
   canonical identity is never silently awarded retrieval credit inside ties.

4. Evaluation first replays every rank receipt, then and only then calls
   `load_metric_labels`. The primary mAP, precision, recall, and binary nDCG
   integrate exactly over uniform permutations inside evidence ties. The
   canonical-storage mAP is reported as a diagnostic, not the primary score.

   ```bash
   python -m raw_rebuilt_neural evaluate \
     --runtime /root/.../runtime/raw_rebuilt_v1/mirflickr \
     --rank-root /root/.../rank-... \
     --output-parent /root/.../neural/metrics/mirflickr/seed-20260822
   ```

## Frozen geometry and leakage rules

- MIRFlickr: 24 labels; NUS-WIDE: TC21 only; MS COCO: 80 labels.
- Features are exactly 512-D image/text vectors from the sealed extraction.
- The trainer receives only `indT`; it cannot accept query/database labels.
- Rank workers receive full features, splits, row IDs, and source seal, but no
  labels. Metric workers are a separate invocation after `rank_state_frozen`.
- Zero hash logits map to `+1`, matching `RZCSD512`.
- Every artifact binds the raw runtime source seal and exact code/config hashes.
- Outputs under protected raw/prepared input trees and legacy MAT artifacts are
  rejected.

Run the synthetic boundary, poisoning, resume, rank-direction, and expected-tie
tests with:

```bash
python -m pytest -q raw_rebuilt_neural/tests
```
