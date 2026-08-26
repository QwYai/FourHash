# raw_rebuilt_v1 fixed-feature baselines

This package is the only supported entry point for the controlled UCCH-F,
DCMH-F-SemInit, and CIRH-F comparison runs on the newly extracted CLIP512
features.  It deliberately does not expose a MAT/ProcessData loader or a
legacy command-line interface.

## Temporal boundary

1. Create/open the shared content-addressed `raw_rebuilt_neural.FitArtifact`.
   It contains exactly `indT`: paired float32 `[T,512]` features, the training
   multi-label rows, canonical row IDs/indices, and the raw-source seal.
2. Load the full dataset through
   `raw_rebuilt_runtime.load_label_free_rank_inputs`.  Convert it with
   `label_free_inputs_from_runtime(dataset, rank)`.  This object contains no
   label field.
3. Call `train_baseline(fit, label_free, config, output_parent)`.  Only the fit
   arrays reach an optimizer.  The full input contributes row/split hashes to
   the final checkpoint; its Q/D features are not passed to training.
4. Call `encode_label_free(checkpoint, label_free)`, then optionally
   `write_code_artifact`.  The returned rank contract has
   `labels_loaded_during_freeze=false` and can be handed to the separate metric
   label gate.

Every checkpoint and code receipt binds the raw source seal, fit artifact,
full and train row-ID hashes, exact `indT/indQ/indD` hashes, combined split
hash, wrapper/core source inventory, method configuration, bit width, and
seed.  Loading fails if any of those or the current code changes.

Only 16, 32, and 64 bits are admitted.  Feature width is exactly 512.
NUS-WIDE is exactly TC21; 81-label inputs fail before training.  The registered
paper seeds are `20260822`, `20260823`, and `20260824`.  PyTorch deterministic
algorithms, deterministic cuDNN, fixed cuBLAS workspace, disabled TF32, and
highest float32 matmul precision are enabled before model construction.
Both indT training features and full-dataset encoding features are copied into
owned, writable, C-contiguous float32 buffers before any retained core can call
`torch.from_numpy`; a read-only runtime memmap is never wrapped directly.

## What is honestly reused

- **UCCH-F** calls only `train_ucch_f` and `encode_all` from
  `encoders/ucch_feature.py`: feature-mode MLPs, momentum contrastive memory,
  and cross-modal ranking.  It is an unsupervised controlled adaptation, not
  the authors' end-to-end reproduction.
- **DCMH-F-SemInit** calls only `train_dcmh_f` and `encode_all` from
  `encoders/dcmh_feature.py`.  It is the one supervised baseline and receives
  only fit-artifact labels.  Its documented train-label semantic warm start is
  part of the reporting name; it must not be reported as official DCMH.
- **CIRH-F** calls only `train_cirh_f` and `encode_all` from
  `encoders/cirh_feature.py`: the collaborated train graph, reconstruction,
  mixing, and separate image/text hash networks.  It is a controlled
  adaptation.  The retained train graph is quadratic in `T`; the wrapper
  records the minimum one-matrix byte cost and does not replace it with an
  unreported approximation.

The old loader functions, dataset aliases, exporters, held-out gates, CLIs,
and every existing prepared result are outside this package and are never
called.  No result produced by an old feature/order contract is valid evidence
for `raw_rebuilt_v1`.

Minimal orchestration:

```python
from raw_rebuilt_neural import open_fit_artifact
from raw_rebuilt_runtime import load_label_free_rank_inputs
from raw_rebuilt_baselines import (
    BaselineRunConfig,
    encode_label_free,
    label_free_inputs_from_runtime,
    train_baseline,
)

fit = open_fit_artifact(fit_dir)
rank = load_label_free_rank_inputs(runtime_dir)
label_free = label_free_inputs_from_runtime(fit.dataset, rank)
checkpoint = train_baseline(
    fit,
    label_free,
    BaselineRunConfig(method="ucch-f", bits=32, seed=20260822),
    run_root,
)
codes = encode_label_free(checkpoint, label_free)
```

Run boundary, tamper, dispatch, and deterministic synthetic tests with:

```bash
python -m pytest -q raw_rebuilt_baselines/tests
```

Server orchestration uses the same boundary through the package CLI:

```bash
python -m raw_rebuilt_baselines train \
  --fit-artifact FIT_ROOT --runtime RUNTIME_ROOT \
  --method dcmh-f-seminit --bits 32 --seed 20260822 \
  --output CHECKPOINT_PARENT

python -m raw_rebuilt_baselines encode \
  --checkpoint CHECKPOINT_ROOT --runtime RUNTIME_ROOT \
  --output CODE_PARENT
```

Method-specific changes are accepted only as one JSON object through
`--overrides-json`; unknown fields and attempts to override bits, seed, or device
fail before training.

For a registered multi-cell server run, use
`run_raw_rebuilt_baseline_sweep.py` from the project root.  The driver accepts
only the three declared datasets, methods, bit widths, and seeds; resolves
exactly one sealed fit artifact per dataset; resumes content-addressed
checkpoints/codes; verifies both output manifests; appends JSONL audit events;
and stops before crossing a configurable free-space floor.  Its registered
device string is `auto`, which resolves to CUDA on the experiment server while
avoiding a duplicate run contract for an explicit but equivalent `cuda`
setting.
