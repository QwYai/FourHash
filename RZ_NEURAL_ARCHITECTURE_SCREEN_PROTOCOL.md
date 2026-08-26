# RZ neural architecture screen (development only)

Date: 2026-08-20

## Purpose

This screen tests whether candidate interaction models can improve the RZ-Shell
fallback consistently before any full-gallery diagnostic is run.  It is model
development, not a confirmation experiment and not evidence of SOTA.

## Fixed data boundary

- Supervision is restricted to the deterministic PromptHash dev400 labels.
- Four roles are disjoint within each outer rotation: train query, train
  candidate, evaluation query, and evaluation candidate.
- The live training query is excluded from every reference/query bank.
- q600 and q100 split definitions and metric artifacts are not inputs.
- The loader may materialize complete MAT label arrays; only dev400 labels may
  enter targets, losses, scores, model selection, or reported metrics.

## Shared ranking contract

- Every candidate uses the V6 fixed-K rank scale `clip(rank,31)/31`.
- The learned region is the tie-complete RZ top-32 window.
- Outside that region the integer RZ-Shell key is unchanged.
- Exact neural ties fall back to the RZ-Shell key.
- One model is shared across all datasets, bit widths, query directions, gallery
  mixtures, and assignment seeds.  Per-cell model or architecture selection is
  forbidden.
- Each candidate architecture has at most 100,000 trainable parameters.

## Frozen architecture candidates

1. V6 residual MLP (existing control).
2. DeepSets contextual ranker: per-item encoder, permutation-invariant mean/max
   context, and a residual item scorer.
3. Induced Set Transformer: a fixed small inducing set and linear-in-list-size
   attention; full quadratic candidate self-attention is forbidden.
This initial screen ends after the two new candidate results are inspected.
DeepSets and the balanced-loss induced Set Transformer both failed its gate.  A
later user-authorized dual-head hypothesis is not retroactively added to this
screen; it is governed by the separately frozen
`RZ_DUALHEAD_SET_RANKER_PROTOCOL.md`.  Its gate is query-level and global rather
than a dataset/cell architecture switch.

## Development endpoint gate and selection

For each architecture, report mAP, J-NDCG@10, J-NDCG@100, and balanced utility
for all three datasets and all three bit widths.  The paired comparator is
RZ-Shell-CSLS under the identical query-condition rows.

An architecture is eligible only when:

1. all 36 point differences versus RZ-Shell are nonnegative; and
2. the mean of the nine utility differences is positive.

If more than one architecture is eligible, select globally by the largest
minimum endpoint difference, then by the largest mean utility difference, then
by the smaller parameter count.  No dataset-specific or metric-specific switch
is allowed.

Only the single selected architecture may proceed to a scale-matched official-r
gallery development diagnostic.  That diagnostic must recompute all methods on
the same rows and may not splice historical q200/q600 values.

## Required controls

- permutation equivariance for set models;
- active-feature and score invariance when irrelevant tail candidates are added;
- exact zero-residual reproduction of the RZ-Shell fallback;
- complete boundary ties, active-count distribution, parameter count, peak
  memory, and runtime;
- a same-input linear scorer and a capacity-matched unstructured MLP before any
  neural-architecture novelty claim.

## Claim boundary

Passing this screen supports only a development decision.  A formal all-metric
SOTA claim additionally requires a new-identity external lockbox and simultaneous
one-sided confidence bounds above the frozen per-endpoint baseline frontier.
