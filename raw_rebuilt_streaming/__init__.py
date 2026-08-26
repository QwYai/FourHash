"""Bounded, process-isolated streaming evaluation for raw-rebuilt codes.

The package deliberately exposes three immutable boundaries:

* :func:`freeze_code_state` performs label-free encoding and stores only
  packed binary codes;
* :func:`freeze_rank_plan` freezes the complete Hamming evidence protocol;
* rank and metric workers communicate only through hash-bound distance
  bundles and acknowledgements.
"""

__all__ = [
    "CodeFreezeConfig",
    "CodeState",
    "FrozenRankPlan",
    "StreamingPlanConfig",
    "expected_tie_metrics_from_distances",
    "freeze_code_state",
    "freeze_rank_plan",
    "open_code_state",
    "open_rank_plan",
]
