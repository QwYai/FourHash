"""Trace-backed qualitative evidence for ``raw_rebuilt_v1`` results."""

from .evidence import (
    EvidenceError,
    SelectionCase,
    collect_trace_rows,
    materialize_evidence,
    parse_selection_manifest,
)

__all__ = [
    "EvidenceError",
    "SelectionCase",
    "collect_trace_rows",
    "materialize_evidence",
    "parse_selection_manifest",
]
