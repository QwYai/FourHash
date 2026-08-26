"""Select a non-regressing code-length route from an indT anchor sweep.

Hash code length is known before inference, so a model family may safely route
16/32/64-bit requests to different predeclared checkpoints without increasing
per-request inference cost.  Selection is development-only: for each width, a
candidate must be no worse than the compact control in both directions for
mAP and NDCG@50.  If no candidate passes, that width falls back exactly to the
control checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_runtime.contract import atomic_write_json, sha256_file, sha256_json
from rz_csd_clip512 import BITS
from tools.dev_rzcsd_architecture_sweep import _delta_report


PRIMARY = ("map_expected_ties", "ndcg_at_50_expected_ties")
GRADED = "jndcg_at_50_expected_ties"


def _validated_document(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    claimed = document.pop("result_sha256")
    actual = sha256_json(document)
    if claimed != actual:
        raise RuntimeError(f"result hash mismatch: {path}")
    return {**document, "result_sha256": claimed}


def _width_report(
    candidate: dict[str, Any], control: dict[str, Any], bits: int
) -> dict[str, Any]:
    primary = []
    graded = []
    cells = []
    for direction in ("i2t", "t2i"):
        for metric in PRIMARY:
            delta = float(candidate[str(bits)][direction][metric]) - float(
                control[str(bits)][direction][metric]
            )
            primary.append(delta)
            cells.append(
                {"direction": direction, "metric": metric, "delta": delta}
            )
        graded.append(
            float(candidate[str(bits)][direction][GRADED])
            - float(control[str(bits)][direction][GRADED])
        )
    primary_array = np.asarray(primary, dtype=np.float64)
    graded_array = np.asarray(graded, dtype=np.float64)
    return {
        "eligible_all_four_primary_nonnegative": bool(
            np.all(primary_array >= 0.0)
        ),
        "negative_primary_cells": int(np.sum(primary_array < 0.0)),
        "mean_primary_delta": float(primary_array.mean()),
        "minimum_primary_delta": float(primary_array.min()),
        "mean_graded_jndcg_at_50_delta": float(graded_array.mean()),
        "minimum_graded_jndcg_at_50_delta": float(graded_array.min()),
        "cells": cells,
    }


def _width_key(record: dict[str, Any]) -> tuple[float, ...]:
    report = record["width_report"]
    return (
        1.0 if report["eligible_all_four_primary_nonnegative"] else 0.0,
        float(report["mean_primary_delta"]),
        float(report["mean_graded_jndcg_at_50_delta"]),
        float(report["minimum_primary_delta"]),
        -float(record["inference_parameter_count"]),
    )


def select_routes(sweep: dict[str, Any], sweep_root: Path) -> dict[str, Any]:
    if sweep["status"] != "DEVELOPMENT_ONLY_NOT_A_PAPER_CLAIM":
        raise RuntimeError("width routing accepts development sweeps only")
    if sweep["formal_query_or_database_labels_opened"] is not False:
        raise RuntimeError("width routing refuses a sweep that opened formal labels")
    records = sweep["records"]
    controls = [
        record
        for record in records
        if record["anchor_spec"]["name"] == "compact_unanchored_control"
    ]
    if len(controls) != 1:
        raise RuntimeError("the anchor sweep needs exactly one compact control")
    control = controls[0]
    assembled: dict[str, Any] = {}
    routes = {}
    for bits in BITS:
        candidates = []
        for record in records:
            checkpoint = sweep_root / f"{record['anchor_spec']['name']}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            candidate = {
                "candidate": record["anchor_spec"]["name"],
                "candidate_result_sha256": record["result_sha256"],
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": sha256_file(checkpoint),
                "inference_parameter_count": record["inference_parameter_count"],
                "width_report": _width_report(
                    record["evaluation"], control["evaluation"], bits
                ),
                "evaluation": record["evaluation"][str(bits)],
            }
            candidates.append(candidate)
        eligible = [
            candidate
            for candidate in candidates
            if candidate["width_report"]["eligible_all_four_primary_nonnegative"]
        ]
        if not eligible:
            raise AssertionError("compact control must always be width-eligible")
        selected = max(eligible, key=_width_key)
        routes[str(bits)] = {
            "selected_candidate": selected["candidate"],
            "selected_candidate_result_sha256": selected[
                "candidate_result_sha256"
            ],
            "selected_checkpoint": selected["checkpoint"],
            "selected_checkpoint_sha256": selected["checkpoint_sha256"],
            "selected_width_report": selected["width_report"],
            "eligible_candidates": [
                candidate["candidate"]
                for candidate in eligible
            ],
            "all_candidate_width_reports": {
                candidate["candidate"]: candidate["width_report"]
                for candidate in candidates
            },
        }
        assembled[str(bits)] = selected["evaluation"]
    aggregate = _delta_report(assembled, control["evaluation"])
    if not aggregate["all_twelve_nonnegative"]:
        raise AssertionError("assembled width routes violated the non-regression gate")
    return {
        "routes": routes,
        "assembled_evaluation": assembled,
        "assembled_delta_report": aggregate,
        "control_candidate_result_sha256": control["result_sha256"],
    }


def run(sweep_path: Path, output: Path) -> dict[str, Any]:
    sweep_path = sweep_path.resolve(strict=True)
    sweep = _validated_document(sweep_path)
    selected = select_routes(sweep, sweep_path.parent)
    body = {
        "schema": "raw_rebuilt_rzcsd_code_length_router_indt_v1",
        "status": "DEVELOPMENT_ONLY_NOT_A_PAPER_CLAIM",
        "dataset": sweep["dataset"],
        "source_seal_sha256": sweep["source_seal_sha256"],
        "fit_artifact_sha256": sweep["fit_artifact_sha256"],
        "formal_query_or_database_labels_opened": False,
        "source_sweep": sweep_path.name,
        "source_sweep_result_sha256": sweep["result_sha256"],
        "selection_rule": (
            "independently for each requested code width, require all four "
            "i2t/t2i mAP/NDCG50 deltas >=0 versus compact; then maximize mean "
            "primary delta, mean graded JNDCG50 delta, minimum primary delta, "
            "and prefer fewer inference parameters; otherwise exact fallback"
        ),
        **selected,
    }
    result = {**body, "result_sha256": sha256_json(body)}
    atomic_write_json(output, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.sweep, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "routes": {
                    bits: route["selected_candidate"]
                    for bits, route in result["routes"].items()
                },
                "result_sha256": result["result_sha256"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

