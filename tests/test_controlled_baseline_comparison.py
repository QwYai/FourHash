from __future__ import annotations

import json
from pathlib import Path

from raw_rebuilt_runtime.contract import sha256_json
from tools import build_controlled_baseline_comparison as comparison


def _write_aggregate(path: Path, body: dict) -> None:
    value = {**body, "aggregate_sha256": sha256_json(body)}
    path.write_text(json.dumps(value), encoding="utf-8")


def _formal(path: Path, datasets: tuple[str, ...]) -> None:
    rows = []
    for dataset in datasets:
        for bits in comparison.BITS:
            for direction in comparison.DIRECTIONS:
                rows.append(
                    {
                        "dataset": dataset,
                        "bits": bits,
                        "direction": direction,
                        "primary_map": 0.70,
                        "primary_ndcg50": 0.71,
                        "primary_jndcg50": 0.72,
                        "ccde_map": 0.80,
                        "ccde_ndcg50": 0.81,
                        "ccde_jndcg50": 0.82,
                    }
                )
    _write_aggregate(
        path,
        {
            "schema": comparison.FORMAL_SCHEMA,
            "status": "VERIFIED",
            "rows": rows,
        },
    )


def _baseline(path: Path, dataset: str) -> None:
    rows = []
    for method_index, method in enumerate(comparison.BASELINE_METHODS):
        for bits in comparison.BITS:
            for direction in comparison.DIRECTIONS:
                score = 0.50 + 0.01 * method_index
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "bits": bits,
                        "direction": direction,
                        "map": score,
                        "binary_ndcg_at_50": score,
                        "j_ndcg_at_50": score,
                        "metric_result_sha256": str(method_index) * 64,
                    }
                )
    _write_aggregate(
        path,
        {
            "schema": comparison.BASELINE_SCHEMA,
            "status": "VERIFIED",
            "dataset": dataset,
            "seeds": [20260822],
            "deep_verified_cells": 12,
            "rows": rows,
        },
    )


def test_multi_dataset_comparison_and_latex(tmp_path: Path) -> None:
    datasets = ("mirflickr", "nuswide")
    formal = tmp_path / "formal.json"
    _formal(formal, datasets)
    baselines = {}
    for dataset in datasets:
        path = tmp_path / f"{dataset}.json"
        _baseline(path, dataset)
        baselines[dataset] = path
    output_json = tmp_path / "comparison.json"
    output_csv = tmp_path / "comparison.csv"
    output_tex = tmp_path / "comparison.tex"
    result = comparison.build_comparison(
        formal_path=formal,
        baseline_paths=baselines,
        json_output=output_json,
        csv_output=output_csv,
        tex_output=output_tex,
    )
    assert result["datasets"] == list(datasets)
    assert len(result["rows"]) == 72
    assert result["strict_best_cells"] == result["comparison_cells"] == 36
    latex = output_tex.read_text(encoding="utf-8")
    assert "MIRFlickr-25K" in latex
    assert "NUS-WIDE-TC21" in latex
    assert "\\textbf{0.800}" in latex
