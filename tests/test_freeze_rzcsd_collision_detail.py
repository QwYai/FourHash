import json

from raw_rebuilt_runtime.contract import sha256_json
from tools.dev_rzcsd_detail_budget_sweep import DETAIL_BUDGETS
from tools.freeze_rzcsd_collision_detail import DATASETS, _summarize_cap, run


def _sweep(dataset: str) -> dict:
    candidates = {}
    for bits in (16, 32, 64):
        candidates[str(bits)] = {}
        for budget in (value for value in DETAIL_BUDGETS if value <= bits):
            # Only cap>=16 is globally eligible.  cap=32 deliberately has one
            # negative at 64 bits to verify smallest-eligible selection rather
            # than monotonic assumptions.
            primary = 0.01 if budget >= 16 else -0.01
            if bits == 64 and budget == 32 and dataset == "mirflickr":
                primary = -0.001
            candidates[str(bits)][str(budget)] = {
                "deltas": {
                    direction: {
                        "map_expected_ties": primary,
                        "ndcg_at_50_expected_ties": primary,
                        "jndcg_at_50_expected_ties": 0.02 if budget >= 16 else -0.02,
                    }
                    for direction in ("i2t", "t2i")
                }
            }
    body = {
        "schema": "raw_rebuilt_rzcsd_detail_budget_sweep_indt_v1",
        "status": "DEVELOPMENT_ONLY_NOT_A_PAPER_CLAIM",
        "dataset": dataset,
        "formal_query_or_database_labels_opened": False,
        "configuration_frozen_for_formal_evaluation": False,
        "candidate_budget_grid": list(DETAIL_BUDGETS),
        "source_seal_sha256": f"source-{dataset}",
        "fit_artifact_sha256": f"fit-{dataset}",
        "split": {"fit": 10, "query": 2, "database": 3},
        "split_hashes": {"fit": "a", "query": "b", "database": "c"},
        "candidates": candidates,
    }
    return {**body, "result_sha256": sha256_json(body)}


def test_summary_counts_all_cross_dataset_cells() -> None:
    sweeps = {dataset: _sweep(dataset) for dataset in DATASETS}
    summary = _summarize_cap(sweeps, 16)
    assert summary["primary_cells"] == 36
    assert summary["graded_cells"] == 18
    assert summary["eligible"] is True


def test_freeze_selects_smallest_globally_eligible_cap(tmp_path) -> None:
    paths = {}
    for dataset in DATASETS:
        path = tmp_path / f"{dataset}.json"
        path.write_text(json.dumps(_sweep(dataset)), encoding="utf-8")
        paths[dataset] = path
    output = tmp_path / "freeze.json"
    result = run(paths, output)
    assert result["frozen_architecture"]["global_detail_cap"] == 16
    assert result["selected_development_summary"]["negative_primary_cells"] == 0
    assert output.exists()
