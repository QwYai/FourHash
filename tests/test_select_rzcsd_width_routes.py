from pathlib import Path

from tools.select_rzcsd_width_routes import select_routes


def _evaluation(value: float) -> dict:
    return {
        str(bits): {
            direction: {
                "map_expected_ties": value,
                "ndcg_at_50_expected_ties": value,
                "jndcg_at_50_expected_ties": value,
            }
            for direction in ("i2t", "t2i")
        }
        for bits in (16, 32, 64)
    }


def test_width_router_uses_candidate_only_where_all_four_cells_pass(tmp_path: Path) -> None:
    control_evaluation = _evaluation(0.5)
    candidate_evaluation = _evaluation(0.49)
    candidate_evaluation["32"] = _evaluation(0.51)["32"]
    records = []
    for name, evaluation in (
        ("compact_unanchored_control", control_evaluation),
        ("semantic", candidate_evaluation),
    ):
        (tmp_path / f"{name}.pt").write_bytes(name.encode("utf-8"))
        records.append(
            {
                "anchor_spec": {"name": name},
                "result_sha256": name,
                "inference_parameter_count": 10,
                "evaluation": evaluation,
            }
        )
    sweep = {
        "status": "DEVELOPMENT_ONLY_NOT_A_PAPER_CLAIM",
        "formal_query_or_database_labels_opened": False,
        "records": records,
    }
    result = select_routes(sweep, tmp_path)
    assert result["routes"]["16"]["selected_candidate"] == "compact_unanchored_control"
    assert result["routes"]["32"]["selected_candidate"] == "semantic"
    assert result["routes"]["64"]["selected_candidate"] == "compact_unanchored_control"
    assert result["assembled_delta_report"]["all_twelve_nonnegative"] is True

