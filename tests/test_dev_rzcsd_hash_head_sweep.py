from raw_rebuilt_neural.hash_head_variants import HASH_HEAD_VARIANTS
from tools.dev_rzcsd_architecture_sweep import _delta_report, _selection_key


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


def test_registry_control_is_first() -> None:
    assert HASH_HEAD_VARIANTS[0].name == "compact_linear_control"


def test_selection_gate_rejects_a_single_regression() -> None:
    baseline = _evaluation(0.5)
    candidate = _evaluation(0.501)
    candidate["32"]["t2i"]["map_expected_ties"] = 0.499
    rejected = {
        "delta_report": _delta_report(candidate, baseline),
        "inference_parameter_count": 1,
    }
    accepted = {
        "delta_report": _delta_report(_evaluation(0.5001), baseline),
        "inference_parameter_count": 100,
    }
    assert rejected["delta_report"]["negative_primary_cells"] == 1
    assert _selection_key(accepted) > _selection_key(rejected)
