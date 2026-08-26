from raw_rebuilt_neural.architecture_variants import ROUTING_VARIANTS
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


def test_routing_registry_is_predeclared_and_control_first() -> None:
    names = [variant.name for variant in ROUTING_VARIANTS]
    assert len(names) == len(set(names)) == 5
    assert names[0] == "separate_gelu_linear_control"


def test_selection_gate_rejects_one_regressed_primary_cell() -> None:
    baseline = _evaluation(0.5)
    candidate = _evaluation(0.51)
    candidate["64"]["t2i"]["map_expected_ties"] = 0.499
    report = _delta_report(candidate, baseline)
    assert report["negative_primary_cells"] == 1
    assert report["all_twelve_nonnegative"] is False
    accepted = {
        "delta_report": _delta_report(_evaluation(0.501), baseline),
        "inference_parameter_count": 10,
    }
    rejected = {
        "delta_report": report,
        "inference_parameter_count": 1,
    }
    assert _selection_key(accepted) > _selection_key(rejected)

