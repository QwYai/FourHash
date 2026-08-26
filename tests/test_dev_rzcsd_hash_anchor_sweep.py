from raw_rebuilt_neural.hash_anchors import HASH_ANCHOR_SPECS
from tools.dev_rzcsd_hash_anchor_sweep import CONTROL


def test_hash_anchor_registry_is_predeclared_and_unique() -> None:
    names = [CONTROL["name"], *[spec.name for spec in HASH_ANCHOR_SPECS]]
    assert len(names) == len(set(names)) == 5
    assert names[0] == "compact_unanchored_control"
    assert any(spec.clip_pca for spec in HASH_ANCHOR_SPECS)
    assert any(spec.semantic_bridge for spec in HASH_ANCHOR_SPECS)

