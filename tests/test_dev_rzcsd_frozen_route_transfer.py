import numpy as np

from tools.dev_rzcsd_frozen_route_transfer import _assemble, _transfer_split


def test_frozen_transfer_route_uses_semantic_model_only_at_32_bits() -> None:
    compact = {str(bits): {"source": f"compact-{bits}"} for bits in (16, 32, 64)}
    semantic = {str(bits): {"source": f"semantic-{bits}"} for bits in (16, 32, 64)}
    result = _assemble(compact, semantic)
    assert result["16"]["source"] == "compact-16"
    assert result["32"]["source"] == "semantic-32"
    assert result["64"]["source"] == "compact-64"


def test_transfer_split_supports_mir_sized_indt_without_overlap() -> None:
    identities = np.arange(5_000, dtype=np.uint64)
    fit, query, database = _transfer_split(identities)
    assert (len(fit), len(query), len(database)) == (3_000, 500, 1_500)
    assert np.intersect1d(fit, query).size == 0
    assert np.intersect1d(fit, database).size == 0
