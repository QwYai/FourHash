"""Post-plan metric consumer; this is the only worker that opens Q/D labels."""

from __future__ import annotations

from pathlib import Path
import math
from typing import Any, Mapping

import numpy as np

from raw_rebuilt_runtime import MetricLabels
from raw_rebuilt_runtime.metric_loader import (
    load_frozen_metric_labels as load_metric_labels,
)

from .codes import CodeState, open_code_state
from .integrity import (
    StreamingIntegrityError,
    atomic_write_json,
    load_json,
    numeric_sha256,
    production_code_inventory,
    reject_unsafe_output_path,
    require_disjoint_paths,
    require_hashed_json,
    require_no_link_components,
    sha256_file,
    sha256_json,
)
from .metrics import (
    _VerifiedJaccardIDCG,
    _precompute_jaccard_idcg,
    build_metric_prefixes,
    expected_tie_metrics_from_distances,
    mean_query_metrics,
)
from .plan import ChunkSpec, FrozenRankPlan, open_rank_plan
from .protocol import (
    ack_path,
    bundle_path,
    chunk_binding,
    delete_acknowledged_bundle,
    open_bundle,
    open_rank_evidence_seal,
    prepare_spool_root,
    replay_cell_acks,
    verify_ack,
    write_ack,
)


METRIC_RESULT_SCHEMA = "raw_rebuilt_streaming_metric_result_v1"
EVALUATION_SCHEMA = "raw_rebuilt_streaming_evaluation_v1"
METRIC_PARTIAL_SCHEMA = "raw_rebuilt_streaming_private_metric_partial_v1"


def _evaluation_root(
    output_parent: Path,
    plan: FrozenRankPlan,
    *,
    forbidden: Mapping[str, Path],
) -> Path:
    output = reject_unsafe_output_path(Path(output_parent), field="streaming metric output")
    root = reject_unsafe_output_path(
        output / f"metrics-{plan.plan_sha256[:16]}",
        field="streaming metric output root",
    )
    require_disjoint_paths(root, forbidden, field="streaming metric output")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _partial_path(root: Path, chunk: ChunkSpec) -> Path:
    return root / "private_partials" / chunk.cell_id / f"{chunk.name}.json"


def _validate_metric_records(
    records: Any,
    chunk: ChunkSpec,
    cutoffs: tuple[int, ...],
) -> list[dict[str, Any]]:
    if not isinstance(records, list) or len(records) != chunk.end - chunk.start:
        raise StreamingIntegrityError("private metric receipt query coverage changed")
    expected_keys = {
        "database_rows",
        "relevant_rows",
        "has_relevant",
        "average_precision_expected_ties",
        "query_position",
        "query_row_id",
    }
    for cutoff in cutoffs:
        expected_keys.update(
            {
                f"precision_at_{cutoff}_expected_ties",
                f"recall_at_{cutoff}_expected_ties",
                f"binary_ndcg_at_{cutoff}_expected_ties",
                f"j_ndcg_at_{cutoff}_expected_ties",
            }
        )
    if [record.get("query_position") for record in records if isinstance(record, Mapping)] != list(
        range(chunk.start, chunk.end)
    ):
        raise StreamingIntegrityError("private metric receipt query positions changed")
    for record in records:
        if not isinstance(record, Mapping) or set(record) != expected_keys:
            raise StreamingIntegrityError("private metric record schema changed")
        if record["database_rows"] != chunk.database_rows:
            raise StreamingIntegrityError("private metric database geometry changed")
        relevant = record["relevant_rows"]
        if type(relevant) is not int or not 0 <= relevant <= chunk.database_rows:
            raise StreamingIntegrityError("private metric relevant count is invalid")
        if type(record["has_relevant"]) is not bool or record["has_relevant"] != (relevant > 0):
            raise StreamingIntegrityError("private metric relevance marker changed")
        row_id = record["query_row_id"]
        if not isinstance(row_id, str) or len(row_id) != 64 or any(
            value not in "0123456789abcdef" for value in row_id
        ):
            raise StreamingIntegrityError("private metric query row identity is invalid")
        for key in expected_keys - {
            "database_rows",
            "relevant_rows",
            "has_relevant",
            "query_position",
            "query_row_id",
        }:
            value = record[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise StreamingIntegrityError("private metric score is nonnumeric")
            score = float(value)
            if not math.isfinite(score) or score < -1.0e-12 or score > 1.0 + 1.0e-12:
                raise StreamingIntegrityError("private metric score lies outside [0,1]")
    return records


def _write_private_partial(
    root: Path,
    plan: FrozenRankPlan,
    state: CodeState,
    chunk: ChunkSpec,
    bundle: Mapping[str, Any],
    records: list[dict[str, Any]],
    previous_chain: str,
) -> Mapping[str, Any]:
    cutoffs = tuple(int(value) for value in plan.manifest["binding"]["config"]["cutoffs"])
    _validate_metric_records(records, chunk, cutoffs)
    body = {
        "schema": METRIC_PARTIAL_SCHEMA,
        "status": "COMMITTED",
        "binding": chunk_binding(plan, state, chunk),
        "bundle_manifest_sha256": bundle["bundle_manifest_sha256"],
        "distances_file_sha256": bundle["distances"]["file_sha256"],
        "distances_numeric_sha256": bundle["distances"]["numeric_sha256"],
        "per_query": records,
        "previous_private_chain_sha256": previous_chain,
        "private_metric_payload": True,
    }
    receipt_sha = sha256_json(body)
    chain = sha256_json(
        {"previous_private_chain_sha256": previous_chain, "partial_sha256": receipt_sha}
    )
    receipt = {**body, "partial_sha256": receipt_sha, "private_chain_sha256": chain}
    target = _partial_path(root, chunk)
    require_no_link_components(
        target, field="private metric receipt output", allow_missing=True
    )
    if target.exists():
        existing = load_json(target)
        if existing != receipt:
            raise StreamingIntegrityError("private metric receipt was rebound")
    else:
        atomic_write_json(target, receipt)
    return receipt


def _verify_private_partial(
    root: Path,
    plan: FrozenRankPlan,
    state: CodeState,
    chunk: ChunkSpec,
    previous_chain: str,
) -> Mapping[str, Any]:
    target = _partial_path(root, chunk)
    private_root = root / "private_partials"
    try:
        target.relative_to(private_root)
    except ValueError as error:
        raise StreamingIntegrityError("private metric receipt escapes output") from error
    current = root
    for part in target.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise StreamingIntegrityError("private metric receipt contains a symlink component")
    if target.is_symlink() or not target.is_file():
        raise StreamingIntegrityError("private metric receipt must be a regular non-symlink file")
    receipt = load_json(target)
    expected_keys = {
        "schema",
        "status",
        "binding",
        "bundle_manifest_sha256",
        "distances_file_sha256",
        "distances_numeric_sha256",
        "per_query",
        "previous_private_chain_sha256",
        "private_metric_payload",
        "partial_sha256",
        "private_chain_sha256",
    }
    if set(receipt) != expected_keys:
        raise StreamingIntegrityError("private metric receipt contains unbound fields")
    body = {
        key: receipt[key]
        for key in receipt
        if key not in {"partial_sha256", "private_chain_sha256"}
    }
    if (
        receipt.get("schema") != METRIC_PARTIAL_SCHEMA
        or receipt.get("status") != "COMMITTED"
        or receipt.get("partial_sha256") != sha256_json(body)
        or receipt.get("binding") != chunk_binding(plan, state, chunk)
        or receipt.get("previous_private_chain_sha256") != previous_chain
        or receipt.get("private_metric_payload") is not True
    ):
        raise StreamingIntegrityError("private metric receipt binding/hash changed")
    expected_chain = sha256_json(
        {
            "previous_private_chain_sha256": previous_chain,
            "partial_sha256": receipt["partial_sha256"],
        }
    )
    if receipt.get("private_chain_sha256") != expected_chain:
        raise StreamingIntegrityError("private metric receipt chain changed")
    cutoffs = tuple(int(value) for value in plan.manifest["binding"]["config"]["cutoffs"])
    _validate_metric_records(receipt.get("per_query"), chunk, cutoffs)
    return receipt


def _replay_private_partials(
    root: Path,
    plan: FrozenRankPlan,
    state: CodeState,
    cell_id: str,
) -> tuple[list[Mapping[str, Any]], str]:
    receipts = []
    chain = "0" * 64
    missing = False
    for chunk in (item for item in plan.chunks() if item.cell_id == cell_id):
        target = _partial_path(root, chunk)
        if not target.exists():
            missing = True
            continue
        if missing:
            raise StreamingIntegrityError("private metric receipts are noncontiguous")
        receipt = _verify_private_partial(root, plan, state, chunk, chain)
        receipts.append(receipt)
        chain = str(receipt["private_chain_sha256"])
    return receipts, chain


def _known_paths(plan: FrozenRankPlan, spool: Path) -> tuple[set[Path], set[Path]]:
    bundles = {bundle_path(spool, chunk).resolve(strict=False) for chunk in plan.chunks()}
    acks = {ack_path(spool, chunk).resolve(strict=False) for chunk in plan.chunks()}
    return bundles, acks


def _require_spool_entry_inventory(
    spool: Path, plan: FrozenRankPlan, *, final: bool
) -> None:
    require_no_link_components(spool, field="streaming spool")
    expected = {"rank_evidence_complete.json"}
    bundles_root = spool / "bundles"
    acks_root = spool / "acks"
    if bundles_root.exists():
        expected.add("bundles")
    if acks_root.exists():
        expected.add("acks")
    for cell in plan.manifest["cells"]:
        cell_id = str(cell["cell_id"])
        bundle_cell = bundles_root / cell_id
        ack_cell = acks_root / cell_id
        if bundle_cell.exists():
            expected.add(bundle_cell.relative_to(spool).as_posix())
        if ack_cell.exists():
            expected.add(ack_cell.relative_to(spool).as_posix())
    for chunk in plan.chunks():
        bundle = bundle_path(spool, chunk)
        if bundle.exists():
            expected.update(
                {
                    bundle.relative_to(spool).as_posix(),
                    (bundle / "bundle.json").relative_to(spool).as_posix(),
                    (bundle / "distances.npy").relative_to(spool).as_posix(),
                }
            )
        ack = ack_path(spool, chunk)
        if ack.exists():
            expected.add(ack.relative_to(spool).as_posix())
        elif final:
            raise StreamingIntegrityError("final spool lacks a planned ACK")
    actual_items = list(spool.rglob("*"))
    for item in actual_items:
        require_no_link_components(item, field="streaming spool entry")
    actual = {item.relative_to(spool).as_posix() for item in actual_items}
    if actual != expected:
        raise StreamingIntegrityError("spool entry inventory differs from sealed protocol")


def _require_metric_output_inventory(
    root: Path, plan: FrozenRankPlan, *, final: bool
) -> None:
    require_no_link_components(root, field="metric output")
    expected: set[str] = set()
    private_root = root / "private_partials"
    if private_root.exists():
        expected.add("private_partials")
    for cell in plan.manifest["cells"]:
        cell_id = str(cell["cell_id"])
        private_cell = private_root / cell_id
        result_cell = root / cell_id
        if private_cell.exists():
            expected.add(private_cell.relative_to(root).as_posix())
        if result_cell.exists():
            expected.add(result_cell.relative_to(root).as_posix())
        metrics = result_cell / "metrics.json"
        if metrics.exists():
            expected.add(metrics.relative_to(root).as_posix())
        elif final:
            raise StreamingIntegrityError("final metric output lacks a cell result")
    for chunk in plan.chunks():
        partial = _partial_path(root, chunk)
        if partial.exists():
            expected.add(partial.relative_to(root).as_posix())
        elif final:
            raise StreamingIntegrityError("final metric output lacks a private receipt")
    completion = root / "evaluation_complete.json"
    if completion.exists():
        expected.add("evaluation_complete.json")
    elif final:
        raise StreamingIntegrityError("final metric output lacks completion seal")
    actual_items = list(root.rglob("*"))
    for item in actual_items:
        require_no_link_components(item, field="metric output entry")
    actual = {item.relative_to(root).as_posix() for item in actual_items}
    if actual != expected:
        raise StreamingIntegrityError(
            "metric output contains an unregistered file or directory"
        )


def _preflight_before_labels(
    spool: Path,
    plan: FrozenRankPlan,
    state: CodeState,
    metric_root: Path,
) -> tuple[
    dict[str, list[Mapping[str, Any]]],
    dict[str, str],
    dict[str, list[Mapping[str, Any]]],
    dict[str, str],
]:
    """Replay every published byte/structure before crossing the label gate."""

    seal = open_rank_evidence_seal(spool, plan, state)
    _require_spool_entry_inventory(spool, plan, final=False)
    _require_metric_output_inventory(metric_root, plan, final=False)
    known_spool_files = {"rank_evidence_complete.json"}
    for chunk in plan.chunks():
        bundle = bundle_path(spool, chunk)
        if bundle.exists():
            known_spool_files.update(
                {
                    (bundle / "bundle.json").relative_to(spool).as_posix(),
                    (bundle / "distances.npy").relative_to(spool).as_posix(),
                }
            )
        ack = ack_path(spool, chunk)
        if ack.exists():
            known_spool_files.add(ack.relative_to(spool).as_posix())
    actual_spool_files = {
        item.relative_to(spool).as_posix()
        for item in spool.rglob("*")
        if item.is_file() or item.is_symlink()
    }
    if actual_spool_files != known_spool_files:
        raise StreamingIntegrityError("spool contains files outside the sealed protocol")
    known_bundles, known_acks = _known_paths(plan, spool)
    known_partials = {_partial_path(metric_root, chunk).resolve(strict=False) for chunk in plan.chunks()}
    actual_acks = {
        path.resolve(strict=True)
        for path in (spool / "acks").rglob("*.json")
    } if (spool / "acks").exists() else set()
    if not actual_acks.issubset(known_acks):
        raise StreamingIntegrityError("spool contains an ACK outside the frozen plan")
    actual_bundle_roots = {
        path.parent.resolve(strict=True)
        for path in (spool / "bundles").rglob("bundle.json")
    } if (spool / "bundles").exists() else set()
    if not actual_bundle_roots.issubset(known_bundles):
        raise StreamingIntegrityError("spool contains a bundle outside the frozen plan")
    actual_partials = {
        path.resolve(strict=True)
        for path in (metric_root / "private_partials").rglob("*.json")
    } if (metric_root / "private_partials").exists() else set()
    if not actual_partials.issubset(known_partials):
        raise StreamingIntegrityError("metric output contains a private receipt outside the plan")

    receipts_by_cell: dict[str, list[Mapping[str, Any]]] = {}
    chains: dict[str, str] = {}
    partials_by_cell: dict[str, list[Mapping[str, Any]]] = {}
    partial_chains: dict[str, str] = {}
    for cell in plan.manifest["cells"]:
        cell_id = str(cell["cell_id"])
        receipts, chain = replay_cell_acks(spool, plan, state, cell_id)
        receipts_by_cell[cell_id] = receipts
        chains[cell_id] = chain
        partials, partial_chain = _replay_private_partials(metric_root, plan, state, cell_id)
        if len(partials) not in {len(receipts), len(receipts) + 1}:
            raise StreamingIntegrityError("ACK/private metric receipt frontiers disagree")
        partials_by_cell[cell_id] = partials
        partial_chains[cell_id] = partial_chain
    for descriptor, chunk in zip(seal["bundles"], plan.chunks()):
        target = bundle_path(spool, chunk)
        if target.exists():
            evidence = open_bundle(spool, plan, state, chunk)
            try:
                if (
                    descriptor["bundle_manifest_sha256"]
                    != evidence.manifest["bundle_manifest_sha256"]
                    or descriptor["distances_file_sha256"]
                    != evidence.manifest["distances"]["file_sha256"]
                    or descriptor["distances_numeric_sha256"]
                    != evidence.manifest["distances"]["numeric_sha256"]
                ):
                    raise StreamingIntegrityError("distance bundle differs from frozen evidence seal")
            finally:
                evidence.close()
        acked = chunk.ordinal < len(receipts_by_cell[chunk.cell_id])
        partial = (
            partials_by_cell[chunk.cell_id][chunk.ordinal]
            if chunk.ordinal < len(partials_by_cell[chunk.cell_id])
            else None
        )
        if acked:
            ack = receipts_by_cell[chunk.cell_id][chunk.ordinal]
            if partial is None or ack["opaque_metric_commitment_sha256"] != partial["partial_sha256"]:
                raise StreamingIntegrityError("opaque ACK lacks its metric-private commitment")
            if any(
                partial[key] != descriptor[key]
                for key in (
                    "bundle_manifest_sha256",
                    "distances_file_sha256",
                    "distances_numeric_sha256",
                )
            ):
                raise StreamingIntegrityError("private metric receipt differs from evidence seal")
        elif partial is not None:
            if not target.exists():
                raise StreamingIntegrityError("unacknowledged private receipt lost its evidence bundle")
        elif not target.exists():
            raise StreamingIntegrityError("frozen evidence is neither present nor privately acknowledged")
    return receipts_by_cell, chains, partials_by_cell, partial_chains


def _open_verified_metric_labels(
    runtime_root: Path,
    plan: FrozenRankPlan,
    *,
    process_data_root: Path | None,
    _test_allow_synthetic: bool,
) -> MetricLabels:
    # This mapping is built only from a FrozenRankPlan returned by the full
    # verifier above; callers cannot pass an arbitrary gate dictionary here.
    labels = load_metric_labels(
        runtime_root,
        rank_contract=plan.manifest,
        process_data_root=process_data_root,
        _test_allow_synthetic=_test_allow_synthetic,
    )
    identity = plan.manifest["runtime_identity"]
    if labels.source_seal_sha256 != plan.manifest["source_seal_sha256"]:
        raise StreamingIntegrityError("metric labels and frozen source seal differ")
    if numeric_sha256(labels.query_row_ids) != identity["query_row_ids_numeric_sha256"]:
        raise StreamingIntegrityError("metric query row identities changed")
    if numeric_sha256(labels.database_row_ids) != identity["database_row_ids_numeric_sha256"]:
        raise StreamingIntegrityError("metric database row identities changed")
    if labels.query.shape != (identity["query_rows"], plan.manifest["label_dim"]):
        raise StreamingIntegrityError("metric query label geometry changed")
    if labels.database.shape != (identity["database_rows"], plan.manifest["label_dim"]):
        raise StreamingIntegrityError("metric database label geometry changed")
    return labels


def _query_truth(
    labels: MetricLabels,
    query_position: int,
    *,
    cutoffs: tuple[int, ...],
    database_label_counts: np.ndarray,
    prefixes: Any,
) -> tuple[np.ndarray, np.ndarray, _VerifiedJaccardIDCG]:
    query = np.asarray(labels.query[query_position], dtype=np.uint8)
    active = np.flatnonzero(query)
    if active.size:
        intersection = np.asarray(
            labels.database[:, active].sum(axis=1, dtype=np.uint16), dtype=np.uint16
        )
    else:
        intersection = np.zeros(len(labels.database), dtype=np.uint16)
    union = database_label_counts + np.uint16(active.size) - intersection
    gains = np.divide(
        intersection,
        union,
        out=np.zeros(len(intersection), dtype=np.float64),
        where=union != 0,
    )
    ideal = _precompute_jaccard_idcg(gains, cutoffs)
    return intersection != 0, ideal.gains, ideal


def _query_record(
    labels: MetricLabels,
    query_position: int,
    distances: np.ndarray,
    *,
    bits: int,
    cutoffs: tuple[int, ...],
    prefixes: Any,
    truth: tuple[np.ndarray, np.ndarray, _VerifiedJaccardIDCG],
) -> dict[str, Any]:
    relevance, gains, ideal = truth
    record = expected_tie_metrics_from_distances(
        relevance,
        distances,
        bits=bits,
        graded_gains=gains,
        cutoffs=cutoffs,
        prefixes=prefixes,
        ideal_jaccard_dcg=ideal,
    )
    record["query_position"] = query_position
    record["query_row_id"] = bytes(labels.query_row_ids[query_position]).decode("ascii")
    return record


def _write_results(
    root: Path,
    plan: FrozenRankPlan,
    state: CodeState,
    spool: Path,
) -> Path:
    descriptors = []
    for cell in plan.manifest["cells"]:
        cell_id = str(cell["cell_id"])
        cell_chunks = [chunk for chunk in plan.chunks() if chunk.cell_id == cell_id]
        ack_receipts, final_ack_chain = replay_cell_acks(
            spool, plan, state, cell_id
        )
        receipts, final_private_chain = _replay_private_partials(
            root, plan, state, cell_id
        )
        if len(receipts) != len(cell_chunks):
            raise StreamingIntegrityError("cannot finalize an incompletely acknowledged cell")
        if len(ack_receipts) != len(receipts) or len(ack_receipts) != len(
            cell["chunks"]
        ):
            raise StreamingIntegrityError(
                "ACK/private receipts are not a complete per-cell bijection"
            )
        pairs = []
        for chunk, ack, private in zip(cell_chunks, ack_receipts, receipts):
            if (
                ack.get("binding") != private.get("binding")
                or ack.get("opaque_metric_commitment_sha256")
                != private.get("partial_sha256")
                or any(
                    ack.get(key) != private.get(key)
                    for key in (
                        "bundle_manifest_sha256",
                        "distances_file_sha256",
                        "distances_numeric_sha256",
                    )
                )
                or ack.get("binding", {}).get("cell_id") != cell_id
                or ack.get("binding", {}).get("ordinal") != chunk.ordinal
                or ack.get("binding", {}).get("start") != chunk.start
                or ack.get("binding", {}).get("end") != chunk.end
            ):
                raise StreamingIntegrityError(
                    "ACK/private receipt commitment or cell identity changed"
                )
            pairs.append(
                {
                    "ordinal": chunk.ordinal,
                    "ack_sha256": ack["ack_sha256"],
                    "partial_sha256": private["partial_sha256"],
                }
            )
        bijection_sha256 = sha256_json({"cell_id": cell_id, "pairs": pairs})
        records = [record for receipt in receipts for record in receipt["per_query"]]
        if [record["query_position"] for record in records] != list(range(int(cell["query_rows"]))):
            raise StreamingIntegrityError("metric ACKs do not cover every query exactly once")
        summary = mean_query_metrics(records)
        body = {
            "schema": METRIC_RESULT_SCHEMA,
            "status": "COMPLETE",
            "dataset": plan.manifest["dataset"],
            "source_seal_sha256": plan.manifest["source_seal_sha256"],
            "rank_plan_sha256": plan.plan_sha256,
            "code_state_manifest_sha256": state.manifest["manifest_sha256"],
            "cell_id": cell_id,
            "direction": cell["direction"],
            "bits": cell["bits"],
            "cutoffs": plan.manifest["binding"]["config"]["cutoffs"],
            "primary_metric": "map_expected_ties",
            "graded_metric": "ground-truth linear soft-Jaccard J-NDCG",
            "final_ack_chain_sha256": final_ack_chain,
            "final_private_metric_chain_sha256": final_private_chain,
            "ack_private_bijection_sha256": bijection_sha256,
            "ack_private_bijection_verified": True,
            "summary": summary,
            "per_query": records,
            "metric_labels_opened_after_verified_plan": True,
            "streaming_code_inventory": production_code_inventory(),
        }
        result = {**body, "metric_result_sha256": sha256_json(body)}
        target = root / cell_id / "metrics.json"
        require_no_link_components(
            target, field="metric result output", allow_missing=True
        )
        if target.exists():
            existing = load_json(target)
            if existing != result:
                raise StreamingIntegrityError("completed metric result was rebound")
        else:
            atomic_write_json(target, result)
        descriptors.append(
            {
                "path": target.relative_to(root).as_posix(),
                "size": target.stat().st_size,
                "sha256": sha256_file(target),
                "cell_id": cell_id,
                "direction": cell["direction"],
                "bits": cell["bits"],
                "map_expected_ties": summary["map_expected_ties"],
                "metric_result_sha256": result["metric_result_sha256"],
                "final_ack_chain_sha256": final_ack_chain,
                "final_private_metric_chain_sha256": final_private_chain,
                "ack_private_bijection_sha256": bijection_sha256,
            }
        )
    complete_body = {
        "schema": EVALUATION_SCHEMA,
        "status": "COMPLETE",
        "dataset": plan.manifest["dataset"],
        "source_seal_sha256": plan.manifest["source_seal_sha256"],
        "rank_plan_sha256": plan.plan_sha256,
        "code_state_manifest_sha256": state.manifest["manifest_sha256"],
        "metric_boundary": "verified-frozen-plan then metric labels",
        "results": descriptors,
    }
    complete = {**complete_body, "complete_sha256": sha256_json(complete_body)}
    target = root / "evaluation_complete.json"
    require_no_link_components(
        target, field="evaluation completion output", allow_missing=True
    )
    if target.exists():
        existing = load_json(target)
        require_hashed_json(
            existing,
            hash_field="complete_sha256",
            schema=EVALUATION_SCHEMA,
            status="COMPLETE",
            field="streaming evaluation completion",
        )
        if existing != complete:
            raise StreamingIntegrityError("evaluation completion was rebound")
    else:
        atomic_write_json(target, complete)
    _require_spool_entry_inventory(spool, plan, final=True)
    _require_metric_output_inventory(root, plan, final=True)
    return root


def _run_metric_session(
    runtime_root: Path,
    code_state_root: Path,
    plan_root: Path,
    spool_root: Path,
    output_parent: Path,
    *,
    max_new_acks: int | None,
    wait_for_bundles: bool,
    poll_seconds: float,
    process_data_root: Path | None,
    _test_allow_synthetic: bool,
) -> dict[str, Any]:
    if max_new_acks is not None and max_new_acks < 0:
        raise ValueError("max_new_acks must be nonnegative")
    require_no_link_components(runtime_root, field="metric runtime")
    if process_data_root is not None:
        require_no_link_components(process_data_root, field="metric ProcessData source")
    state = open_code_state(code_state_root)
    try:
        plan = open_rank_plan(plan_root, code_state_root)
        metric_root = _evaluation_root(
            output_parent,
            plan,
            forbidden={
                "runtime": Path(runtime_root),
                "packed code state": Path(code_state_root),
                "rank plan": Path(plan_root),
                "streaming spool": Path(spool_root),
            },
        )
        spool = prepare_spool_root(
            spool_root,
            forbidden={
                "runtime": Path(runtime_root),
                "packed code state": Path(code_state_root),
                "rank plan": Path(plan_root),
                "metric output": metric_root,
            },
        )
        (
            receipts_by_cell,
            chains,
            partials_by_cell,
            partial_chains,
        ) = _preflight_before_labels(spool, plan, state, metric_root)
        # Recover the sole legal crash window: private commit published, ACK
        # not yet published. No score payload crosses into the ACK.
        for cell in plan.manifest["cells"]:
            cell_id = str(cell["cell_id"])
            if len(partials_by_cell[cell_id]) == len(receipts_by_cell[cell_id]) + 1:
                chunk = [item for item in plan.chunks() if item.cell_id == cell_id][
                    len(receipts_by_cell[cell_id])
                ]
                partial = partials_by_cell[cell_id][-1]
                evidence = open_bundle(spool, plan, state, chunk)
                try:
                    receipt = write_ack(
                        spool,
                        plan,
                        state,
                        chunk,
                        evidence.manifest,
                        str(partial["partial_sha256"]),
                        chains[cell_id],
                    )
                finally:
                    evidence.close()
                verified = verify_ack(spool, plan, state, chunk, chains[cell_id])
                delete_acknowledged_bundle(spool, plan, state, chunk, verified)
                receipts_by_cell[cell_id].append(receipt)
                chains[cell_id] = str(receipt["ack_chain_sha256"])
        for cell in plan.manifest["cells"]:
            cell_id = str(cell["cell_id"])
            chunks = [item for item in plan.chunks() if item.cell_id == cell_id]
            for chunk, receipt in zip(chunks, receipts_by_cell[cell_id]):
                delete_acknowledged_bundle(spool, plan, state, chunk, receipt)
        metric_labels = _open_verified_metric_labels(
            runtime_root,
            plan,
            process_data_root=process_data_root,
            _test_allow_synthetic=_test_allow_synthetic,
        )
        for receipts in partials_by_cell.values():
            for receipt in receipts:
                for record in receipt["per_query"]:
                    position = int(record["query_position"])
                    expected_row_id = bytes(metric_labels.query_row_ids[position]).decode("ascii")
                    if record["query_row_id"] != expected_row_id:
                        raise StreamingIntegrityError(
                            "resumed metric ACK is bound to another query identity"
                        )
        cutoffs = tuple(int(value) for value in plan.manifest["binding"]["config"]["cutoffs"])
        prefixes = build_metric_prefixes(len(metric_labels.database), cutoffs)
        database_label_counts = np.asarray(
            metric_labels.database.sum(axis=1, dtype=np.uint16), dtype=np.uint16
        )
        new_acks = 0
        chunks_in_order = list(plan.chunks())
        total = len(chunks_in_order)
        truth_chunk: tuple[int, int] | None = None
        truth_cache: dict[
            int, tuple[np.ndarray, np.ndarray, dict[int, float]]
        ] = {}
        acknowledged = sum(len(values) for values in receipts_by_cell.values())
        for chunk in chunks_in_order:
            frontier = len(receipts_by_cell[chunk.cell_id])
            if chunk.ordinal < frontier:
                continue
            if chunk.ordinal > frontier:
                raise StreamingIntegrityError("metric cell frontier has a hole")
            if max_new_acks is not None and new_acks >= max_new_acks:
                return {
                    "status": "IN_PROGRESS",
                    "total_chunks": total,
                    "acknowledged_chunks": acknowledged,
                    "new_acks": new_acks,
                }
            target = bundle_path(spool, chunk)
            if not target.exists():
                raise StreamingIntegrityError(
                    "sealed unacknowledged evidence bundle is missing"
                )
            evidence = open_bundle(spool, plan, state, chunk)
            try:
                distance_snapshot = np.array(
                    evidence.distances, dtype=np.uint8, order="C", copy=True
                )
                if numeric_sha256(distance_snapshot) != evidence.manifest["distances"][
                    "numeric_sha256"
                ]:
                    raise StreamingIntegrityError("distance evidence changed during snapshot")
                bundle_manifest = dict(evidence.manifest)
            finally:
                evidence.close()
            try:
                if truth_chunk != (chunk.start, chunk.end):
                    truth_cache = {
                        query_position: _query_truth(
                            metric_labels,
                            query_position,
                            cutoffs=cutoffs,
                            database_label_counts=database_label_counts,
                            prefixes=prefixes,
                        )
                        for query_position in range(chunk.start, chunk.end)
                    }
                    truth_chunk = (chunk.start, chunk.end)
                records = [
                    _query_record(
                        metric_labels,
                        query_position,
                        distance_snapshot[query_position - chunk.start],
                        bits=chunk.bits,
                        cutoffs=cutoffs,
                        prefixes=prefixes,
                        truth=truth_cache[query_position],
                    )
                    for query_position in range(chunk.start, chunk.end)
                ]
                partial = _write_private_partial(
                    metric_root,
                    plan,
                    state,
                    chunk,
                    bundle_manifest,
                    records,
                    partial_chains[chunk.cell_id],
                )
                receipt = write_ack(
                    spool,
                    plan,
                    state,
                    chunk,
                    bundle_manifest,
                    str(partial["partial_sha256"]),
                    chains[chunk.cell_id],
                )
            finally:
                del distance_snapshot
            verified = verify_ack(spool, plan, state, chunk, chains[chunk.cell_id])
            delete_acknowledged_bundle(spool, plan, state, chunk, verified)
            receipts_by_cell[chunk.cell_id].append(receipt)
            chains[chunk.cell_id] = str(receipt["ack_chain_sha256"])
            partials_by_cell[chunk.cell_id].append(partial)
            partial_chains[chunk.cell_id] = str(partial["private_chain_sha256"])
            new_acks += 1
            acknowledged += 1
        if acknowledged != total:
            raise StreamingIntegrityError("metric pass did not cover the sealed plan")
        evaluation = _write_results(metric_root, plan, state, spool)
        return {
            "status": "COMPLETE",
            "total_chunks": total,
            "acknowledged_chunks": acknowledged,
            "new_acks": new_acks,
            "evaluation_root": str(evaluation),
        }
    finally:
        state.close()


def consume_metric_bundles(
    runtime_root: Path,
    code_state_root: Path,
    plan_root: Path,
    spool_root: Path,
    output_parent: Path,
    *,
    max_new_acks: int | None = None,
    process_data_root: Path | None = None,
    _test_allow_synthetic: bool = False,
) -> dict[str, Any]:
    return _run_metric_session(
        runtime_root,
        code_state_root,
        plan_root,
        spool_root,
        output_parent,
        max_new_acks=max_new_acks,
        wait_for_bundles=False,
        poll_seconds=0.1,
        process_data_root=process_data_root,
        _test_allow_synthetic=_test_allow_synthetic,
    )


def serve_metric_worker(
    runtime_root: Path,
    code_state_root: Path,
    plan_root: Path,
    spool_root: Path,
    output_parent: Path,
    *,
    poll_seconds: float = 0.1,
    process_data_root: Path | None = None,
) -> dict[str, Any]:
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    return _run_metric_session(
        runtime_root,
        code_state_root,
        plan_root,
        spool_root,
        output_parent,
        max_new_acks=None,
        wait_for_bundles=True,
        poll_seconds=poll_seconds,
        process_data_root=process_data_root,
        _test_allow_synthetic=False,
    )


__all__ = ["consume_metric_bundles", "serve_metric_worker"]
