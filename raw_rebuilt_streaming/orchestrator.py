"""Subprocess-only orchestration for strict rank/metric process isolation."""

from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from .integrity import (
    StreamingIntegrityError,
    load_json,
    production_code_inventory,
    reject_unsafe_output_path,
    require_disjoint_paths,
    require_hashed_json,
    require_no_link_components,
    sha256_file,
    sha256_json,
)
from .metrics import mean_query_metrics
from .plan import PLAN_SCHEMA
from .protocol import prepare_spool_root


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_SCHEMA = "raw_rebuilt_streaming_evaluation_v1"
METRIC_RESULT_SCHEMA = "raw_rebuilt_streaming_metric_result_v1"


def _start_worker(command: Sequence[str]) -> subprocess.Popen:
    """Launch a source-tree worker without depending on the caller's cwd."""
    return subprocess.Popen(
        list(command),
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
    )


def _hex64(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _require_safe_descendant(target: Path, root: Path, *, field: str) -> None:
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise StreamingIntegrityError(f"{field} escapes evaluation output") from error
    current = root
    if current.is_symlink():
        raise StreamingIntegrityError(f"{field} root is a symlink")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise StreamingIntegrityError(f"{field} contains a symlink component")


def _stable_json(path: Path, root: Path, *, field: str) -> tuple[dict[str, Any], int, str]:
    _require_safe_descendant(path, root, field=field)
    if path.is_symlink() or not path.is_file():
        raise StreamingIntegrityError(f"{field} must be a regular non-symlink file")
    before_size = path.stat().st_size
    before_sha = sha256_file(path)
    value = load_json(path)
    if path.stat().st_size != before_size or sha256_file(path) != before_sha:
        raise StreamingIntegrityError(f"{field} changed during verification")
    return value, before_size, before_sha


def _validate_records(
    records: Any,
    *,
    start: int,
    end: int,
    database_rows: int,
    cutoffs: tuple[int, ...],
) -> list[Mapping[str, Any]]:
    if not isinstance(records, list) or len(records) != end - start:
        raise StreamingIntegrityError("metric record query coverage changed")
    score_keys = {"average_precision_expected_ties"}
    for cutoff in cutoffs:
        score_keys.update(
            {
                f"precision_at_{cutoff}_expected_ties",
                f"recall_at_{cutoff}_expected_ties",
                f"binary_ndcg_at_{cutoff}_expected_ties",
                f"j_ndcg_at_{cutoff}_expected_ties",
            }
        )
    expected_keys = score_keys | {
        "database_rows",
        "relevant_rows",
        "has_relevant",
        "query_position",
        "query_row_id",
    }
    for offset, record in enumerate(records, start=start):
        if not isinstance(record, Mapping) or set(record) != expected_keys:
            raise StreamingIntegrityError("metric record schema changed")
        relevant = record["relevant_rows"]
        if (
            record["query_position"] != offset
            or record["database_rows"] != database_rows
            or type(relevant) is not int
            or not 0 <= relevant <= database_rows
            or type(record["has_relevant"]) is not bool
            or record["has_relevant"] != (relevant > 0)
            or not _hex64(record["query_row_id"])
        ):
            raise StreamingIntegrityError("metric record identity/count semantics changed")
        for key in score_keys:
            score = record[key]
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
            ):
                raise StreamingIntegrityError("metric record score lies outside [0,1]")
    return records


def _verify_cell_receipt_bijection(
    evaluation: Path,
    spool: Path,
    plan: Mapping[str, Any],
    cell: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    cell_id = str(cell["cell_id"])
    ack_root = spool / "acks" / cell_id
    private_root = evaluation / "private_partials" / cell_id
    expected_ack_names = {
        f"chunk-{int(chunk['start']):09d}-{int(chunk['end']):09d}.json"
        for chunk in cell["chunks"]
    }
    actual_ack_names = (
        {path.name for path in ack_root.iterdir()} if ack_root.is_dir() else set()
    )
    actual_private_names = (
        {path.name for path in private_root.iterdir()} if private_root.is_dir() else set()
    )
    if actual_ack_names != expected_ack_names or actual_private_names != expected_ack_names:
        raise StreamingIntegrityError("ACK/private receipt file inventory is not bijective")
    ack_chain = "0" * 64
    private_chain = "0" * 64
    pairs = []
    private_records = []
    cutoffs = tuple(int(value) for value in result["cutoffs"])
    for chunk in cell["chunks"]:
        ordinal = int(chunk["ordinal"])
        start = int(chunk["start"])
        end = int(chunk["end"])
        name = f"chunk-{start:09d}-{end:09d}.json"
        ack, _ack_size, _ack_file_sha = _stable_json(
            ack_root / name, spool, field="metric ACK"
        )
        private, _private_size, _private_file_sha = _stable_json(
            private_root / name, evaluation, field="private metric receipt"
        )
        if set(ack) != {
            "schema", "status", "binding", "bundle_manifest_sha256",
            "distances_file_sha256", "distances_numeric_sha256",
            "opaque_metric_commitment_sha256", "previous_ack_chain_sha256",
            "metric_payload_exposed_to_rank_worker", "ack_sha256",
            "ack_chain_sha256",
        } or set(private) != {
            "schema", "status", "binding", "bundle_manifest_sha256",
            "distances_file_sha256", "distances_numeric_sha256", "per_query",
            "previous_private_chain_sha256", "private_metric_payload",
            "partial_sha256", "private_chain_sha256",
        }:
            raise StreamingIntegrityError("ACK/private receipt schema changed")
        ack_body = {
            key: value for key, value in ack.items()
            if key not in {"ack_sha256", "ack_chain_sha256"}
        }
        private_body = {
            key: value for key, value in private.items()
            if key not in {"partial_sha256", "private_chain_sha256"}
        }
        expected_ack_chain = sha256_json(
            {"previous_ack_chain_sha256": ack_chain, "ack_sha256": ack.get("ack_sha256")}
        )
        expected_private_chain = sha256_json(
            {
                "previous_private_chain_sha256": private_chain,
                "partial_sha256": private.get("partial_sha256"),
            }
        )
        binding = ack.get("binding")
        if (
            ack.get("schema") != "raw_rebuilt_streaming_metric_ack_v1"
            or private.get("schema")
            != "raw_rebuilt_streaming_private_metric_partial_v1"
            or ack.get("status") != "ACKNOWLEDGED"
            or private.get("status") != "COMMITTED"
            or ack.get("ack_sha256") != sha256_json(ack_body)
            or private.get("partial_sha256") != sha256_json(private_body)
            or ack.get("previous_ack_chain_sha256") != ack_chain
            or private.get("previous_private_chain_sha256") != private_chain
            or ack.get("ack_chain_sha256") != expected_ack_chain
            or private.get("private_chain_sha256") != expected_private_chain
            or ack.get("metric_payload_exposed_to_rank_worker") is not False
            or private.get("private_metric_payload") is not True
            or binding != private.get("binding")
            or not isinstance(binding, Mapping)
            or binding.get("rank_plan_sha256") != plan["rank_plan_sha256"]
            or binding.get("cell_id") != cell_id
            or binding.get("direction") != cell["direction"]
            or binding.get("bits") != cell["bits"]
            or binding.get("ordinal") != ordinal
            or binding.get("start") != start
            or binding.get("end") != end
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
        ):
            raise StreamingIntegrityError("ACK/private chain or commitment changed")
        records = _validate_records(
            private.get("per_query"),
            start=start,
            end=end,
            database_rows=int(cell["database_rows"]),
            cutoffs=cutoffs,
        )
        private_records.extend(records)
        pairs.append(
            {
                "ordinal": ordinal,
                "ack_sha256": ack["ack_sha256"],
                "partial_sha256": private["partial_sha256"],
            }
        )
        ack_chain = expected_ack_chain
        private_chain = expected_private_chain
    if (
        result.get("final_ack_chain_sha256") != ack_chain
        or result.get("final_private_metric_chain_sha256") != private_chain
        or result.get("ack_private_bijection_sha256")
        != sha256_json({"cell_id": cell_id, "pairs": pairs})
        or result.get("ack_private_bijection_verified") is not True
        or result.get("per_query") != private_records
    ):
        raise StreamingIntegrityError("metric result is not anchored to both receipt chains")


def _verify_evaluation_after_workers(
    evaluation: Path,
    spool: Path,
    plan: Mapping[str, Any],
) -> None:
    if evaluation.is_symlink() or not evaluation.is_dir():
        raise StreamingIntegrityError("evaluation output must be a regular directory")
    initial_inventory = _verify_final_entry_inventories(evaluation, spool, plan)
    completion_path = evaluation / "evaluation_complete.json"
    completion, completion_size, completion_file_sha = _stable_json(
        completion_path, evaluation, field="evaluation completion"
    )
    require_hashed_json(
        completion,
        hash_field="complete_sha256",
        schema=EVALUATION_SCHEMA,
        status="COMPLETE",
        field="orchestrated evaluation completion",
    )
    if set(completion) != {
        "schema", "status", "dataset", "source_seal_sha256",
        "rank_plan_sha256", "code_state_manifest_sha256", "metric_boundary",
        "results", "complete_sha256",
    } or (
        completion.get("dataset") != plan["dataset"]
        or completion.get("source_seal_sha256") != plan["source_seal_sha256"]
        or completion.get("rank_plan_sha256") != plan["rank_plan_sha256"]
        or completion.get("code_state_manifest_sha256")
        != plan["binding"]["code_state_manifest_sha256"]
        or completion.get("metric_boundary")
        != "verified-frozen-plan then metric labels"
    ):
        raise StreamingIntegrityError("evaluation completion content contract changed")
    descriptors = completion.get("results")
    cells = plan["cells"]
    if not isinstance(descriptors, list) or len(descriptors) != len(cells):
        raise StreamingIntegrityError("evaluation result coverage changed")
    inventory = production_code_inventory()
    for descriptor, cell in zip(descriptors, cells):
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "path", "size", "sha256", "cell_id", "direction", "bits",
            "map_expected_ties", "metric_result_sha256",
            "final_ack_chain_sha256", "final_private_metric_chain_sha256",
            "ack_private_bijection_sha256",
        }:
            raise StreamingIntegrityError("evaluation result descriptor schema changed")
        cell_id = str(cell["cell_id"])
        expected_relative = f"{cell_id}/metrics.json"
        if (
            descriptor.get("path") != expected_relative
            or descriptor.get("cell_id") != cell_id
            or descriptor.get("direction") != cell["direction"]
            or descriptor.get("bits") != cell["bits"]
            or type(descriptor.get("size")) is not int
            or int(descriptor["size"]) < 1
            or not _hex64(descriptor.get("sha256"))
        ):
            raise StreamingIntegrityError("evaluation result descriptor content changed")
        metric_path = evaluation / expected_relative
        result, metric_size, metric_file_sha = _stable_json(
            metric_path, evaluation, field="metric result"
        )
        if metric_size != descriptor["size"] or metric_file_sha != descriptor["sha256"]:
            raise StreamingIntegrityError("metric result bytes differ from completion")
        require_hashed_json(
            result,
            hash_field="metric_result_sha256",
            schema=METRIC_RESULT_SCHEMA,
            status="COMPLETE",
            field="metric result",
        )
        if set(result) != {
            "schema", "status", "dataset", "source_seal_sha256",
            "rank_plan_sha256", "code_state_manifest_sha256", "cell_id",
            "direction", "bits", "cutoffs", "primary_metric", "graded_metric",
            "final_ack_chain_sha256", "final_private_metric_chain_sha256",
            "ack_private_bijection_sha256", "ack_private_bijection_verified",
            "summary", "per_query", "metric_labels_opened_after_verified_plan",
            "streaming_code_inventory", "metric_result_sha256",
        } or (
            result.get("dataset") != plan["dataset"]
            or result.get("source_seal_sha256") != plan["source_seal_sha256"]
            or result.get("rank_plan_sha256") != plan["rank_plan_sha256"]
            or result.get("code_state_manifest_sha256")
            != plan["binding"]["code_state_manifest_sha256"]
            or result.get("cell_id") != cell_id
            or result.get("direction") != cell["direction"]
            or result.get("bits") != cell["bits"]
            or result.get("cutoffs") != plan["binding"]["config"]["cutoffs"]
            or result.get("primary_metric") != "map_expected_ties"
            or result.get("graded_metric")
            != "ground-truth linear soft-Jaccard J-NDCG"
            or result.get("metric_labels_opened_after_verified_plan") is not True
            or result.get("streaming_code_inventory") != inventory
        ):
            raise StreamingIntegrityError("metric result content contract changed")
        records = _validate_records(
            result.get("per_query"),
            start=0,
            end=int(cell["query_rows"]),
            database_rows=int(cell["database_rows"]),
            cutoffs=tuple(int(value) for value in result["cutoffs"]),
        )
        if result.get("summary") != mean_query_metrics(list(records)):
            raise StreamingIntegrityError("metric summary differs from per-query records")
        _verify_cell_receipt_bijection(evaluation, spool, plan, cell, result)
        for key in (
            "metric_result_sha256", "final_ack_chain_sha256",
            "final_private_metric_chain_sha256", "ack_private_bijection_sha256",
        ):
            if descriptor.get(key) != result.get(key) or not _hex64(result.get(key)):
                raise StreamingIntegrityError("completion/result chain anchor changed")
        if descriptor.get("map_expected_ties") != result["summary"]["map_expected_ties"]:
            raise StreamingIntegrityError("completion/result primary metric changed")
        if metric_path.stat().st_size != metric_size or sha256_file(metric_path) != metric_file_sha:
            raise StreamingIntegrityError("metric result changed after content verification")
    if (
        completion_path.stat().st_size != completion_size
        or sha256_file(completion_path) != completion_file_sha
    ):
        raise StreamingIntegrityError("evaluation completion changed after verification")
    final_inventory = _verify_final_entry_inventories(evaluation, spool, plan)
    if final_inventory != initial_inventory:
        raise StreamingIntegrityError("final output inventory changed during parent verification")


def _verify_final_entry_inventories(
    evaluation: Path,
    spool: Path,
    plan: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    expected_evaluation = {
        "evaluation_complete.json": "file",
        "private_partials": "directory",
    }
    expected_spool = {
        "rank_evidence_complete.json": "file",
        "bundles": "directory",
        "acks": "directory",
    }
    for cell in plan["cells"]:
        cell_id = str(cell["cell_id"])
        expected_evaluation.update(
            {
                cell_id: "directory",
                f"{cell_id}/metrics.json": "file",
                f"private_partials/{cell_id}": "directory",
            }
        )
        expected_spool.update(
            {f"bundles/{cell_id}": "directory", f"acks/{cell_id}": "directory"}
        )
        for chunk in cell["chunks"]:
            name = (
                f"chunk-{int(chunk['start']):09d}-{int(chunk['end']):09d}.json"
            )
            expected_evaluation[f"private_partials/{cell_id}/{name}"] = "file"
            expected_spool[f"acks/{cell_id}/{name}"] = "file"
    actual_evaluation_items = list(evaluation.rglob("*"))
    actual_spool_items = list(spool.rglob("*"))
    for item in actual_evaluation_items:
        _require_safe_descendant(item, evaluation, field="final evaluation entry")
        require_no_link_components(item, field="final evaluation entry")
    for item in actual_spool_items:
        _require_safe_descendant(item, spool, field="final spool entry")
        require_no_link_components(item, field="final spool entry")

    def snapshot(items: list[Path], root: Path) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for item in items:
            relative = item.relative_to(root).as_posix()
            if item.is_dir():
                result[relative] = {"type": "directory"}
            elif item.is_file():
                result[relative] = {
                    "type": "file",
                    "size": item.stat().st_size,
                    "sha256": sha256_file(item),
                }
            else:
                raise StreamingIntegrityError("final inventory contains a special file")
        return result

    actual_evaluation = snapshot(actual_evaluation_items, evaluation)
    actual_spool = snapshot(actual_spool_items, spool)
    if {
        path: descriptor["type"] for path, descriptor in actual_evaluation.items()
    } != expected_evaluation:
        raise StreamingIntegrityError("final evaluation entry inventory changed")
    if {path: descriptor["type"] for path, descriptor in actual_spool.items()} != expected_spool:
        raise StreamingIntegrityError("final spool entry inventory changed")
    return actual_evaluation, actual_spool


def _path(value: Path) -> str:
    return str(
        require_no_link_components(value, field="orchestrator input").resolve(
            strict=True
        )
    )


def run_streaming_evaluation(
    runtime_root: Path,
    code_state_root: Path,
    plan_root: Path,
    spool_root: Path,
    output_parent: Path,
    *,
    poll_seconds: float = 0.1,
    rank_device: str = "cpu",
    process_data_root: Path | None = None,
) -> Path:
    """Run two isolated workers; this parent opens neither arrays nor labels."""

    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    if rank_device not in {"cpu", "cuda"}:
        raise ValueError("rank_device must be 'cpu' or 'cuda'")
    verified_plan_root = require_no_link_components(plan_root, field="rank plan").resolve(
        strict=True
    )
    plan = load_json(verified_plan_root / "rank_plan.json")
    require_hashed_json(
        plan,
        hash_field="rank_plan_sha256",
        schema=PLAN_SCHEMA,
        status="rank_state_frozen",
        field="orchestrated rank plan",
    )
    plan_sha256 = str(plan["rank_plan_sha256"])
    output = reject_unsafe_output_path(Path(output_parent), field="streaming metric output")
    output_root = reject_unsafe_output_path(
        output / f"metrics-{plan_sha256[:16]}",
        field="streaming metric output root",
    )
    require_disjoint_paths(
        output_root,
        {
            "runtime": Path(runtime_root),
            "packed code state": Path(code_state_root),
            "rank plan": Path(plan_root),
            "streaming spool": Path(spool_root),
        },
        field="streaming metric output",
    )
    spool = prepare_spool_root(
        spool_root,
        forbidden={
            "runtime": Path(runtime_root),
            "packed code state": Path(code_state_root),
            "rank plan": Path(plan_root),
            "metric output": output_root,
        },
    )
    output.mkdir(parents=True, exist_ok=True)
    state_common = [
        "--code-state",
        _path(code_state_root),
        "--plan",
        _path(plan_root),
        "--spool",
        str(spool),
        "--poll-seconds",
        str(poll_seconds),
    ]
    metric_runtime = ["--runtime", _path(runtime_root)]
    if process_data_root is not None:
        metric_runtime.extend(["--process-data-root", _path(process_data_root)])
    metric_command = [
        sys.executable,
        "-m",
        "raw_rebuilt_streaming",
        "metric-worker",
        *state_common,
        *metric_runtime,
        "--output-parent",
        str(output),
        "--serve",
    ]
    rank_command = [
        sys.executable,
        "-m",
        "raw_rebuilt_streaming",
        "rank-worker",
        *state_common,
        "--rank-device",
        rank_device,
        "--serve",
    ]
    metric = None
    rank = None
    try:
        rank_seal = spool / "rank_evidence_complete.json"
        if not rank_seal.exists():
            rank = _start_worker(rank_command)
            while rank.poll() is None:
                time.sleep(min(1.0, max(0.05, poll_seconds)))
            if rank.returncode != 0:
                raise StreamingIntegrityError(f"rank worker exited with status {rank.returncode}")
        metric = _start_worker(metric_command)
        while metric.poll() is None:
            time.sleep(min(1.0, max(0.05, poll_seconds)))
        if metric.returncode != 0:
            raise StreamingIntegrityError(f"metric worker exited with status {metric.returncode}")
    except BaseException:
        for process in (metric, rank):
            if process is not None and process.poll() is None:
                process.terminate()
        for process in (metric, rank):
            if process is None:
                continue
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    evaluation = output / f"metrics-{plan_sha256[:16]}"
    _verify_evaluation_after_workers(evaluation, spool, plan)
    return evaluation


__all__ = ["run_streaming_evaluation"]
