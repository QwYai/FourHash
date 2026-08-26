"""Validate the frozen NUS-selected code-length route on another indT split.

This command is a transfer check, not another selection sweep.  It always
trains the exact compact control and the exact posterior semantic-bridge model
under the frozen 40+5 schedule, then assembles 16=compact, 32=semantic bridge,
64=compact.  Results are recorded even if transfer regresses; they cannot
change the frozen route and no formal query/database artifact is accepted.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_neural.hash_anchors import HASH_ANCHOR_SPECS
from raw_rebuilt_runtime.contract import atomic_write_json, sha256_file, sha256_json
from tools.dev_rzcsd_architecture_sweep import _delta_report
from tools.dev_rzcsd_hash_anchor_sweep import CONTROL, _train_candidate
from tools.dev_semantic_codebook_pilot import _load_fit


ARCHITECTURE_FREEZE_FILE_SHA256 = (
    "15cb33030a3506c2980925ab929606a9864680078b927deb085dabe2b3ed1033"
)
WIDTH_ROUTER_RESULT_SHA256 = (
    "a115b62740d3d028c564ac4da981dacecbd52bce511fc397137929758f054674"
)
SEMANTIC_SPEC = next(
    spec for spec in HASH_ANCHOR_SPECS if spec.name == "posterior_semantic_anchor_g010"
)


def _transfer_split(
    identity_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reuse the registered hash split while allowing smaller transfer sets.

    The NUS selection helper requires at least 5,000 fitting rows.  MIRFlickr's
    entire indT artifact is smaller after its fixed query/database holdout, so
    transfer validation uses the identical ordering/count formula with a
    2,000-row fail-closed minimum.  This cannot affect the frozen NUS route.
    """

    identities = np.asarray(identity_ids)
    if identities.ndim != 1 or identities.dtype.kind not in "iu":
        raise ValueError("identity_ids must be a one-dimensional integer array")
    buckets = np.empty(len(identities), dtype=np.uint64)
    for index, identity in enumerate(identities.tolist()):
        payload = f"semantic-codebook-dev-v1:{identity}".encode("ascii")
        buckets[index] = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    order = np.argsort(buckets, kind="stable")
    query_count = min(1_500, max(500, len(order) // 12))
    database_count = min(4_500, max(1_500, len(order) // 4))
    query = np.sort(order[:query_count])
    database = np.sort(order[query_count : query_count + database_count])
    fit = np.sort(order[query_count + database_count :])
    if len(fit) < 2_000:
        raise ValueError("transfer-only internal fit split is too small")
    if (
        np.intersect1d(query, database).size
        or np.intersect1d(query, fit).size
        or np.intersect1d(database, fit).size
    ):
        raise AssertionError("transfer-only indT partitions overlap")
    return fit, query, database


def _assemble(
    compact: dict[str, Any], semantic: dict[str, Any]
) -> dict[str, Any]:
    return {
        "16": compact["16"],
        "32": semantic["32"],
        "64": compact["64"],
    }


def run(fit_root: Path, output_dir: Path, *, seed: int, device: torch.device) -> dict[str, Any]:
    image64, text64, labels_u8, identity_ids, manifest = _load_fit(fit_root)
    image = np.asarray(image64, dtype=np.float32)
    text = np.asarray(text64, dtype=np.float32)
    fit, query, database = _transfer_split(identity_ids)
    # Neither fixed candidate uses the CLIP-PCA branch.  The argument remains
    # present only because the shared candidate trainer supports all anchors.
    unused_pca_state = {
        "center": np.zeros(512, dtype=np.float32),
        "projection": np.zeros((512, 64), dtype=np.float32),
        "scale": np.ones(64, dtype=np.float32),
        "eigenvalues": np.empty(0, dtype=np.float64),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_records = []
    evaluations = {}
    for spec in (None, SEMANTIC_SPEC):
        record, state = _train_candidate(
            spec,
            unused_pca_state,
            image,
            text,
            labels_u8,
            identity_ids,
            fit,
            query,
            database,
            seed=seed,
            device=device,
        )
        name = CONTROL["name"] if spec is None else spec.name
        body = {
            "schema": "raw_rebuilt_rzcsd_frozen_route_transfer_candidate_indt_v1",
            "status": "DEVELOPMENT_TRANSFER_ONLY_NOT_A_PAPER_CLAIM",
            "dataset": manifest["dataset"],
            "source_seal_sha256": manifest["source_seal_sha256"],
            "fit_artifact_sha256": manifest["fit_artifact_sha256"],
            "formal_query_or_database_labels_opened": False,
            "labels_consumed": "indT_internal_fit_and_development_only",
            "architecture_freeze_file_sha256": ARCHITECTURE_FREEZE_FILE_SHA256,
            "width_router_result_sha256": WIDTH_ROUTER_RESULT_SHA256,
            "seed": seed,
            **record,
        }
        result = {**body, "result_sha256": sha256_json(body)}
        atomic_write_json(output_dir / f"{name}.json", result)
        checkpoint = output_dir / f"{name}.pt"
        torch.save(
            {
                "schema": body["schema"],
                "result_sha256": result["result_sha256"],
                "model_config": record["model_config"],
                "anchor_spec": record["anchor_spec"],
                **state,
            },
            checkpoint,
        )
        result["checkpoint"] = checkpoint.name
        result["checkpoint_sha256"] = sha256_file(checkpoint)
        candidate_records.append(result)
        evaluations[name] = record["evaluation"]
    compact = evaluations[CONTROL["name"]]
    semantic = evaluations[SEMANTIC_SPEC.name]
    assembled = _assemble(compact, semantic)
    delta = _delta_report(assembled, compact)
    body = {
        "schema": "raw_rebuilt_rzcsd_frozen_route_transfer_indt_v1",
        "status": "DEVELOPMENT_TRANSFER_ONLY_NOT_A_PAPER_CLAIM",
        "dataset": manifest["dataset"],
        "source_seal_sha256": manifest["source_seal_sha256"],
        "fit_artifact_sha256": manifest["fit_artifact_sha256"],
        "formal_query_or_database_labels_opened": False,
        "architecture_freeze_file_sha256": ARCHITECTURE_FREEZE_FILE_SHA256,
        "width_router_result_sha256": WIDTH_ROUTER_RESULT_SHA256,
        "frozen_routes": {
            "16": CONTROL["name"],
            "32": SEMANTIC_SPEC.name,
            "64": CONTROL["name"],
        },
        "configuration_changed_from_nuswide_freeze": False,
        "transfer_result_may_change_frozen_route": False,
        "seed": seed,
        "candidate_records": candidate_records,
        "assembled_evaluation": assembled,
        "assembled_delta_report": delta,
    }
    result = {**body, "result_sha256": sha256_json(body)}
    atomic_write_json(output_dir / "transfer.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    result = run(
        args.fit.resolve(strict=True),
        args.output_dir.resolve(),
        seed=args.seed,
        device=torch.device(args.device),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset": result["dataset"],
                "delta_report": result["assembled_delta_report"],
                "result_sha256": result["result_sha256"],
                "output": str(args.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
