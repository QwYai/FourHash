"""Label-free producer for the fully sealed Hamming-evidence stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .codes import CodeState, open_code_state
from .integrity import StreamingIntegrityError
from .plan import ChunkSpec, FrozenRankPlan, open_rank_plan
from .protocol import (
    bundle_path,
    discard_pending_bundle,
    evidence_seal_path,
    open_bundle,
    open_rank_evidence_seal,
    prepare_spool_root,
    publish_bundle,
    seal_rank_evidence,
)


_POPCOUNT = np.asarray([bin(value).count("1") for value in range(256)], dtype=np.uint8)


def _verify_rank_spool_inventory(spool: Path, plan: FrozenRankPlan) -> None:
    expected: set[str] = set()
    for chunk in plan.chunks():
        discard_pending_bundle(spool, chunk)
        root = bundle_path(spool, chunk)
        if root.exists():
            expected.update(
                {
                    (root / "bundle.json").relative_to(spool).as_posix(),
                    (root / "distances.npy").relative_to(spool).as_posix(),
                }
            )
    seal = evidence_seal_path(spool)
    if seal.exists():
        expected.add(seal.relative_to(spool).as_posix())
    actual = {
        item.relative_to(spool).as_posix()
        for item in spool.rglob("*")
        if item.is_file() or item.is_symlink()
    }
    if actual != expected:
        raise StreamingIntegrityError("rank spool contains files outside frozen evidence")


class _HammingBackend:
    """Distance-only backend; neither branch has a runtime/label interface."""

    def __init__(self, state: CodeState, device: str) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("rank_device must be 'cpu' or 'cuda'")
        self.state = state
        self.device = device
        self._database_cache: dict[tuple[str, int], Any] = {}
        self._torch: Any | None = None
        self._lut: Any | None = None
        if device == "cuda":
            # Deliberately lazy: importing the rank module on a CPU host must
            # not import torch or any training/runtime package.
            import torch

            if not torch.cuda.is_available():
                raise StreamingIntegrityError("CUDA rank backend requested but unavailable")
            self._torch = torch
            self._lut = torch.as_tensor(_POPCOUNT, dtype=torch.uint8, device="cuda")

    def _codes(self, chunk: ChunkSpec) -> tuple[np.ndarray, np.ndarray, str]:
        query_modality = "image" if chunk.direction == "i2t" else "text"
        database_modality = "text" if chunk.direction == "i2t" else "image"
        query_codes = np.asarray(
            self.state.arrays[("query", query_modality, chunk.bits)][
                chunk.start : chunk.end
            ],
            dtype=np.uint8,
        )
        database_codes = np.asarray(
            self.state.arrays[("database", database_modality, chunk.bits)],
            dtype=np.uint8,
        )
        return query_codes, database_codes, database_modality

    def distances(self, chunk: ChunkSpec) -> np.ndarray:
        query_codes, database_codes, database_modality = self._codes(chunk)
        result = np.empty((len(query_codes), len(database_codes)), dtype=np.uint8)
        if self.device == "cpu":
            for position, query in enumerate(query_codes):
                xor = np.bitwise_xor(database_codes, query[None, :])
                result[position] = _POPCOUNT[xor].sum(
                    axis=1, dtype=np.uint16
                ).astype(np.uint8, copy=False)
        else:
            torch = self._torch
            if torch is None or self._lut is None:  # pragma: no cover - constructor invariant
                raise AssertionError("CUDA backend is not initialized")
            key = (database_modality, chunk.bits)
            database_gpu = self._database_cache.get(key)
            if database_gpu is None:
                database_gpu = torch.as_tensor(
                    np.ascontiguousarray(database_codes), dtype=torch.uint8, device="cuda"
                )
                self._database_cache[key] = database_gpu
            query_gpu = torch.as_tensor(
                np.ascontiguousarray(query_codes), dtype=torch.uint8, device="cuda"
            )
            gpu_rows = []
            for position in range(len(query_codes)):
                xor = torch.bitwise_xor(database_gpu, query_gpu[position].unsqueeze(0))
                distances = self._lut[xor.to(torch.int64)].sum(dim=1)
                gpu_rows.append(distances.to(dtype=torch.uint8))
            if gpu_rows:
                result[:] = torch.stack(gpu_rows, dim=0).cpu().numpy()
        if np.any(result > chunk.bits):
            raise AssertionError("packed-code Hamming distance exceeds code width")
        return result


def hamming_distance_chunk(
    state: CodeState,
    chunk: ChunkSpec,
    *,
    rank_device: str = "cpu",
) -> np.ndarray:
    return _HammingBackend(state, rank_device).distances(chunk)


def _preflight_rank_frontier(
    spool: Path,
    plan: FrozenRankPlan,
    state: CodeState,
) -> int:
    if (spool / "acks").exists() and any((spool / "acks").rglob("*.json")):
        raise StreamingIntegrityError("rank process refuses a spool after metrics began")
    committed = 0
    missing_seen = False
    for chunk in plan.chunks():
        bundle = bundle_path(spool, chunk)
        if not bundle.exists():
            missing_seen = True
            continue
        if missing_seen:
            raise StreamingIntegrityError("rank bundle frontier has a hole before later evidence")
        evidence = open_bundle(spool, plan, state, chunk)
        evidence.close()
        committed += 1
    return committed


def produce_rank_bundles(
    code_state_root: Path,
    plan_root: Path,
    spool_root: Path,
    *,
    max_new_bundles: int | None = None,
    rank_device: str = "cpu",
) -> dict[str, Any]:
    """Produce a contiguous evidence frontier and seal it only when complete."""

    if max_new_bundles is not None and max_new_bundles < 0:
        raise ValueError("max_new_bundles must be nonnegative")
    if rank_device not in {"cpu", "cuda"}:
        raise ValueError("rank_device must be 'cpu' or 'cuda'")
    state = open_code_state(code_state_root)
    try:
        plan = open_rank_plan(plan_root, code_state_root)
        spool = prepare_spool_root(
            spool_root,
            forbidden={
                "packed code state": Path(code_state_root),
                "rank plan": Path(plan_root),
            },
        )
        total = sum(1 for _ in plan.chunks())
        _verify_rank_spool_inventory(spool, plan)
        if evidence_seal_path(spool).exists():
            seal = open_rank_evidence_seal(spool, plan, state)
            for chunk in plan.chunks():
                evidence = open_bundle(spool, plan, state, chunk)
                evidence.close()
            return {
                "status": "COMPLETE",
                "total_chunks": total,
                "frozen_chunks": int(seal["total_chunks"]),
                "new_bundles": 0,
                "rank_worker_loaded_labels": False,
                "rank_device": rank_device,
            }
        committed = _preflight_rank_frontier(spool, plan, state)
        produced = 0
        backend = _HammingBackend(state, rank_device)
        for chunk in list(plan.chunks())[committed:]:
            if max_new_bundles is not None and produced >= max_new_bundles:
                break
            distances = backend.distances(chunk)
            publish_bundle(spool, plan, state, chunk, distances)
            produced += 1
            committed += 1
        if committed == total:
            seal_rank_evidence(spool, plan, state)
        return {
            "status": "COMPLETE" if committed == total else "IN_PROGRESS",
            "total_chunks": total,
            "frozen_chunks": committed,
            "new_bundles": produced,
            "rank_worker_loaded_labels": False,
            "rank_device": rank_device,
        }
    finally:
        state.close()


def serve_rank_worker(
    code_state_root: Path,
    plan_root: Path,
    spool_root: Path,
    *,
    poll_seconds: float = 0.1,
    rank_device: str = "cpu",
) -> dict[str, Any]:
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    return produce_rank_bundles(
        code_state_root, plan_root, spool_root, rank_device=rank_device
    )


__all__ = ["hamming_distance_chunk", "produce_rank_bundles", "serve_rank_worker"]
