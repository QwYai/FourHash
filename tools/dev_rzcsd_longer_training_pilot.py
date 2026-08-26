"""IndT-only 40-vs-80 epoch diagnostic for the existing RZ-CSD backbone."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_runtime.contract import numeric_sha256, sha256_json
from rz_csd_clip512 import (
    FROZEN_CONFIG,
    RZCSD512,
    compute_training_objective,
    configure_training_label_prior,
)
from tools.dev_semantic_codebook_pilot import _expected_metrics, _hamming, _load_fit, _split


def _seed_everything(seed: int) -> None:
    expected = ":4096:8"
    observed = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if observed not in (None, expected):
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG conflicts with deterministic pilot")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _epoch_order(indices: np.ndarray, seed: int, epoch: int) -> np.ndarray:
    payload = f"raw-rebuilt-neural-epoch-v1:{seed}:{epoch}".encode("ascii")
    epoch_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF
    return indices[np.random.default_rng(epoch_seed).permutation(len(indices))]


@torch.no_grad()
def _encode(model: RZCSD512, features: np.ndarray, modality: str, device: torch.device) -> dict[int, np.ndarray]:
    model.eval()
    collected = {bits: [] for bits in (16, 32, 64)}
    for start in range(0, len(features), 512):
        tensor = torch.from_numpy(np.asarray(features[start : start + 512], dtype=np.float32)).to(device)
        output = model(tensor, modality)
        for bits in collected:
            collected[bits].append(output.continuous_codes[bits].cpu().numpy())
    return {bits: np.concatenate(blocks, axis=0) for bits, blocks in collected.items()}


def _evaluate(
    model: RZCSD512,
    image: np.ndarray,
    text: np.ndarray,
    labels: np.ndarray,
    fit: np.ndarray,
    query: np.ndarray,
    database: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    fit_image = _encode(model, image[fit], "image", device)
    fit_text = _encode(model, text[fit], "text", device)
    query_image = _encode(model, image[query], "image", device)
    query_text = _encode(model, text[query], "text", device)
    database_image = _encode(model, image[database], "image", device)
    database_text = _encode(model, text[database], "text", device)
    result = {}
    for bits in (16, 32, 64):
        threshold = np.median(
            np.concatenate((fit_image[bits], fit_text[bits]), axis=0), axis=0
        )
        variants = {}
        for name, offset in (("formal_zero", 0.0), ("fit_median_diagnostic", threshold)):
            i2t = _expected_metrics(
                _hamming(query_image[bits] - offset, database_text[bits] - offset),
                labels[query],
                labels[database],
            )
            t2i = _expected_metrics(
                _hamming(query_text[bits] - offset, database_image[bits] - offset),
                labels[query],
                labels[database],
            )
            variants[name] = {
                "i2t": i2t,
                "t2i": t2i,
                "mean_map": 0.5 * (i2t["map_expected_ties"] + t2i["map_expected_ties"]),
            }
        result[str(bits)] = variants
    return result


def run(
    fit_root: Path,
    output: Path,
    *,
    epochs: int,
    eval_epochs: tuple[int, ...],
    seed: int,
    device: torch.device,
) -> dict:
    _seed_everything(seed)
    image64, text64, labels_u8, identity_ids, manifest = _load_fit(fit_root)
    image = np.asarray(image64, dtype=np.float32)
    text = np.asarray(text64, dtype=np.float32)
    labels = labels_u8.astype(np.float32)
    fit, query, database = _split(identity_ids)
    config = replace(FROZEN_CONFIG, seed=seed, epochs=epochs)
    model = RZCSD512(label_dim=labels.shape[1], config=config).to(device)
    positive_weight = configure_training_label_prior(model, labels_u8[fit]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history = []
    evaluations = {}
    for epoch in range(epochs):
        model.train()
        order = _epoch_order(fit, seed, epoch)
        totals: dict[str, float] = {}
        examples = 0
        for start in range(0, len(order), config.batch_size):
            index = order[start : start + config.batch_size]
            if len(index) < 2:
                continue
            image_batch = torch.from_numpy(image[index]).to(device)
            text_batch = torch.from_numpy(text[index]).to(device)
            label_batch = torch.from_numpy(labels[index]).to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = compute_training_objective(
                model,
                image_batch,
                text_batch,
                label_batch,
                identity_ids[index],
                positive_weight,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * len(index)
            examples += len(index)
        record = {"epoch": epoch + 1, "examples": examples}
        record.update({name: value / examples for name, value in totals.items()})
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if epoch + 1 in eval_epochs:
            evaluations[str(epoch + 1)] = _evaluate(
                model, image, text, labels_u8, fit, query, database, device
            )
            print(
                json.dumps(
                    {"epoch": epoch + 1, "evaluation": evaluations[str(epoch + 1)]},
                    sort_keys=True,
                ),
                flush=True,
            )
    config_body = asdict(config)
    body = {
        "schema": "raw_rebuilt_rzcsd_indt_longer_training_pilot_v1",
        "status": "DEVELOPMENT_ONLY_NOT_A_PAPER_CLAIM",
        "dataset": manifest["dataset"],
        "source_seal_sha256": manifest["source_seal_sha256"],
        "fit_artifact_sha256": manifest["fit_artifact_sha256"],
        "formal_query_or_database_labels_opened": False,
        "labels_consumed": "indT_internal_fit_and_development_only",
        "split": {"fit": len(fit), "query": len(query), "database": len(database)},
        "split_hashes": {
            "fit_identity_sha256": numeric_sha256(identity_ids[fit]),
            "query_identity_sha256": numeric_sha256(identity_ids[query]),
            "database_identity_sha256": numeric_sha256(identity_ids[database]),
        },
        "config": config_body,
        "config_sha256": sha256_json(config_body),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "evaluations": evaluations,
    }
    result = {**body, "result_sha256": sha256_json(body)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.save(
        {
            "schema": result["schema"],
            "result_sha256": result["result_sha256"],
            "config": config_body,
            "model_state_dict": model.state_dict(),
        },
        output.with_suffix(".pt"),
    )
    return result


def _parse_epochs(value: str) -> tuple[int, ...]:
    result = tuple(sorted(set(int(item) for item in value.split(","))))
    if not result or result[0] < 1:
        raise argparse.ArgumentTypeError("eval epochs must be positive CSV integers")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--eval-epochs", type=_parse_epochs, default=(40, 80))
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.epochs < max(args.eval_epochs):
        raise ValueError("epochs must reach every requested evaluation epoch")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    result = run(
        args.fit.resolve(strict=True),
        args.output.resolve(),
        epochs=args.epochs,
        eval_epochs=args.eval_epochs,
        seed=args.seed,
        device=device,
    )
    print(json.dumps({"status": result["status"], "evaluations": result["evaluations"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
