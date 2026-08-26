"""Branch a shared indT warm-up into a fair posterior-loss weight sweep.

All branches restore the same encoder, training-only decoder, optimizer, and
random-number states after the base warm-up.  Formal query/database artifacts
are not accepted.  Results are development diagnostics only.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
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
from tools.dev_graded_semantic_hash_pilot import (
    HashSemanticDecoders,
    _balanced_bce,
    _balanced_graded_pair_loss,
    _evaluate,
    _posterior_soft_jaccard_loss,
    _straight_through_binary,
)
from tools.dev_rzcsd_longer_training_pilot import _epoch_order, _seed_everything
from tools.dev_semantic_codebook_pilot import _load_fit, _split


BITS = (16, 32, 64)


def _parse_weights(value: str) -> tuple[float, ...]:
    weights = tuple(sorted(set(float(item.strip()) for item in value.split(",") if item.strip())))
    if not weights or weights[0] < 0.0:
        raise argparse.ArgumentTypeError("weights must be a nonempty nonnegative CSV")
    return weights


def _weight_token(weight: float) -> str:
    return format(weight, ".8g").replace("-", "m").replace(".", "p")


def _train_epoch(
    model: RZCSD512,
    decoders: HashSemanticDecoders,
    optimizer: torch.optim.Optimizer,
    image: np.ndarray,
    text: np.ndarray,
    labels: np.ndarray,
    identity_ids: np.ndarray,
    fit: np.ndarray,
    positive_weight: torch.Tensor,
    *,
    epoch: int,
    seed: int,
    device: torch.device,
    code_bce_weight: float,
    graded_weight: float,
    posterior_weight: float,
    auxiliary_scale: float,
) -> dict[str, float | int]:
    model.train()
    decoders.train()
    totals: dict[str, float] = {}
    examples = 0
    order = _epoch_order(fit, seed, epoch)
    batch_size = int(model.config.batch_size)
    for start in range(0, len(order), batch_size):
        index = order[start : start + batch_size]
        if len(index) < 2:
            continue
        image_batch = torch.from_numpy(image[index]).to(device)
        text_batch = torch.from_numpy(text[index]).to(device)
        label_batch = torch.from_numpy(labels[index]).to(device)
        optimizer.zero_grad(set_to_none=True)
        base = compute_training_objective(
            model,
            image_batch,
            text_batch,
            label_batch,
            identity_ids[index],
            positive_weight,
        )
        if auxiliary_scale == 0.0:
            code_bce = base["total"].new_zeros(())
            graded = base["total"].new_zeros(())
            posterior_jaccard = base["total"].new_zeros(())
            total = base["total"]
        else:
            image_output = model(image_batch, "image")
            text_output = model(text_batch, "text")
            image_ste = {
                bits: _straight_through_binary(image_output.continuous_codes[bits])
                for bits in BITS
            }
            text_ste = {
                bits: _straight_through_binary(text_output.continuous_codes[bits])
                for bits in BITS
            }
            image_logits = decoders(image_ste)
            text_logits = decoders(text_ste)
            code_bce = torch.stack(
                [
                    0.5
                    * (
                        _balanced_bce(image_logits[bits], label_batch, positive_weight)
                        + _balanced_bce(text_logits[bits], label_batch, positive_weight)
                    )
                    for bits in BITS
                ]
            ).mean()
            graded = torch.stack(
                [
                    _balanced_graded_pair_loss(image_ste[bits], text_ste[bits], label_batch)
                    for bits in BITS
                ]
            ).mean()
            posterior_jaccard = 0.5 * (
                _posterior_soft_jaccard_loss(image_output.posterior_heads, label_batch)
                + _posterior_soft_jaccard_loss(text_output.posterior_heads, label_batch)
            )
            total = base["total"] + auxiliary_scale * (
                code_bce_weight * code_bce
                + graded_weight * graded
                + posterior_weight * posterior_jaccard
            )
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(decoders.parameters()), 5.0
        )
        optimizer.step()
        values = {
            **base,
            "code_bce": code_bce,
            "graded": graded,
            "posterior_jaccard": posterior_jaccard,
            "augmented_total": total,
        }
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * len(index)
        examples += len(index)
    record: dict[str, float | int] = {
        "epoch": epoch + 1,
        "examples": examples,
        "auxiliary_scale": float(auxiliary_scale),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }
    record.update({name: value / examples for name, value in totals.items()})
    return record


def run(
    fit_root: Path,
    output_dir: Path,
    *,
    posterior_weights: tuple[float, ...],
    warmup_epochs: int,
    fine_tune_epochs: int,
    seed: int,
    code_bce_weight: float,
    graded_weight: float,
    fine_tune_learning_rate: float,
    device: torch.device,
) -> dict[str, object]:
    _seed_everything(seed)
    image64, text64, labels_u8, identity_ids, manifest = _load_fit(fit_root)
    image = np.asarray(image64, dtype=np.float32)
    text = np.asarray(text64, dtype=np.float32)
    labels = labels_u8.astype(np.float32)
    fit, query, database = _split(identity_ids)
    total_epochs = warmup_epochs + fine_tune_epochs
    config = replace(FROZEN_CONFIG, seed=seed, epochs=total_epochs)
    model = RZCSD512(label_dim=labels.shape[1], config=config).to(device)
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    decoders = HashSemanticDecoders(label_dim=labels.shape[1]).to(device)
    torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    positive_weight = configure_training_label_prior(model, labels_u8[fit]).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(decoders.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    warmup_history = []
    for epoch in range(warmup_epochs):
        record = _train_epoch(
            model,
            decoders,
            optimizer,
            image,
            text,
            labels,
            identity_ids,
            fit,
            positive_weight,
            epoch=epoch,
            seed=seed,
            device=device,
            code_bce_weight=code_bce_weight,
            graded_weight=graded_weight,
            posterior_weight=0.0,
            auxiliary_scale=0.0,
        )
        warmup_history.append(record)
        print(json.dumps({"stage": "warmup", **record}, sort_keys=True), flush=True)
    warmup_evaluation = _evaluate(model, image, text, labels_u8, query, database, device)
    base_state = {
        "model": copy.deepcopy(model.state_dict()),
        "decoders": copy.deepcopy(decoders.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "cpu_rng": torch.random.get_rng_state().clone(),
        "cuda_rng": [value.clone() for value in torch.cuda.get_rng_state_all()]
        if device.type == "cuda"
        else None,
    }
    branches = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for posterior_weight in posterior_weights:
        branch_model = RZCSD512(label_dim=labels.shape[1], config=config).to(device)
        branch_decoders = HashSemanticDecoders(label_dim=labels.shape[1]).to(device)
        branch_model.load_state_dict(base_state["model"], strict=True)
        branch_decoders.load_state_dict(base_state["decoders"], strict=True)
        branch_optimizer = torch.optim.AdamW(
            list(branch_model.parameters()) + list(branch_decoders.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        branch_optimizer.load_state_dict(base_state["optimizer"])
        for group in branch_optimizer.param_groups:
            group["lr"] = fine_tune_learning_rate
        torch.random.set_rng_state(base_state["cpu_rng"])
        if base_state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(base_state["cuda_rng"])
        history = []
        for offset in range(fine_tune_epochs):
            epoch = warmup_epochs + offset
            scale = min(1.0, (offset + 1) / max(fine_tune_epochs, 1))
            record = _train_epoch(
                branch_model,
                branch_decoders,
                branch_optimizer,
                image,
                text,
                labels,
                identity_ids,
                fit,
                positive_weight,
                epoch=epoch,
                seed=seed,
                device=device,
                code_bce_weight=code_bce_weight,
                graded_weight=graded_weight,
                posterior_weight=posterior_weight,
                auxiliary_scale=scale,
            )
            history.append(record)
            print(
                json.dumps(
                    {"stage": "branch", "posterior_weight": posterior_weight, **record},
                    sort_keys=True,
                ),
                flush=True,
            )
        evaluation = _evaluate(
            branch_model, image, text, labels_u8, query, database, device
        )
        checkpoint_name = f"posterior_weight_{_weight_token(posterior_weight)}.pt"
        checkpoint_body = {
            "schema": "raw_rebuilt_posterior_weight_sweep_checkpoint_v1",
            "status": "DEVELOPMENT_ONLY_NOT_A_PAPER_CLAIM",
            "posterior_jaccard_weight": posterior_weight,
            "config": asdict(config),
            "method": {
                "code_bce_weight": code_bce_weight,
                "graded_weight": graded_weight,
                "posterior_jaccard_weight": posterior_weight,
                "warmup_epochs": warmup_epochs,
                "fine_tune_epochs": fine_tune_epochs,
                "fine_tune_learning_rate": fine_tune_learning_rate,
                "shared_branch_state": True,
            },
            "evaluation": evaluation,
        }
        checkpoint_result_sha256 = sha256_json(checkpoint_body)
        torch.save(
            {
                **checkpoint_body,
                "result_sha256": checkpoint_result_sha256,
                "model_state_dict": branch_model.state_dict(),
                "decoder_state_dict": branch_decoders.state_dict(),
            },
            output_dir / checkpoint_name,
        )
        branches.append(
            {
                "posterior_jaccard_weight": posterior_weight,
                "history": history,
                "evaluation": evaluation,
                "checkpoint": checkpoint_name,
                "checkpoint_result_sha256": checkpoint_result_sha256,
            }
        )
        print(
            json.dumps(
                {"stage": "branch_complete", "posterior_weight": posterior_weight, "evaluation": evaluation},
                sort_keys=True,
            ),
            flush=True,
        )
    body: dict[str, object] = {
        "schema": "raw_rebuilt_posterior_weight_sweep_indt_v1",
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
        "config": asdict(config),
        "method": {
            "posterior_jaccard_weights": list(posterior_weights),
            "code_bce_weight": code_bce_weight,
            "graded_weight": graded_weight,
            "warmup_epochs": warmup_epochs,
            "fine_tune_epochs": fine_tune_epochs,
            "fine_tune_learning_rate": fine_tune_learning_rate,
            "shared_model_decoder_optimizer_and_rng_branch_state": True,
        },
        "warmup_history": warmup_history,
        "warmup_evaluation": warmup_evaluation,
        "branches": branches,
    }
    result = {**body, "result_sha256": sha256_json(body)}
    (output_dir / "sweep.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--posterior-jaccard-weights", type=_parse_weights, default=(0.0, 0.005, 0.01, 0.02, 0.05)
    )
    parser.add_argument("--warmup-epochs", type=int, default=40)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--code-bce-weight", type=float, default=0.05)
    parser.add_argument("--graded-weight", type=float, default=0.10)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.warmup_epochs < 1 or args.fine_tune_epochs < 1:
        raise ValueError("warm-up and fine-tune epochs must be positive")
    if args.code_bce_weight < 0.0 or args.graded_weight < 0.0:
        raise ValueError("auxiliary weights must be nonnegative")
    if args.fine_tune_learning_rate <= 0.0:
        raise ValueError("fine-tune learning rate must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    result = run(
        args.fit.resolve(strict=True),
        args.output_dir.resolve(),
        posterior_weights=args.posterior_jaccard_weights,
        warmup_epochs=args.warmup_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        seed=args.seed,
        code_bce_weight=args.code_bce_weight,
        graded_weight=args.graded_weight,
        fine_tune_learning_rate=args.fine_tune_learning_rate,
        device=device,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "result_sha256": result["result_sha256"],
                "output": str(args.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
