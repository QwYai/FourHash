"""IndT-only pilot for graded, directly decodable RZ-CSD hash codes.

This development screen keeps the released RZ-CSD encoder intact and adds two
train-only objectives: every bit-width must decode the multi-label target, and
cross-modal code similarity must reproduce label-set Jaccard overlap.  The
formal query/database labels are never opened.
"""

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
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_runtime.contract import numeric_sha256, sha256_json
from rz_csd_clip512 import (
    BITS,
    FROZEN_CONFIG,
    RZCSD512,
    compute_training_objective,
    configure_training_label_prior,
)
from tools.dev_rzcsd_longer_training_pilot import _encode, _epoch_order, _seed_everything
from tools.dev_semantic_codebook_pilot import _expected_metrics, _hamming, _load_fit, _split


class HashSemanticDecoders(nn.Module):
    """Small auxiliary heads used only to make every code width semantic."""

    def __init__(self, label_dim: int) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                str(bits): nn.Sequential(
                    nn.LayerNorm(bits),
                    nn.Linear(bits, label_dim),
                )
                for bits in BITS
            }
        )

    def forward(self, codes: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        return {bits: self.heads[str(bits)](codes[bits]) for bits in BITS}


def _balanced_bce(logits: torch.Tensor, labels: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


def _straight_through_binary(code: torch.Tensor) -> torch.Tensor:
    binary = torch.where(code >= 0.0, torch.ones_like(code), -torch.ones_like(code))
    return code + (binary - code).detach()


def _balanced_graded_pair_loss(
    image_code: torch.Tensor,
    text_code: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Regress cross-modal cosine to multi-label Jaccard with balanced strata."""

    intersection = labels @ labels.T
    cardinality = labels.sum(dim=1)
    union = cardinality[:, None] + cardinality[None, :] - intersection
    target = intersection / union.clamp_min(1.0)
    prediction = 0.5 * (
        F.normalize(image_code, dim=1, eps=1.0e-8)
        @ F.normalize(text_code, dim=1, eps=1.0e-8).T
        + 1.0
    )
    element = F.smooth_l1_loss(prediction, target, reduction="none", beta=0.10)
    positive = intersection > 0.0
    negative = ~positive
    parts = []
    if bool(positive.any().item()):
        parts.append(element[positive].mean())
    if bool(negative.any().item()):
        parts.append(element[negative].mean())
    return torch.stack(parts).mean()


def _posterior_soft_jaccard_loss(
    posterior_heads: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    target = labels[:, None, :].expand_as(posterior_heads)
    intersection = (posterior_heads * target).sum(dim=2)
    union = (posterior_heads + target - posterior_heads * target).sum(dim=2)
    return (1.0 - intersection / union.clamp_min(1.0e-6)).mean()


def _evaluate(
    model: RZCSD512,
    image: np.ndarray,
    text: np.ndarray,
    labels: np.ndarray,
    query: np.ndarray,
    database: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    query_image = _encode(model, image[query], "image", device)
    query_text = _encode(model, text[query], "text", device)
    database_image = _encode(model, image[database], "image", device)
    database_text = _encode(model, text[database], "text", device)
    result = {}
    for bits in BITS:
        i2t = _expected_metrics(
            _hamming(query_image[bits], database_text[bits]),
            labels[query],
            labels[database],
        )
        t2i = _expected_metrics(
            _hamming(query_text[bits], database_image[bits]),
            labels[query],
            labels[database],
        )
        result[str(bits)] = {
            "i2t": i2t,
            "t2i": t2i,
            "mean_map": 0.5 * (i2t["map_expected_ties"] + t2i["map_expected_ties"]),
            "mean_ndcg_at_50": 0.5
            * (i2t["ndcg_at_50_expected_ties"] + t2i["ndcg_at_50_expected_ties"]),
        }
    return result


def run(
    fit_root: Path,
    output: Path,
    *,
    epochs: int,
    eval_epochs: tuple[int, ...],
    seed: int,
    code_bce_weight: float,
    graded_weight: float,
    posterior_jaccard_weight: float,
    warmup_epochs: int,
    fine_tune_learning_rate: float,
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
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    decoders = HashSemanticDecoders(label_dim=labels.shape[1]).to(device)
    # Auxiliary-head initialization must not perturb the backbone's dropout
    # stream during the base warm-up, so that epoch-warmup is a fair control.
    torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    positive_weight = configure_training_label_prior(model, labels_u8[fit]).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(decoders.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history = []
    evaluations = {}
    for epoch in range(epochs):
        if epoch == warmup_epochs and warmup_epochs > 0:
            for group in optimizer.param_groups:
                group["lr"] = float(fine_tune_learning_rate)
        model.train()
        decoders.train()
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
            base = compute_training_objective(
                model,
                image_batch,
                text_batch,
                label_batch,
                identity_ids[index],
                positive_weight,
            )
            if epoch < warmup_epochs:
                code_bce = base["total"].new_zeros(())
                graded = base["total"].new_zeros(())
                posterior_jaccard = base["total"].new_zeros(())
                auxiliary_scale = 0.0
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
                        _balanced_graded_pair_loss(
                            image_ste[bits],
                            text_ste[bits],
                            label_batch,
                        )
                        for bits in BITS
                    ]
                ).mean()
                posterior_jaccard = 0.5 * (
                    _posterior_soft_jaccard_loss(
                        image_output.posterior_heads, label_batch
                    )
                    + _posterior_soft_jaccard_loss(
                        text_output.posterior_heads, label_batch
                    )
                )
                auxiliary_scale = min(1.0, (epoch - warmup_epochs + 1) / 5.0)
                total = base["total"] + auxiliary_scale * (
                    code_bce_weight * code_bce
                    + graded_weight * graded
                    + posterior_jaccard_weight * posterior_jaccard
                )
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(decoders.parameters()), 5.0
            )
            optimizer.step()
            batch_values = {
                **base,
                "code_bce": code_bce,
                "graded": graded,
                "posterior_jaccard": posterior_jaccard,
                "augmented_total": total,
            }
            for name, value in batch_values.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * len(index)
            examples += len(index)
        record = {
            "epoch": epoch + 1,
            "examples": examples,
            "auxiliary_scale": 0.0
            if epoch < warmup_epochs
            else min(1.0, (epoch - warmup_epochs + 1) / 5.0),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        record.update({name: value / examples for name, value in totals.items()})
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if epoch + 1 in eval_epochs:
            evaluations[str(epoch + 1)] = _evaluate(
                model, image, text, labels_u8, query, database, device
            )
            print(
                json.dumps(
                    {"epoch": epoch + 1, "evaluation": evaluations[str(epoch + 1)]},
                    sort_keys=True,
                ),
                flush=True,
            )
    config_body = asdict(config)
    method = {
        "code_bce_weight": float(code_bce_weight),
        "graded_weight": float(graded_weight),
        "posterior_jaccard_weight": float(posterior_jaccard_weight),
        "hash_semantic_decoder": "LayerNorm-Linear-per-width-train-only",
        "graded_target": "balanced cross-modal label-set Jaccard SmoothL1",
        "discrete_training": "straight-through bipolar codes",
        "warmup_epochs": int(warmup_epochs),
        "auxiliary_ramp_epochs": 5,
        "fine_tune_learning_rate": float(fine_tune_learning_rate),
    }
    body = {
        "schema": "raw_rebuilt_graded_semantic_hash_indt_pilot_v4_posterior_curriculum",
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
        "method": method,
        "method_sha256": sha256_json(method),
        "encoder_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_only_decoder_parameter_count": sum(
            parameter.numel() for parameter in decoders.parameters()
        ),
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
            "method": method,
            "model_state_dict": model.state_dict(),
            "decoder_state_dict": decoders.state_dict(),
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
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--eval-epochs", type=_parse_epochs, default=(20, 40))
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--code-bce-weight", type=float, default=0.35)
    parser.add_argument("--graded-weight", type=float, default=0.25)
    parser.add_argument("--posterior-jaccard-weight", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.epochs < max(args.eval_epochs):
        raise ValueError("epochs must reach every requested evaluation epoch")
    if (
        args.code_bce_weight < 0.0
        or args.graded_weight < 0.0
        or args.posterior_jaccard_weight < 0.0
    ):
        raise ValueError("loss weights must be nonnegative")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        raise ValueError("warmup epochs must lie in [0, epochs)")
    if args.fine_tune_learning_rate <= 0.0:
        raise ValueError("fine-tune learning rate must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    result = run(
        args.fit.resolve(strict=True),
        args.output.resolve(),
        epochs=args.epochs,
        eval_epochs=args.eval_epochs,
        seed=args.seed,
        code_bce_weight=args.code_bce_weight,
        graded_weight=args.graded_weight,
        posterior_jaccard_weight=args.posterior_jaccard_weight,
        warmup_epochs=args.warmup_epochs,
        fine_tune_learning_rate=args.fine_tune_learning_rate,
        device=device,
    )
    print(json.dumps({"status": result["status"], "evaluations": result["evaluations"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
