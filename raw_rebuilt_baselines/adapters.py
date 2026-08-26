"""Thin array-level calls into the audited UCCH-F/DCMH-F/CIRH-F cores."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math
import os
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from encoders import cirh_feature, dcmh_feature, ucch_feature

from .contract import SUPPORTED_BITS, FitArtifact, validate_fit_artifact


METHODS = ("ucch-f", "dcmh-f-seminit", "cirh-f")
DEFAULT_SEEDS = (20_260_822, 20_260_823, 20_260_824)
METHOD_CLAIMS = {
    "ucch-f": (
        "controlled fixed-CLIP512 adaptation retaining the UCCH feature-mode "
        "MLPs, momentum contrastive memory, and cross-modal ranking loss; not "
        "an official end-to-end reproduction"
    ),
    "dcmh-f-seminit": (
        "controlled supervised fixed-CLIP512 adaptation retaining DCMH "
        "alternating pairwise/quantization/balance optimization with the "
        "documented train-label semantic warm start; not official DCMH"
    ),
    "cirh-f": (
        "controlled fixed-CLIP512 adaptation retaining CIRH collaborated "
        "similarity, reconstruction, graph mixing, and independent hash "
        "functions; not an official end-to-end reproduction"
    ),
}


@dataclass(frozen=True)
class BaselineRunConfig:
    method: str
    bits: int
    seed: int
    device: str = "auto"
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if self.bits not in SUPPORTED_BITS:
            raise ValueError(f"bits must be one of {SUPPORTED_BITS}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.device not in {"auto", "cpu", "cuda"} and not self.device.startswith(
            "cuda:"
        ):
            raise ValueError("device must be auto, cpu, cuda, or cuda:N")
        if not isinstance(self.overrides, Mapping):
            raise TypeError("overrides must be a mapping")
        forbidden = {"bits", "seed", "device"}.intersection(self.overrides)
        if forbidden:
            raise ValueError(
                "bits, seed, and device have dedicated frozen fields; "
                f"remove overrides {sorted(forbidden)}"
            )


@dataclass
class TrainedCore:
    method: str
    core_config: dict[str, Any]
    state_dicts: dict[str, dict[str, torch.Tensor]]
    history: list[dict[str, float]]
    resolved_device: str
    training_summary: dict[str, Any]
    deterministic_runtime: dict[str, Any]


def _core_config_class(method: str) -> type:
    if method == "ucch-f":
        return ucch_feature.UCCHConfig
    if method == "dcmh-f-seminit":
        return dcmh_feature.TrainConfig
    if method == "cirh-f":
        return cirh_feature.TrainConfig
    raise ValueError(f"unsupported method {method!r}")


def make_core_config(config: BaselineRunConfig) -> object:
    """Create one core config while preventing silent/unknown overrides."""

    config.validate()
    cls = _core_config_class(config.method)
    allowed = {item.name for item in fields(cls)} - {"bits", "seed", "device"}
    unknown = set(config.overrides) - allowed
    if unknown:
        raise ValueError(
            f"unknown {config.method} override fields: {sorted(unknown)}"
        )
    values = dict(config.overrides)
    values.update(bits=config.bits, seed=config.seed, device=config.device)
    core_config = cls(**values)
    # Core validation sometimes also needs n_train; it is repeated in the core.
    if config.method == "ucch-f":
        core_config.validate()
    elif config.method == "dcmh-f-seminit":
        if core_config.initialization != "semantic":
            raise ValueError(
                "the canonical DCMH baseline requires semantic initialization "
                "under dcmh-f-seminit; random-buffer "
                "initialization is not admitted under this reporting name"
            )
        core_config.validate()
    return core_config


def enable_strict_determinism(seed: int) -> dict[str, Any]:
    """Freeze PyTorch RNG/backend state before any baseline model is built."""

    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in (None, ":4096:8", ":16:8"):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG conflicts with deterministic execution"
        )
    if workspace is None:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            raise RuntimeError(
                "CUDA was initialized before CUBLAS_WORKSPACE_CONFIG was sealed"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    return {
        "seed": int(seed),
        "torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch_deterministic_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
            if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled")
            else False
        ),
        "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)),
        "cudnn_deterministic": bool(
            getattr(torch.backends.cudnn, "deterministic", False)
        ),
        "cuda_matmul_allow_tf32": bool(
            getattr(getattr(torch.backends, "cuda", object()), "matmul", object()).allow_tf32
            if hasattr(getattr(torch.backends, "cuda", object()), "matmul")
            else False
        ),
        "cudnn_allow_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
        "float32_matmul_precision": (
            torch.get_float32_matmul_precision()
            if hasattr(torch, "get_float32_matmul_precision")
            else "unavailable"
        ),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "torch_version": str(torch.__version__),
        "cuda_build": torch.version.cuda,
    }


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {requested}, but CUDA is unavailable")
    return requested


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def owned_float32_input(value: np.ndarray, *, field: str) -> np.ndarray:
    """Return an owned, writable C-order buffer before any Torch conversion.

    ``np.ascontiguousarray`` is insufficient for a read-only contiguous
    memmap because it may return the original view.  The explicit copy keeps
    the fixed-feature baselines from passing non-writable storage to legacy
    core routines that call ``torch.from_numpy``.
    """

    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 512:
        raise ValueError(f"{field} must have shape [N,512]")
    owned = np.array(array, dtype=np.float32, order="C", copy=True)
    if not owned.flags.owndata or not owned.flags.writeable or not owned.flags.c_contiguous:
        raise RuntimeError(f"{field} owned-buffer invariant failed")
    return owned


def train_core(
    fit: FitArtifact,
    config: BaselineRunConfig,
    *,
    verbose: bool = True,
) -> TrainedCore:
    """Call only the legacy-free train/encode cores, never their loaders/CLIs."""

    validate_fit_artifact(fit)
    core_config = make_core_config(config)
    deterministic = enable_strict_determinism(config.seed)
    image = owned_float32_input(fit.image, field="fit image features")
    text = owned_float32_input(fit.text, field="fit text features")

    if config.method == "ucch-f":
        result = ucch_feature.train_ucch_f(image, text, core_config, verbose=verbose)
        states = {
            "image": _cpu_state(result.image_model),
            "text": _cpu_state(result.text_model),
            "memory": _cpu_state(result.memory_bank),
        }
        summary = {
            "image_parameter_delta_l2": float(result.image_parameter_delta),
            "text_parameter_delta_l2": float(result.text_parameter_delta),
            "memory_delta_l2": float(result.memory_delta),
            "labels_passed_to_core": False,
        }
    elif config.method == "dcmh-f-seminit":
        # This is the sole supervised baseline.  ``fit.labels`` is exactly the
        # verified indT slice; no full-runtime/Q/D label handle exists here.
        result = dcmh_feature.train_dcmh_f(
            image,
            text,
            np.array(fit.labels, order="C", copy=True),
            core_config,
            verbose=verbose,
        )
        states = {
            "image": _cpu_state(result.image_model),
            "text": _cpu_state(result.text_model),
        }
        summary = {
            "image_parameter_delta_l2": float(result.image_parameter_delta),
            "text_parameter_delta_l2": float(result.text_parameter_delta),
            "initialization": result.initialization_metadata,
            "labels_passed_to_core": "indT_only",
        }
    else:
        result = cirh_feature.train_cirh_f(image, text, core_config, verbose=verbose)
        states = {
            "image": _cpu_state(result.image_model),
            "text": _cpu_state(result.text_model),
            "joint": _cpu_state(result.joint_model),
        }
        n_train = int(image.shape[0])
        summary = {
            "image_parameter_delta_l2": float(result.image_parameter_delta),
            "text_parameter_delta_l2": float(result.text_parameter_delta),
            "joint_parameter_delta_l2": float(result.joint_parameter_delta),
            "runtime_seconds": float(result.runtime_seconds),
            "graph_diagnostics": result.graph_diagnostics,
            "minimum_one_float32_train_square_bytes": int(4 * n_train * n_train),
            "labels_passed_to_core": False,
        }

    history = [
        {str(key): float(value) for key, value in row.items()}
        for row in result.history
    ]
    if not history or not all(
        math.isfinite(value) for row in history for value in row.values()
    ):
        raise RuntimeError("baseline core returned empty or non-finite history")
    summary.update(
        {
            "claim_scope": METHOD_CLAIMS[config.method],
            "fit_interface": "raw_rebuilt_neural.FitArtifact (indT only)",
            "checkpoint_selection": "fixed final epoch; no query/database labels",
        }
    )
    return TrainedCore(
        method=config.method,
        core_config=asdict(core_config),
        state_dicts=states,
        history=history,
        resolved_device=str(result.device),
        training_summary=summary,
        deterministic_runtime=deterministic,
    )


def reconstruct_models(
    method: str,
    core_config: Mapping[str, Any],
    state_dicts: Mapping[str, Mapping[str, torch.Tensor]],
) -> tuple[nn.Module, nn.Module]:
    """Rebuild only the two label-free modality encoders from a checkpoint."""

    if method == "ucch-f":
        config = ucch_feature.UCCHConfig(**dict(core_config))
        image = ucch_feature.UCCHFeatureNet(
            512, config.bits, config.image_layers, config.hidden_width
        )
        text = ucch_feature.UCCHFeatureNet(
            512, config.bits, config.text_layers, config.hidden_width
        )
    elif method == "dcmh-f-seminit":
        config = dcmh_feature.TrainConfig(**dict(core_config))
        image = dcmh_feature.FeatureHashNet(
            512, config.bits, config.hidden_dim, config.l2_normalize
        )
        text = dcmh_feature.FeatureHashNet(
            512, config.bits, config.hidden_dim, config.l2_normalize
        )
    elif method == "cirh-f":
        config = cirh_feature.TrainConfig(**dict(core_config))
        image = cirh_feature.ImageHashNet(config.bits, 512, config.image_hidden_dim)
        text = cirh_feature.TextHashNet(config.bits, 512)
    else:
        raise ValueError(f"unsupported method {method!r}")
    if set(state_dicts) not in (
        {"image", "text"},
        {"image", "text", "memory"},
        {"image", "text", "joint"},
    ):
        raise RuntimeError("checkpoint state inventory differs")
    image.load_state_dict(dict(state_dicts["image"]), strict=True)
    text.load_state_dict(dict(state_dicts["text"]), strict=True)
    return image, text


@torch.no_grad()
def encode_core(
    method: str,
    image_model: nn.Module,
    text_model: nn.Module,
    image_features: np.ndarray,
    text_features: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Label-free whole-dataset encoding through the retained core functions."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    resolved = resolve_device(device)
    image_model.to(resolved)
    text_model.to(resolved)
    if method == "ucch-f":
        encode = ucch_feature.encode_all
    elif method == "dcmh-f-seminit":
        encode = dcmh_feature.encode_all
    elif method == "cirh-f":
        encode = cirh_feature.encode_all
    else:
        raise ValueError(f"unsupported method {method!r}")
    image_owned = owned_float32_input(
        image_features, field="rank image features"
    )
    image_codes = encode(image_model, image_owned, batch_size, resolved)
    del image_owned
    text_owned = owned_float32_input(text_features, field="rank text features")
    text_codes = encode(text_model, text_owned, batch_size, resolved)
    del text_owned
    return (
        np.ascontiguousarray(image_codes, dtype=np.int8),
        np.ascontiguousarray(text_codes, dtype=np.int8),
    )


__all__ = [
    "BaselineRunConfig",
    "DEFAULT_SEEDS",
    "METHODS",
    "METHOD_CLAIMS",
    "TrainedCore",
    "enable_strict_determinism",
    "encode_core",
    "make_core_config",
    "owned_float32_input",
    "reconstruct_models",
    "resolve_device",
    "train_core",
]
