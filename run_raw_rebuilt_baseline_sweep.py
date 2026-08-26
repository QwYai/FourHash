"""Resume-safe server driver for the sealed fixed-CLIP512 baseline matrix.

The driver deliberately delegates all data-boundary and artifact verification
to ``raw_rebuilt_baselines``.  It never opens labels, features, checkpoints, or
codes itself; it only launches the package CLI and verifies the resulting JSON
manifests before advancing to the next registered cell.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS: Mapping[str, tuple[str, str]] = {
    "mirflickr": ("mirflickr", "mirflickr_sealed"),
    "nuswide": ("nuswide", "nuswide_sealed"),
    "mscoco": ("mscoco", "mscoco_sealed"),
}
METHODS = ("ucch-f", "dcmh-f-seminit", "cirh-f")
BITS = (16, 32, 64)
SEEDS = (20260822, 20260823, 20260824)


class SweepError(RuntimeError):
    """Raised when a subprocess or sealed artifact differs from the request."""


def _csv_strings(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one comma-separated value is required")
    return result


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SweepError(f"cannot read verified JSON artifact {path}") from error
    if not isinstance(value, dict):
        raise SweepError(f"JSON artifact is not an object: {path}")
    return value


def _last_json_object(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise SweepError("baseline CLI did not emit a final JSON object")


def _run(command: Sequence[str], log_path: Path) -> dict[str, object]:
    completed = subprocess.run(
        list(command),
        check=False,
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(completed.stdout)
    if completed.returncode != 0:
        raise SweepError(
            f"command failed with status {completed.returncode}; see {log_path}"
        )
    return _last_json_object(completed.stdout)


def _resolve_fit(fit_base: Path, dataset_dir: str) -> Path:
    parent = (fit_base / dataset_dir).resolve(strict=True)
    candidates = sorted(
        path for path in parent.glob("fit-*") if path.is_dir() and (path / "manifest.json").is_file()
    )
    if len(candidates) != 1:
        raise SweepError(
            f"expected exactly one sealed fit artifact below {parent}, found {len(candidates)}"
        )
    return candidates[0]


def _require_registered(
    datasets: Iterable[str], methods: Iterable[str], bits: Iterable[int], seeds: Iterable[int]
) -> None:
    unknown_datasets = set(datasets) - set(DATASETS)
    unknown_methods = set(methods) - set(METHODS)
    unknown_bits = set(bits) - set(BITS)
    unknown_seeds = set(seeds) - set(SEEDS)
    if unknown_datasets or unknown_methods or unknown_bits or unknown_seeds:
        raise SweepError(
            "unregistered sweep cell: "
            f"datasets={sorted(unknown_datasets)}, methods={sorted(unknown_methods)}, "
            f"bits={sorted(unknown_bits)}, seeds={sorted(unknown_seeds)}"
        )


def _verify_checkpoint(
    root: Path, *, dataset: str, method: str, bits: int, seed: int
) -> None:
    manifest = _load_json(root / "manifest.json")
    binding = manifest.get("dataset_binding")
    if not isinstance(binding, dict):
        raise SweepError("checkpoint dataset binding is missing")
    expected = {
        "status": "FINAL_EPOCH_FROZEN",
        "method": method,
        "bits": bits,
        "seed": seed,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise SweepError(f"checkpoint {key} differs from requested cell")
    if binding.get("dataset") != dataset:
        raise SweepError("checkpoint dataset differs from requested cell")
    if manifest.get("checkpoint_selection") != (
        "fixed final epoch; query/database labels inaccessible"
    ):
        raise SweepError("checkpoint selection boundary differs")


def _verify_codes(
    root: Path, *, dataset: str, method: str, bits: int, seed: int
) -> None:
    manifest = _load_json(root / "manifest.json")
    contract = manifest.get("rank_contract")
    if not isinstance(contract, dict):
        raise SweepError("code rank contract is missing")
    if manifest.get("status") != "rank_state_frozen" or manifest.get(
        "labels_loaded_during_freeze"
    ) is not False:
        raise SweepError("code artifact crossed the label-free boundary")
    expected = {"method": method, "bits": bits, "seed": seed}
    for key, value in expected.items():
        if contract.get(key) != value:
            raise SweepError(f"code artifact {key} differs from requested cell")
    # Dataset identity is carried indirectly by the checkpoint/fit/source
    # hashes.  The driver also records the requested dataset beside the sealed
    # artifact; the package loader performs the authoritative binding check.
    if not isinstance(contract.get("source_seal_sha256"), str):
        raise SweepError(f"code artifact for {dataset} has no source seal")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-base", type=Path, required=True)
    parser.add_argument("--runtime-base", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", type=_csv_strings, default=tuple(DATASETS))
    parser.add_argument("--methods", type=_csv_strings, default=METHODS)
    parser.add_argument("--bits", type=_csv_ints, default=BITS)
    parser.add_argument("--seeds", type=_csv_ints, default=SEEDS)
    # Keep the public run contract identical to BaselineRunConfig's registered
    # default.  ``auto`` resolves to CUDA on the experiment server, while an
    # explicit ``cuda`` string would create a second content-addressed contract
    # for numerically identical training.
    parser.add_argument("--device", default="auto")
    parser.add_argument("--encode-batch-size", type=int, default=4096)
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    parser.add_argument("--max-cells", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    datasets = tuple(args.datasets)
    methods = tuple(args.methods)
    bits_values = tuple(args.bits)
    seeds = tuple(args.seeds)
    _require_registered(datasets, methods, bits_values, seeds)
    if args.encode_batch_size < 1 or args.min_free_gib < 0:
        raise SweepError("batch size must be positive and free-space floor nonnegative")
    if args.max_cells is not None and args.max_cells < 1:
        raise SweepError("max-cells must be positive")

    fit_base = args.fit_base.expanduser().resolve(strict=True)
    runtime_base = args.runtime_base.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve(strict=False)
    checkpoints = output_root / "checkpoints"
    codes = output_root / "codes"
    logs = output_root / "logs"
    output_root.mkdir(parents=True, exist_ok=True)
    audit = output_root / "sweep_events.jsonl"
    completed_cells = 0

    for dataset in datasets:
        fit_dir_name, runtime_dir_name = DATASETS[dataset]
        fit = _resolve_fit(fit_base, fit_dir_name)
        runtime = (runtime_base / runtime_dir_name).resolve(strict=True)
        if not (runtime / "runtime_manifest.json").is_file():
            raise SweepError(f"sealed runtime manifest is missing: {runtime}")
        for method in methods:
            for bits in bits_values:
                for seed in seeds:
                    if args.max_cells is not None and completed_cells >= args.max_cells:
                        return 0
                    free = shutil.disk_usage(output_root).free
                    floor = int(args.min_free_gib * (1024**3))
                    if free < floor:
                        raise SweepError(
                            f"free-space floor reached: {free / (1024**3):.2f} GiB"
                        )
                    cell = f"{dataset}_{method}_b{bits}_s{seed}"
                    _append_jsonl(
                        audit,
                        {
                            "event": "cell_started",
                            "utc": _utc_now(),
                            "cell": cell,
                            "free_bytes": free,
                        },
                    )
                    train_command = [
                        sys.executable,
                        "-u",
                        "-m",
                        "raw_rebuilt_baselines",
                        "train",
                        "--fit-artifact",
                        str(fit),
                        "--runtime",
                        str(runtime),
                        "--method",
                        method,
                        "--bits",
                        str(bits),
                        "--seed",
                        str(seed),
                        "--device",
                        args.device,
                        "--output",
                        str(checkpoints),
                        "--quiet",
                    ]
                    trained = _run(train_command, logs / f"{cell}.train.log")
                    checkpoint = Path(str(trained.get("output", ""))).resolve(strict=True)
                    _verify_checkpoint(
                        checkpoint,
                        dataset=dataset,
                        method=method,
                        bits=bits,
                        seed=seed,
                    )
                    encode_command = [
                        sys.executable,
                        "-u",
                        "-m",
                        "raw_rebuilt_baselines",
                        "encode",
                        "--checkpoint",
                        str(checkpoint),
                        "--runtime",
                        str(runtime),
                        "--output",
                        str(codes),
                        "--device",
                        args.device,
                        "--batch-size",
                        str(args.encode_batch_size),
                    ]
                    encoded = _run(encode_command, logs / f"{cell}.encode.log")
                    code_root = Path(str(encoded.get("output", ""))).resolve(strict=True)
                    _verify_codes(
                        code_root,
                        dataset=dataset,
                        method=method,
                        bits=bits,
                        seed=seed,
                    )
                    _append_jsonl(
                        audit,
                        {
                            "event": "cell_complete",
                            "utc": _utc_now(),
                            "cell": cell,
                            "checkpoint": str(checkpoint),
                            "codes": str(code_root),
                            "free_bytes": shutil.disk_usage(output_root).free,
                        },
                    )
                    completed_cells += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
