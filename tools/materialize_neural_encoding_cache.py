"""Materialize a verified label-free neural encoding cache without ranking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from raw_rebuilt_neural.integrity import reject_unsafe_output_path
from raw_rebuilt_neural.ranking import ensure_encoding_cache
from raw_rebuilt_neural.training import load_trained_checkpoint
from raw_rebuilt_runtime import load_label_free_rank_inputs
from raw_rebuilt_runtime.contract import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    rank = load_label_free_rank_inputs(args.runtime)
    if rank.labels_loaded_during_freeze is not False:
        raise RuntimeError("label-free runtime crossed the metric boundary")
    checkpoint = load_trained_checkpoint(
        args.checkpoint,
        device=args.device,
        expected_source_seal_sha256=rank.source_seal_sha256,
    )
    output_root = reject_unsafe_output_path(args.output_root, field="encoding-cache output")
    output_root.mkdir(parents=True, exist_ok=True)
    cache = ensure_encoding_cache(rank, checkpoint, output_root)
    try:
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "cache_root": str(cache.root),
                    "manifest_sha256": sha256_file(cache.root / "manifest.json"),
                    "source_seal_sha256": rank.source_seal_sha256,
                    "checkpoint_sha256": checkpoint.checkpoint_sha256,
                    "rows": int(rank.row_ids.shape[0]),
                    "labels_loaded_during_freeze": rank.labels_loaded_during_freeze,
                },
                sort_keys=True,
            )
        )
    finally:
        cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
