#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${1:-/root/autodl-tmp/kbs_rz_csd_clip512_code}"
RUN_ROOT="${2:-/root/autodl-tmp/kbs_rz_csd_clip512_runs}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"

FIT_BASE="${RUN_ROOT}/neural/raw_rebuilt_v1/fit"
RUNTIME_BASE="${RUN_ROOT}/runtime/raw_rebuilt_v1"
OUTPUT_ROOT="${RUN_ROOT}/neural/raw_rebuilt_v1/baselines_raneh_20260828"

test -d "${CODE_ROOT}"
test -d "${FIT_BASE}"
test -d "${RUNTIME_BASE}"
cd "${CODE_ROOT}"

"${PYTHON}" -u run_raw_rebuilt_baseline_sweep.py \
  --fit-base "${FIT_BASE}" \
  --runtime-base "${RUNTIME_BASE}" \
  --output-root "${OUTPUT_ROOT}" \
  --datasets mirflickr,nuswide,mscoco \
  --methods raneh-f \
  --bits 16,32,64 \
  --seeds 20260822,20260823,20260824 \
  --device auto \
  --encode-batch-size 4096 \
  --min-free-gib 8
