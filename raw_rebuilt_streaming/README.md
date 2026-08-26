# Raw-rebuilt packed-code streaming evaluation

`raw_rebuilt_streaming` is an independent evaluation package. It does not
modify `raw_rebuilt_neural`, `raw_rebuilt_runtime`, or `rz_csd_clip512.py`.

## Security and metric contract

- `freeze-code` is the only neural encoding phase. It verifies the v1
  checkpoint and current v1 code inventory, opens the dedicated label-free
  runtime reader, and makes an explicit writable C-contiguous copy of every
  read-only feature slice before inference.
- A frozen code state contains only query/database packed `uint8` binary codes,
  the available widths, and source/row/split/checkpoint/code hashes. It contains
  no features, continuous outputs, labels, label paths, or legacy MAT data.
- `import-baseline-code` accepts only a complete
  `raw_rebuilt_fixed_feature_baseline_codes_v1` directory and its matching
  checkpoint. It verifies the artifact manifest, every array, the current
  baseline code inventory, bipolar `{-1,+1}` values, and all source/row/split
  hashes before producing the same packed evaluator state. Naked arrays are
  not an input interface.
- Evaluation is strictly two-stage. The rank process first freezes every
  planned `uint8[query_chunk,D]` Hamming-distance artifact and a complete
  evidence seal, then exits. Only after that exit does the metric process open
  query/database labels. The rank CLI has no runtime or label-path argument and
  its CPU import graph does not load Torch, the v1 runtime, neural training, or
  metric modules.
- Metric ACKs contain only an opaque SHA-256 commitment. Label-derived metrics
  stay in the metric output tree. After the private commit and opaque ACK are
  both verified, the corresponding distance bundle is deleted. Resume covers
  the single legal crash window between those two commits.
- Full-gallery metrics are exact expected-tie mAP, P/R, binary nDCG, and
  ground-truth soft-Jaccard graded J-NDCG@50/100/1000. The database order is the
  frozen `indD` order; no `Q x D` rank permutation is stored.
- NUS-WIDE fails closed unless every bound artifact uses TC21.

## Minimal neural-model sequence

Run from the project directory containing this package. Each command writes
one JSON object to stdout, which is convenient for a detached driver.

```bash
python -m raw_rebuilt_streaming freeze-code \
  --runtime /path/to/runtime/dataset-sealed \
  --checkpoint /path/to/v1/checkpoint \
  --output-parent /path/to/streaming/code \
  --device cuda
```

Use the returned `code_state` path:

```bash
python -m raw_rebuilt_streaming freeze-plan \
  --code-state /path/to/code-state-... \
  --output-parent /path/to/streaming/plans \
  --bits 16,32,64 \
  --directions i2t,t2i \
  --query-chunk-size 8 \
  --cutoffs 50,100,1000
```

Then run the two stages through the coordinator:

```bash
python -m raw_rebuilt_streaming stream-evaluate \
  --runtime /path/to/runtime/dataset-sealed \
  --code-state /path/to/code-state-... \
  --plan /path/to/plan-... \
  --spool /path/to/streaming/spool \
  --output-parent /path/to/streaming/metrics \
  --rank-device cuda
```

`--rank-device cpu` is the default. CUDA is used only for XOR/popcount distance
calculation and still emits the identical `uint8` evidence format. Start with a
MIRFlickr pilot; use CUDA for NUS-WIDE and MS-COCO after byte-equivalence and
throughput are confirmed on the target server.

## Minimal baseline sequence

One baseline code artifact contains one width. Import it without exposing raw
arrays:

```bash
python -m raw_rebuilt_streaming import-baseline-code \
  --artifact /path/to/ucch-f-b16-s1-codes-... \
  --checkpoint /path/to/ucch-f-b16-s1-checkpoint-... \
  --output-parent /path/to/streaming/baseline-code
```

Freeze a matching plan with `--bits 16`, then use the same `stream-evaluate`
command. Repeat for each method, seed, and width. An unavailable width is
rejected instead of silently synthesized.

The registered multi-cell driver keeps dataset identity in every event and
accepts `mirflickr`, `nuswide`, or `mscoco` explicitly:

```bash
python tools/run_baseline_streaming_eval_sweep.py \
  --dataset nuswide \
  --source-events /path/to/training/sweep_events.jsonl \
  --runtime /path/to/runtime/nuswide_sealed \
  --output-root /path/to/baseline-evaluation \
  --seeds 20260822 --rank-device cuda

python tools/aggregate_baseline_streaming_eval.py \
  --dataset nuswide \
  --event-log /path/to/baseline-evaluation/evaluation_events.jsonl \
  --output-root /path/to/baseline-evaluation \
  --seeds 20260822 \
  --json-output /new/path/aggregate.json \
  --csv-output /new/path/metrics.csv
```

Choose an output filesystem whose free space exceeds both the configured
safety floor and one cell's transient distance footprint.  Distance bundles
are retained while the label-side worker consumes them and are reduced to
small committed receipts only after both private metrics and opaque ACKs
verify.

## Explicit worker sequence and resume

For a driver that wants visible stage boundaries, run:

```bash
python -m raw_rebuilt_streaming rank-worker \
  --code-state /path/to/code-state-... \
  --plan /path/to/plan-... \
  --spool /path/to/spool \
  --rank-device cuda --serve

python -m raw_rebuilt_streaming metric-worker \
  --runtime /path/to/runtime/dataset-sealed \
  --code-state /path/to/code-state-... \
  --plan /path/to/plan-... \
  --spool /path/to/spool \
  --output-parent /path/to/metrics --serve
```

Do not launch the metric command until the rank command exits successfully.
Do not reuse a spool or output directory for another plan. All inputs, spool,
and output roots must be mutually non-overlapping. Content hashes, contiguous
frontiers, exact file inventories, and non-symlink path checks make resume
fail closed on poison, replay, holes, or rebinding.

## Evidence storage

Six cells require exactly `6 * Q * D` distance bytes, plus small `.npy` and JSON
overhead. Representative peaks per seed are approximately 0.22 GiB for
MIRFlickr (`2243 x 17772`), 2.26 GiB for NUS-WIDE (`2085 x 193749`), and
3.28 GiB for MS-COCO (`5000 x 117218`). Metric consumption deletes bundles
incrementally after private commit and opaque ACK.
