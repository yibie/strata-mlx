# e64: supported-file rebase validation, October 21, 2026

PR change `zpxxwxso`, tested snapshot `2411255c15d4005326c6bae82f22f38934b327be`,
rebased onto upstream `5fea2f2833ff0033bf6349f17f1589585035de81`.
All published `strata_mlx/*.py` and client hashes match the executed snapshot; later edits record evidence.

## Setup

20-core M5 Pro, 64 GiB; MLX 0.32.3, float16, TF32 unset, Metal debug layer off.
Standard `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` revision ed59f92082b1,
`Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` plus its second shard.
Exact sizes and previously verified SHA256 pins are in provenance.json; no model rehash during measurement.
Matching BF16 projector, existing raw MTP, unchanged official Strata 296b59d HTTP server/encoder/tokenizer.
Paged expert budget 20 decimal GB, fusion 6, context 8,192, RAM parking off, encoder cap 1,024.
No resident-loader integration. Native arguments and source/fixture/font hashes are retained.

Two generated 1024×768 OCR fixtures. Same engine/weights, six shuffled decode arms in two matched triplets,
with unrelated READY requests replacing live state between arms. F_NOCACHE requested on both model shards
before each measured request; actual page residency is unmeasured. Expert arrays, encoder cache and compiled
kernels stay warm. Selected daily serving and photo inference were paused; no other model used the GPU.

## Results

Full Mac suite: **543 tests and 22 subtests passed**, 42.79 s.
**17/17 requests completed correctly**, including two-image input at 4,442 tokens and follow-up at 4,482
(4,455 reused). Six matched arrays each contain the same 88 output tokens and zero reused prompt tokens.
**Zero new system swap-out bytes** during tests and model validation, including startup.
Machine diagnostics and recovery evidence remain local.

| Triplet | Image, decode tok/s | Text | Duplicate text control |
| --- | ---: | ---: | ---: |
| 0 | 42.2 | 66.2 | 57.2 |
| 1 | 62.3 | 62.7 | 66.1 |

Image prompts contain 840 tokens versus 70 for text/control; this changes positions and prompt length.
These server decode timings exclude prompt reading; full per-request wall time, usage and draft acceptance
remain in api.jsonl. This low-entropy MTP task is not representative chat throughput. Two triplets, warmup
and duplicate-control variation do not isolate ImageRoPE cost or establish a stable overhead percentage.
Do not infer a before/after speed improvement by comparing this day to October 18.

## Conclusion and reproduction

The receiver remains correct under this bounded supported-file test after upstream's submission/dense-kernel
changes. It covers the indexer's 2,048-token boundary, not general visual accuracy or arbitrary long context.
Older unsupported IQ3_XXS integration evidence remains separately labelled e63.
Run `bench/experiments/e64_vision_review.py --official-root ... --config ... --out ...` with the official
server's Python, Pillow and psutil. The caller owns workload isolation, memory guard and restoration.
