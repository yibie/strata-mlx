# Historical e63 image integration — October 17, 2026

These results use separate resident integration `zmuonvlz`, not the standalone PR branch.
The exact unsupported file is
`Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf`,
SC117 revision `eec4e10a2c29b440a2ae85591fa11569ce752963`, with its BF16 projector and existing MTP.
This historical integration evidence does not establish support for that quantization in this repository.

M5 Pro, 20 GPU cores, 64 GiB, MLX 0.32.3, float16, TF32 unset.
Profile: separate resident experts, fusion 6, 16,384 context, 512 MiB/four-slot RAM parking,
expert profiling off, keep-awake 2 s. Official Strata 296b59d HTTP server and GPU encoder,
1,024 image-token cap; loopback HTTP. Fresh process with existing file caches, not cold SSD.

## Measured results

The final run's `api/*.jsonl` and `summary.json` are retained here with synthetic fixture pixels.
12/12 image outputs had correct contents and stopped normally. 10/12 matched strict format;
the two counting outputs added Markdown fences. Ordered two-image reading and image follow-up passed.
Live and parked image reuse retained 430 tokens; the follow-up retained 437. An HTTP disconnect
cancelled after 8 tokens and subsequent image/text requests recovered. The longest prompt was
846 tokens, below the indexer's 2,048-token boundary. There were zero new system swapouts in
37 final-window memory samples. Integrated tests: 548 plus 22 subtests; separate upstream tiny-model suite: 500.

A pre-fix parked-image attempt failed because parking the departing namespace evicted its destination.
The final implementation takes the destination first and includes a one-entry-capacity regression.
Earlier attempts and full machine diagnostics are preserved in the author's external project evidence;
they are not counted as successful final measurements or included in this portable evidence folder.

## Limits

These synthetic observations qualify only the stated integration profile. They establish neither
standalone PR serving, supported-file behavior, indexer-length image context nor general visual accuracy.
Exact-branch Q2_0 follow-up is a separate e64 experiment and results directory.
