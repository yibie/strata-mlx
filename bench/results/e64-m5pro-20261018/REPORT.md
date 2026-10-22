# e64: PR13 supported-file review, October 18, 2026

Exact PR branch snapshot `1dcd1512d29b88892b98170947b05cea1cfeafe8`, change `ununvnvw`.
At measurement time, every `strata_mlx/*.py` hash matched the October 18 PR tree and prior head `7072a8e2`;
no resident/startup integration code was used. The executed benchmark hash is in provenance.json.
October 21 rebase onto upstream 5fea2f2 preserves new upstream submission and dense-kernel work.
These October 18 full-model rows predate that rebase and are not measurements of the rebased engine.
The benchmark file is renamed from e56 to e64 without changing its bytes; original metadata/protocol labels stay unchanged.

## Setup

M5 Pro, 20 GPU cores, 64 GiB; MLX 0.32.3, float16, TF32 unset, Metal debug layer off.
Standard ISTA-DASLab GSQ model revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`:
`Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf`, with its second shard.
Exact file sizes and prior SHA256 verification are in provenance.json; model files were not rehashed during measurement.
Existing BF16 projector (SC117 eec4e10a2c29), existing raw MTP files loaded by the PR's standard loader,
and unchanged official Strata 296b59d HTTP server/standalone GPU encoder/tokenizer.
The original Strata language-model engine was not used.

Native settings: `--expert-memory 20` (decimal GB, paged), `--vision`, `--fusion-limit 6`,
`--max-context 8192`, `--conversation-cache-mib 0`, `--expert-profile off`, `--keep-awake 2`;
other defaults unchanged. Receiver reserves 512 MiB. Encoder cap 1,024 image tokens, loopback HTTP.
Full config is in provenance.json. No allocation or global wired limit was raised.
Two generated 1024×768 images contain STRATA 731 and MAPLE 482.
F_NOCACHE was requested on both shards before measured requests; actual file-page residency was not measured.
The live model, expert arrays, encoder cache and compiled kernels remained warm.

## Results

- Full exact-branch Mac suite: **537 passed, 22 subtests passed**, 41.38 s.
- **17/17 requests complete and correct**: two OCR requests, the long two-image request and follow-up,
  six matched decode requests, six intervening READY requests and one final text recovery.
- Two-image prompt: **4,442 input tokens**, correct ordered labels from the pixels; the answer is not supplied in the prompt.
- Follow-up: **4,482 input tokens**, correct second image label; **4,455 cached tokens** from live state.
- All six decode requests produce the identical, strictly graded 88-token JSON array of integers 0–31.
  Same conversation within each triplet, with only the image attachment differing. RAM parking is off;
  an unrelated request replaces live state before each measurement. All six report **0 reused tokens**.
- **0 new system swap-out bytes** across the measured model window, including startup. System sampling
  and machine restoration diagnostics remain in the author's local evidence rather than this portable contribution.

| Triplet | Image present, tok/s | Text only, tok/s | Duplicate text control, tok/s |
| --- | ---: | ---: | ---: |
| 0 | 46.7 | 67.0 | 58.9 |
| 1 | 61.3 | 63.9 | 67.0 |

These are the server's decode timings, excluding prompt reading; raw token counts, milliseconds,
draft acceptance, usage, stop reasons and wall times are retained. Image prompts have 840 tokens;
text/control prompts have 70. Each triplet's order is shuffled using seed 56013.
The task is deliberately low entropy and uses MTP; these are not representative chat-speed numbers.
Images were slower in these samples, but two triplets and the control variation do not establish
an isolated ImageRoPE cost or a stable percentage overhead. Compilation/cache warmup can affect early samples.
An earlier pilot with independently tagged system prefixes was retained locally and is not the matched table above.

## Conclusion and reproduction

This supplies exact-branch Q2_0 evidence beyond the 2,048-token indexer boundary and bounded matched decode data.
It does not qualify other quantizations, arbitrary long image conversations or general visual accuracy.
The historical unsupported abliterated IQ3_XXS integration remains separately labelled in e63.

Prepare the unchanged official encoder/server and a hand-written config (provenance.json gives the tested settings),
then run `bench/experiments/e64_vision_review.py --official-root ... --config ... --out ...` with the official
server's Python environment including Pillow and psutil. The caller must own workload isolation, the memory guard,
and restoration; the client does not manage production services. It creates only synthetic image fixtures.
