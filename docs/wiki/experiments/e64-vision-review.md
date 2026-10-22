---
id: e64
script: bench/experiments/e64_vision_review.py
change: zpxxwxso
date: 2026-10-21
---
# e64: supported-file image qualification on the PR branch

> 中文摘要:维护者要求的独立 PR 分支验证:Q2_0、两张 1024×768 图、超过 2,048 个输入 token,
> 同一引擎中打乱顺序的图像与文本解码比较。17 次合成请求通过,并保留控制波动和适用范围。

## Question

Does the PR branch's image path work on supported Q2_0 past the 2,048-token indexer boundary,
and what is the bounded decode cost of an image in context?

## Setup

20-core M5 Pro, 64 GiB, MLX 0.32.3, float16, TF32 unset, fusion limit 6.
Exact PR core, without the separate resident integration. Standard GSQ revision ed59f92082b1,
`Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf`, matching BF16 projector,
existing MTP, official Strata 296b59d encoder/server. Paged mode, explicit 20 GB expert budget, 8,192 context ceiling, RAM parking off, encoder cap 1,024.
Two generated 1024×768 OCR fixtures, >2,048-token conversation and follow-up, then two shuffled
image/text/duplicate-text triplets on the same live engine. F_NOCACHE requested on both model shards
before measured requests; actual file-page residency is not measured and expert arrays stay warm.
Exact arguments, source hashes, request usage and per-request decode timings are retained in results.

## Result

### October 18, before the rebase

Full exact-branch Mac suite: 537 passed, 22 subtests, 41.38 s. All 17 synthetic requests passed.
The two-image prompt had 4,442 input tokens, the follow-up 4,482 (4,455 reused); both labels were correct.
Six matched requests produced the same strict 88-token array with zero reused tokens. Image decode:
46.7 / 61.3 tok/s; text: 67.0 / 63.9; duplicate control: 58.9 / 67.0. Zero new system swap-outs
in the sampled model window. Source/client hashes and raw rows are in `bench/results/e64-m5pro-20261018`.
An earlier independently tagged pilot remains local; it is not the matched comparison.

### October 21, rebased onto upstream 5fea2f2

Fresh exact-core suite: 543 tests and 22 subtests passed in 42.79 s. The unchanged client passed
17/17 requests, again at 4,442 / 4,482 input tokens (4,455 reused on follow-up). All six strict
88-token arrays had zero reused tokens. Image decode: 42.2 / 62.3 tok/s; text: 66.2 / 62.7;
duplicate control: 57.2 / 66.1. Zero new system swap-outs during tests and model validation.
Raw rows, hashes, configuration and limits: `bench/results/e64-m5pro-20261021`.
The client and original protocol identifier retain their bytes after renaming e56 to e64;
October 18 evidence is preserved separately. Cross-day timings do not establish a causal speed change.

## Conclusion

This comparison changes image presence and prompt positions/length as well as ImageRoPE work;
it cannot isolate ImageRoPE's kernel cost. The limited sample and control variation do not establish a stable percentage overhead.
This supplies supported-file evidence past the indexer boundary, not general image quality or context qualification.

## Concepts

- [[image-understanding]]
