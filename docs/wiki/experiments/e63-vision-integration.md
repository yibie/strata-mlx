---
id: e63
script: bench/experiments/e63_vision_integration.py
change: zpxxwxso
date: 2026-10-18
---
# e63: official encoder and MLX image receiving

> 中文摘要:复用官方编码器的图像嵌入和位置编码。早期完整模型证据来自独立 resident 集成;
> 本次按维护者要求,将 PR 分支和支持的量化文件单独验证。

## Question

Can the unchanged official standalone encoder and HTTP server drive image understanding through
this PR's MLX receiver, including image context past the indexer's 2,048-token boundary?

## Setup

20-core M5 Pro, 64 GiB, MLX 0.32.3, float16, TF32 unset, Metal debug layer disabled.
Official Strata 296b59d supplies the unchanged encoder/server and tokenizer.
The initial October 17 evidence used SC117's
`Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf`
(revision eec4e10a2c29), BF16 projector and MTP, in separate resident integration zmuonvlz.
It is historical evidence, not qualification of an upstream-supported file or this branch's serving profile.

## Result

Historical integration: 12/12 synthetic image contents, 10/12 strict outputs (two fenced counting answers).
Live and parked reuse retained 430 tokens; image follow-up retained 437. HTTP disconnect cancelled after
8 output tokens and subsequent image/text requests passed. A destination-eviction failure was fixed
and covered by a one-entry parking regression. The longest measured prompt was 846 tokens.
These results do not exercise the indexer's pooled image keys past 2,048 tokens.

Exact-branch Q2_0 follow-up is recorded separately in e64: 17/17 requests, with two-image context
at 4,442 and 4,482 tokens, 537 tests plus 22 subtests. Its native paged profile excludes the resident integration.

## Conclusion

Framing and tiny-model regressions are independent of historical full-model integration measurements.
No general visual-accuracy, long-context or speed conclusion follows from the historical rows.
The portable client and synthetic pixels are retained; machine orchestration remains in the author's local evidence.

## Concepts

- [[image-understanding]]
- [[conversation-parking]]
