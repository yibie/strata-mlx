---
id: e55
kind: experiment
status: measured
script: bench/experiments/e55_vision_integration.py
---
# Official encoder and MLX image receiving

Question: can the existing official standalone encoder and HTTP server drive image understanding in MLX?

Setup: M5 Pro 20 GPU cores,64GiB, SC117 abliterated Flash-Next IQ3_XXS eec4e10a2c29, matching BF16 projector, existing MTP; MLX0.32.3,float16,TF32 unset. Official Strata296b59d encoder/server unchanged. Real-model tests use the separate local integration with resident admission/fusion6,16384 context ceiling,512MiB/four-slot RAM parking,min0,expert profile off,keep-awake2s. These resident/fusion controls are outside the separate upstream-main vision contribution.

Protocol: synthetic OCR, colored counts, two-image reading, image follow-up, live/parked image reuse, HTTP disconnect and recovery, malformed JSON and text recovery. Normal service remains llama.cpp; scheduler/action locks and verified photo pause/drain protect every model window. Receiver adds512MiB reservation; header-only required bytes57,655,751,168 fit57,982,058,496 limit with512MiB parking.1GiB parking exceeds that limit, so it was refused before readiness. No allocation/wired limit was relaxed.

Pre-fix evidence:12/12 content and10/12 strict image outputs; Markdown fences on two count answers. Live reuse430 tokens passed. Parked reuse failed because parking the departing namespace could evict its destination. Destination selection now precedes eviction, with a one-entry-capacity regression test. HTTP disconnect yielded native DONE cancel after8 output tokens and subsequent image/text recovery passed. Zero new system swapouts in that window. Final validation:548 integrated tests/22 subtests and500 independent upstream-main tests passed.12/12 full-model image contents and10/12 strict; live and parked image reuse both430 tokens (0.129/0.131s), follow-up437 tokens. HTTP disconnect produced native cancel after8 tokens and recovered correctly.37 memory samples,zero new system swapouts. Original llama.cpp/text/OCR/health/viewer and unchanged aggregate checkpoints restored, all locks free. Earlier failed cache check and source/test artifacts retained separately. No universal performance or production-readiness claim.

Exact executed machine-specific clients/guards and raw results are retained under bench/results/e55-m5pro-20261017. The e55 client is the parameterized replay version; grading can be checked against retained rows. No private images, uploads or photo operations.

## Concepts
- [[image-understanding]]
- [[conversation-parking]]
