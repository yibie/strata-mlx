# Strata-MLX image understanding — October 17, 2026

Image understanding is implemented and passed bounded synthetic tests through the unchanged official Strata HTTP server and standalone encoder. Production/photos remain on llama.cpp. Nothing was pushed or posted, and PR11 remains unchanged.

## Separate contributions and preserved baseline

- `contributions/strata-vision`, bookmark `codex/strata-vision`, jj change `zmuonvlz`: integration on the existing local serving bookmark `b4b1a2ce` / `rwsvpsnr`. The original latency checkout and PR11 checkout have clean, unchanged working trees.
- `contributions/strata-vision-upstream`, bookmark `codex/strata-vision-upstream`, jj change `zpxxwxso`: independent vision contribution directly on upstream main `1fca8253df55decab9d0bfa803bc34525fe402da`. It excludes PR11's benchmark/fusion controls and the local resident-loader changes. Its full tiny-model suite was tested separately; full-model measurements below used the local integration.
- Official source inspected: `reference/upstream/strata-official-296b59d`, exact revision `296b59d2f73c8f04a805df29c2fa97655adace78`. This includes `tools/vision/strata_vision.cpp`, `serve/server.py`, `metal/src/engine/window.cpp`, `src/program/generate.cpp`, and `include/strata/kernels/mrope.hpp`. The existing Mac encoder/server were reused without source changes. Both projects are MIT licensed; `THIRD_PARTY_NOTICES.md` preserves official copyright and license text.

Earlier text comparisons remain machine/profile-specific: our resident MLX lane beat the tested official streaming configuration. The official resident model mapping exceeded this GPU's single-buffer limit. These results do not prove MLX universally wins.

## Implementation and complexity

This was a moderate, cross-cutting receiver port, rather than simply accepting an image file. No new vision encoder or HTTP implementation was needed.

1. Opt-in `--vision` and `GENI` support accept concatenated official SVE1 records, with existing sampling/checkpoint keys and HTTP `img_keys` compatibility. Validation checks little-endian headers, rectangular grids, model embedding width, exact image-pad coverage, complete finite float32 payloads, and a 128MiB aggregate payload cap before mutating engine state. Actual embedding hashes determine identity; supplied hashes are not trusted.
2. Projected image rows replace token embeddings in plain/fused prompt reads, chunked and layer-major reads, and the draft head's shifted following-token inputs. PLE/n-gram hashing retains the original token IDs.
3. Interleaved M-RoPE maps physical cells to `(t,h,w)`. Image rows share time and vary by grid row/column; subsequent text starts at the image base plus `max(nx,ny)`. Rotary pairs select axes by `pair % 3`, including sparse-indexer pooled keys/queries, trunk attention, verification and MTP. Causal masks retain physical cell indices.
4. Live and parked state carry an image-sequence identity: image location, grid and SHA256 of projected rows. Different images sharing token IDs or filenames cannot reuse stale state. The conservative whole-sequence namespace intentionally forgoes shared-prefix reuse when adding/changing images. Image session SAVE is refused; normal text sessions are unchanged and conversation parking remains in RAM.
5. Cancellation retains existing chunk/piece rollback rules with request image state. A real interruption trial caught an eviction-order error: parking the departing namespace could evict the destination before selection. The final code takes/copies the destination first, with a regression test whose budget holds only one conversation.

The receiver reserves 512MiB before expert allocation; local resident admission also includes it in working space. Header-only arithmetic gives a 57,982,058,496-byte process allowance. With 512MiB parking the estimated requirement is 57,655,751,168 bytes; with 1GiB parking it is 58,192,622,080 bytes, exceeding admission. Initial 1GiB HTTP attempts did not reach READY; the bounded 512MiB profile was used without relaxing limits. The external encoder consumes separate memory, so combined behavior was measured rather than inferred from MLX counters.

## Qualification

Machine: M5 Pro, 20 GPU cores, 64GiB unified memory. Model: existing SC117 abliterated Flash-Next GSQ IQ3_XXS, revision `eec4e10a2c29b440a2ae85591fa11569ce752963`, matching BF16 projector, existing BF16 MTP files internally quantized by the existing loader. MLX 0.32.3; trunk float16; `MLX_ENABLE_TF32` unset during full-model serving. The test suite independently uses its existing float32 precision setting.

Full-model profile: resident experts, fusion6, context ceiling16384, 512MiB/four-slot/min0-token RAM parking, expert profiling off, keep-awake2s. Official GPU encoder cap1024 image tokens. HTTP bound only to `127.0.0.1:12346`. Synthetic fixtures only. Existing weights and OS file caches were retained: fresh processes are not cold-SSD trials. The encoder performs its existing startup warmup; repeated image timings also include encoder-cache effects.

Final guarded run: `vision-20261017T142545Z`.

| Check | Result |
| --- | --- |
| Integrated full Mac suite | 548 passed, 22 subtests passed |
| Independent upstream-main full Mac suite | 500 passed |
| Image contents | 12/12 correct; all completed with `stop` |
| Strict formatting | 10/12; both counting answers added Markdown fences |
| Different-image OCR / colored counts | Passed across two rounds |
| Two images in one request | Correct ordered JSON |
| Image follow-up | Correct; 437 cached tokens |
| Live image reuse | 430 cached tokens, 0.129s whole request |
| Image reuse after an intervening text request | 430 cached tokens, 0.131s whole request |
| Unreused reuse-prime request | 0 cached tokens, 0.868s whole request |
| HTTP disconnect during image generation | Native `DONE ... cancel` after 8 output tokens; subsequent image request correct with 430 cached tokens |
| Text after images / malformed JSON / recovery | READY, HTTP400, READY |
| Continuous system memory samples | 37 across startup and requests |
| New system swapouts | 0 bytes in the final model window |
| HTTP startup to healthy | 24.11s in this fresh-process, existing-file-cache window |

Cached and unreused timings are separate observations, not an equal-work causal speed comparison. Draft acceptance is retained in native frames and HTTP timings. Tiny-model tests additionally cover plain/fused sparse image positions, shifted MTP embeddings, changed image bytes at the same path, rejected files preserving state, prefill cancellation, layer-major reading and piece rollback. Both wiki graphs pass with zero errors/warnings; fatal lint and whitespace checks pass. The parameterized e55 replay grader matches the retained synthetic outputs and distinguishes boolean values from integer counts.

## Evidence and earlier attempts

`evidence/strata-vision-20261017/` retains all six windows, including failures:

- `141636Z`: 545 tests/22 subtests passed, one generated graph check failed from a self-referential changing commit hash. Fixed the evidence reference; production restored.
- `141820Z`: 547/22 passed; 1GiB HTTP profile did not reach READY; production restored.
- `142006Z`: upstream-only 499 passed; the same unadmitted HTTP profile failed; production restored.
- `142140Z`: 8/8 image contents, 6/8 strict, two-image/follow-up and text/error recovery passed on 512MiB parking; production restored.
- `142345Z`: 12/12 contents, 10/12 strict and HTTP cancellation passed; parked-image reuse failed and led to the eviction-order fix. This failed cache check is retained rather than counted as a pass.
- `142545Z`: final results above, with both suites and the cache regression fixed.

Exact source archives, the wiki overlays applied before final documentation tests, original executed machine-specific scripts, source-module hashes, raw native frames, synthetic outputs, configs, memory samples, reports and checksums are preserved. The `e55_vision_integration.py` client is the parameterized replay version; its original executed client is retained separately. Results and scripts also accompany each contribution under `bench/results/e55-m5pro-20261017` with experiment/concept/log links. All80 transferred raw artifacts/archive hashes were verified against a fresh Mac manifest. Original source-module hashes match the local integration and upstream checkouts. No private photo inputs or descriptions were read or retained.

## Restoration and limits

Final guard completed at14:28:16UTC (9:28:16AM CDT); independent read-only verification at14:28:43UTC confirmed production health, context131072, unchanged launcher/plist hashes, disabled serving logs, the expected live photo viewer, non-busy action status, unchanged aggregate checkpoints, no experimental children, and free scheduler/action/experiment locks. Text and synthetic OCR restoration checks passed. No runtime symlink, production flags, global wired allowance, schedule or photo checkpoint was changed. The Mac-side handover receives this qualified result, not a runtime cutover.

This establishes experimental image understanding and bounded reuse/cancellation behavior for these synthetic cases. It does not qualify private-photo classification, arbitrary16K/131K multimodal inputs, long-running or concurrent serving, broad visual reasoning, all image formats/video, or performance superiority. No production promotion or public posting/push is authorized or performed.
