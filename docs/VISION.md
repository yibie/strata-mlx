# Experimental image receiving path

Use with the unchanged official Strata HTTP server and standalone `strata-vision`
encoder pinned at Niko1221/strata 296b59d. Both projects are MIT licensed; the
adapted protocol and positional rules are credited in THIRD_PARTY_NOTICES.md.

Add `--vision` to this engine's existing native serving arguments. Keep the
server's existing `vision` config (matching text GGUF and BF16 mmproj), tokenizer,
and HTTP image handling. Set `exe` to an executable wrapper for this checkout's
`python -m strata_mlx.serve`. Bind experimental HTTP to loopback only.
No Python image encoder, new model weights, HTTP fork, or production change is required.

`GENI <max_new> [sampling/checkpoint keys] <sve-file> <id,id,...>` accepts
concatenated SVE1 little-endian float32 records from the official encoder. Width,
rectangular grids, payload length, finite values and exact pad coverage are
validated before mutating state. Encoded payloads are capped at 128MiB per request. `img_keys` is accepted for protocol compatibility;
identity is computed from actual embedding bytes instead of trusting supplied keys.
Prompt IDs remain unchanged for the PLE n-gram hashes. Image rows replace input
embeddings in plain/fused/chunked/layer-major reads and shifted draft-head inputs.
Sparse query and pooled-key rotary phases use physical cell to (t,h,w) mapping;
causal visibility continues to use physical token indices.

Caches remain in RAM. A request may reuse state only when the entire sequence of
(image position, grid, SHA256 of embedding rows) matches. Different images can be
parked independently, including when their text/pad token IDs match. This safely
forgoes shared text-prefix reuse and prior-image-prefix reuse when adding an image.
Image session SAVE is rejected; text session behavior is unchanged. Current-request
embeddings live only with the active engine; parked state retains small identities.
As in text serving, checkpoints/prompt chunks determine cancellation recovery.

Vision reserves 512MiB from the machine budget before expert allocation for
receiver input buffers/arrays. The separate local integration also accounts for
this reservation in its optional resident-mode admission; resident mode itself
is outside this upstream contribution. The standalone encoder's memory is
external to MLX: budget and measure it separately. HTTP encoder caches use temporary
SVE files under the official server's lifecycle; use synthetic fixtures during
qualification. Normal prompts/outputs must stay unlogged and this is not permission
to enable persistent conversation caches or process private photos.

The separate local resident integration passed12/12 synthetic image contents (10 strict),
live/parked430-token reuse and HTTP-disconnect image recovery on an M5 Pro64GiB.
The independent upstream-main tiny-model suite passed500 tests; the integrated
suite passed548 plus22 subtests. See `bench/results/e55-m5pro-20261017/REPORT.md`
for exact profiles, source hashes, raw outputs, failed attempts and restoration.

Validation status and machine measurements belong in the project report; source
unit tests alone do not establish image understanding on a real model. Open gates include broad visual tasks, long/concurrent serving, arbitrary16K/131K
multimodal contexts, and private-photo accuracy. The bounded synthetic cases above
cover multiple images, reuse, draft/STOP/disconnect recovery and combined memory. No universal MLX versus
Metal performance claim follows from this work.
