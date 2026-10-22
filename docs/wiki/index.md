# Index

> 中文摘要:所有概念的目录,按领域分组:写回答的速度、内存与小内存机型、读提示、对话、精度,以及跨领域的概念。

## Writing answers

- [A decode pass](concepts/decode-pass.md): one forward of a window; what tokens a second are made of.
- [The memory bandwidth floor](concepts/bandwidth-floor.md): the least a pass can take, from the bytes it reads.
- [The fixed cost of a kernel](concepts/kernel-fixed-cost.md): what thousands of small kernels cost beyond their work.
- [The draft window](concepts/draft-window.md): guesses checked in one pass, and how many to make.
- [Guess acceptance](concepts/guess-acceptance.md): the share of guesses kept, the content's more than the engine's.
- [The Q2_0 experts' table kernel](concepts/expert-lookup-kernel.md): experts by a table of sums, and its threads.
- [Hyper-connections](concepts/hyper-connections.md): four residual streams mixed at every layer, 1.27 GB a token.

## Memory and smaller Macs

- [The memory rule](concepts/memory-rule.md): from a machine's memory, held whole or how many experts stay.
- [What is left to macOS](concepts/system-reserve.md): a quarter of the memory, 8.5 GB at least, from two machines that swapped.
- [Working memory](concepts/working-memory.md): what the process holds beyond its weights, 5 GB.
- [Available memory at start](concepts/available-memory.md): an option that counts what is free before loading.
- [The budget following the pressure](concepts/pressure-following.md): down while macOS compresses the engine, back after; being measured.
- [Swapping](concepts/swapping.md): what the rule is there to prevent, seen only on real machines.
- [A smaller machine, simulated](concepts/simulated-machine.md): what capping the budget on a 128 GB Mac can and cannot show.
- [The expert store](concepts/expert-store.md): experts in memory within a budget, the rest read from the SSD.
- [Reads from the SSD](concepts/ssd-reads.md): 6.5 GB/s here; the buffer, not the size, decides the cache.
- [The page cache](concepts/page-cache.md): what makes benchmarks fast and the n-gram table cheap, until memory is short.
- [The dense tensors as the file's blocks](concepts/dense-blocks.md): 1.74 GB given to the experts.
- [A prompt read a layer at a time](concepts/layer-major-prefill.md): each layer's experts read once a piece, the same bits.
- [The next layer read ahead](concepts/layer-read-ahead.md): the disk hidden behind the layer before.
- [Experts guessed before their router](concepts/router-lookahead.md): measured, not built.

## Reading prompts

- [Image understanding](concepts/image-understanding.md): the encoder's rows at the image tokens, (t, h, w) positions, reuse keyed by the images.
- [A prompt read in chunks](concepts/prompt-chunk.md): 2,048 tokens through all the layers at a time.
- [A prompt's experts in tiles](concepts/expert-tiles.md): 16 tokens by 64 rows, from the file's blocks, every type.
- [The floats kept for prompts](concepts/kept-floats.md): the Q6_K layers made float once, 3.3 GB with IQ3_S.
- [The GPU after a rest](concepts/gpu-idle.md): why a short prompt waited half a second, and the fix.
- [The n-gram table](concepts/ngram-table.md): 28.8 GB looked up by hashes, on the disk.

## Conversations

- [The recurrent state](concepts/recurrent-state.md): 110 MiB that cannot be cut back, only restored.
- [The attention caches](concepts/attention-cache.md): 30 KB a token, cut back anywhere, not counted by the rule.
- [Checkpoints](concepts/checkpoints.md): where a follow-up goes on from; the root, `ckpt=0`, `pin=N`.
- [Parked conversations](concepts/conversation-parking.md): a chat interrupted is taken back, 19.7 s to 0.26.
- [A conversation on disk](concepts/session-files.md): `SAVE` and `RESTORE`.
- [Stopping a request](concepts/stopping.md): where a stop is seen and what stays read.

## Precision

- [Exactness](concepts/exactness.md): the same bits for the same input; the rule every change keeps or reports.
- [The float32 path](concepts/float32-path.md): the exact path, at half the speed with Q2_0.
- [Reduced-precision float32 on an M5](concepts/tf32-on-m5.md): MLX's default there, and `MLX_ENABLE_TF32=0`.
- [Agreement with llama.cpp](concepts/reference-agreement.md): how a change that is not bit for bit is judged.

## Across areas

- [The GPU's generation](concepts/gpu-generation.md): thresholds tuned on one chip are not settled for another.
- [Measurement noise](concepts/measurement-noise.md): why changes are compared in one process, passes shuffled.

## Experiments

- [e43: the experts through MLX's quantized product](experiments/e43-experts-as-qmm.md): slower, not used.
- [e44: threads to a row of the table kernel](experiments/e44-lookup-threads.md): 3-27% on the M4 Max.
- [e45: pull requests 9 and 10 checked on the M4 Max](experiments/e45-pull-request-9-check.md): merged, fused up to six.
- [e46: as a 48 GB machine, with the memory held](experiments/e46-held-as-48gb.md): 25-34 tok/s; 25 GB swapped, whose not told.
- [e47: as a 32 GB machine, the memory held but not wired](experiments/e47-held-as-32gb-unwired.md): invalid; an idle hold is pushed out first.
- [e48: as a 48 GB machine, the memory held and wired](experiments/e48-wired-as-48gb.md): the engine pushed out, 10-15 tok/s, with a full desktop.
- [e49: as e48, the experts bounded by what was available](experiments/e49-wired-as-48gb-available.md): twice as fast, hardly any swap, no margin left.
- [e50: as a quiet 32 GB machine, the memory wired](experiments/e50-quiet-as-32gb.md): no swap, a quarter slower than the real Air.
- [e51: a margin on what is available](experiments/e51-available-margin.md): not settled; what is available moved by 10.7 GB in 30 s.
- [e52: letting go of arrays takes their compression off](experiments/e52-shed-compressed.md): one for one, whichever arrays.
- [e53: the budget following the pressure, off and on](experiments/e53-follow-pressure.md): no swap, answers not faster; stays off.
- [e54: the fusion limit on an M5 Pro, against a control engine](experiments/e54-shared-weight-controls.md): 6 faster there; `--fusion-limit`.
- [e63: image receiving, the author's integration](experiments/e63-vision-integration.md): an unsupported file on another branch; historical.
- [e64: images on this branch with Q2_0](experiments/e64-vision-review.md): 17 of 17 right, two images at 4,442 tokens, past the indexer's 2,048.

Earlier experiments, e01-e42, are in [ROADMAP.md](../../ROADMAP.md) and `bench/experiments/`.

The relations as a diagram: [graph.md](graph.md).
