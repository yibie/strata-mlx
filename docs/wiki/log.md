# Log

> 中文摘要:知识库的变更记录,最新的在前。每行写日期、jj change ID、改了哪些页、依据什么。

- 2026-10-23 · [xwlsutut] · issue 12: conversation parking on by default where the model and its floats are held
  whole with 4 GiB to spare (`check.parking_budget`, `--conversation-cache-mib auto`), `-min-tokens` 256; found on
  the CPU that the slots go oldest first whatever their size, so four short requests took a long conversation's.
  conversation-parking (In this engine, two claims, one superseded, two open questions), memory-rule (one claim);
  the reporter's M2 Max numbers as theirs [#12].
- 2026-10-21 · [ouxxnwtq] · pull request 13 merged ([zpxxwxso]): images, opt-in (`--vision`), a new concept
  (image-understanding) and two experiments of the author's, renumbered e63 and e64 at the merge (this repository's
  e62 was in flight): e63 the author's integration with an unsupported file, kept as history; e64 this branch
  with Q2_0 on the 20-core M5 Pro, 17 of 17 right, two images at 4,442 tokens; conversation-parking.
- 2026-10-21 · [lqvqmrmr] · e62 and Q3_K held as planes where the dense tensors are blocks (experts from the SSD): a
  pass at the file's bits is 1.3 ms shorter at one token, longer from four; held whole the model keeps 8-bit
  arrays; agreement with llama.cpp 378 of 384; dense-blocks.
- 2026-10-21 · [yrkwxyqy] · e61: window kernels for the dense tensors at the file's bits: Q3_K as planes faster than MLX's
  8-bit at every size, IQ4_XS through e36's kernel at 1-2 tokens; bound by decoding; dense-blocks.
- 2026-10-21 · [oyvuuyuk] · e60: the dense tensors as the file's blocks agree with llama.cpp as well as the 8-bit arrays
  (378 against 374 of 384 top tokens after three prompts); D3 can go on; dense-blocks, reference-agreement.
- 2026-10-21 · [rtvwnktp] · e59: the delta-net recurrence alone is 1.2 ms of a pass, not e58's 2.7 (leaving a part out also
  leaves out what only fed it); its kernel near what a kernel of 6 MB takes; decode-pass, e58 corrected.
- 2026-10-21 · [ruspomnz] · e58: each part of a pass left out, shuffled: the delta-net recurrence is 2.7 ms and reads no
  weights (2.2 over its state's floor); the delta-net projections 5.4 ms, near their 8-bit floor; decode-pass.
- 2026-10-21 · [rmxrzznt] · e57 measured as single passes: a pass submitted every 6 layers 2.4-3.0 ms shorter at one token,
  gate and up in one kernel 0.4-0.7 more, the same bits; both on by default; decode-pass, expert-lookup-kernel,
  measurement-noise.
- 2026-10-21 · [qmvxvpnt] · e57 (not measured: the GPU busy with other programs) and two opt-in changes, both off: a pass
  submitted in parts (submit_every), gate and up in one kernel (same bits, tested); decode-pass, expert-lookup-kernel.
- 2026-10-21 · [sksmuwnv] · e56: the experts' gate product in variants: not bound by arithmetic or divergent lookups but by
  how its bytes are read; at one token 2.5x the floor whatever reads it; codes as planes reach the floor at 4-8
  tokens, 7-9% faster; expert-lookup-kernel ("bound by arithmetic" superseded).
- 2026-10-20 · [sylqlors] · e55: a step timed in place and the bandwidth measured again: the floor is 10.8 ms (490 GB/s), not
  13.9; the GPU takes 16.8 ms of a one-token pass, the CPU alone 2.3-4 ms a step; bandwidth-floor, decode-pass.
- 2026-10-17 · [uvwtszxp] · e54, pull request 11 merged ([uvuuktql]): on a 20-core M5 Pro fused up to six was
  faster than eight at seven and eight tokens, against a control engine, so the GPU's generation does not choose the
  fusion limit alone; `--fusion-limit 6|8` sets it by hand, the engine's choice unchanged. Its script, page and
  results renumbered from e49/e46, which are other experiments here; gpu-generation, hyper-connections,
  measurement-noise. e53's line added to index.md.
- 2026-10-16 · [lptnxxvz] · e53: following the pressure, off and on as a 48 GB machine wired: no swap and faster prompts,
  answers not faster (the budget fell to 3-7 GB, the experts read from the disk); stays off; pressure-following.
- 2026-10-16 · [yytmkrlk] · pressure-following, new: the budget follows the process's compression and the pressure
  level (opt-in, --follow-pressure); runs under held memory wait for the machine to settle first.
- 2026-10-16 · [yowlwvyo] · e52: under pressure, letting go of arrays takes their compression off one for one, and macOS
  compresses arrays the GPU reads as much as idle ones; expert-store, swapping, available-memory.
- 2026-10-16 · [zxmztxsx] · e51: margins of 8, 0, 4 GB on --expert-memory available, not settled (what is available
  moved by 10.7 GB between runs); AVAILABLE_MARGIN stays 0; available-memory, measurement-noise, simulated-machine.
- 2026-10-16 · [kmznkvlw] · e50 and --hold quiet: a quiet 32 GB machine, no swap, a quarter slower than issue 6's Air;
  simulated-machine, swapping, system-reserve.
- 2026-10-16 · [kzynqwqp] · e49: --expert-memory available under wired memory, twice e48's speed and next to no swap;
  available-memory, swapping, simulated-machine, system-reserve.
- 2026-10-16 · [szyzmoou] · e48: as a 48 GB machine wired, a full desktop open: the engine compressed, 10-15 tok/s;
  simulated-machine, swapping, system-reserve, expert-store.
- 2026-10-16 · [zzzlsupr] · e47: an unwired hold of 96 GiB kept 8.9 resident, so the hold is wired since; simulated-machine,
  swapping, measurement-noise.
- 2026-10-16 · [tukzwlpx] · A hold is judged by its resident part, not its footprint, which counts swapped pages too;
  the run counts this process's own memory compressed or swapped out (bench/memory.py:own). e46's "hold whole"
  corrected; simulated-machine, swapping.
- 2026-10-16 · [vrnvzqql] · e46: as a 48 GB machine with 80 GiB held; simulated-machine, swapping, system-reserve,
  measurement-noise.
- 2026-10-15 · [wumtsnom] · simulated-machine, working-memory: memory held by another process and the kernel's
  footprint (bench/memory.py, bench/paged.py --hold); a first run as a 96 GB machine; MLX's arrays miss 1.55 GB
  of what the process holds.
- 2026-10-15 · [owsylkrr] · No concept changed. Updating this knowledge base made a fixed step: AGENTS.md lists it,
  and the checker fails a change to strata_mlx/ or bench/ without a line here, and an experiment from e43 on
  without its page.
- 2026-10-15 · [oonxmoxu] · measurement-noise: a control engine, the same setting as the baseline, says what a
  one-process comparison cannot tell apart; from the M5 Ultra's run in pull request 9's thread.
- 2026-10-15 · [uvunqsoq] · hyper-connections, gpu-generation: the fusion limit chosen by GPU generation (8 from 17,
  6 below), from the M5 Ultra's measurement in pull request 9's thread.
- 2026-10-14 · [nlykrpkv] · Every concept given an area, and graph.md drawn as an overview of the areas and one diagram
  for each. Three areas added: "Reading prompts" (prompt-chunk, expert-tiles, kept-floats, gpu-idle; ngram-table
  moved here), "Conversations" (recurrent-state, attention-cache, checkpoints, conversation-parking,
  session-files, stopping), "Precision" (float32-path, tf32-on-m5, reference-agreement; exactness moved here),
  from ROADMAP.md's "Done" and prompt sections, e29 to e33, issues 1, 3 and 5, pull request 7.
- 2026-10-14 · [wvuromwk] · "Memory and smaller Macs": thirteen concepts (memory-rule, system-reserve, working-memory,
  available-memory, swapping, simulated-machine, expert-store, ssd-reads, page-cache, dense-blocks,
  layer-major-prefill, layer-read-ahead, router-lookahead) from ROADMAP.md's sections on small Macs, e35 to e40,
  issues 3, 4, 6 and 8 and pull request 7; ngram-table given its relation to the page cache.
- 2026-10-14 · [skkpzrnm] · started as a pilot: "Writing answers", eleven concepts, three
  experiments (e43-e45) drawn from ROADMAP.md, pull requests 9 and 10, and [skxmzozu] [wykuvxvr] [tmyqkuqv]
  [zryukqou]; the checker and the graph.
