# Log

> 中文摘要:知识库的变更记录,最新的在前。每行写日期、jj change ID、改了哪些页、依据什么。

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

- [zpxxwxso] Implement opt-in SVE1/GENI receiver, image positions, shifted draft inputs and image-key RAM parking; preserve official MIT attribution. Tests and bounded real-model integration recorded separately.

- [zpxxwxso] [e55] Final548+22/500 Mac tests;12/12 synthetic image contents (10 strict),430-token live/parked reuse,437-token follow-up, HTTP cancel/recovery,zero new system swapouts across37 samples. Preserve failed graph/admission/eviction attempts and original service restoration evidence.

- [zpxxwxso] Rebase image receiving onto main295c965 after PR11 merged; preserve upstream fusion and pressure-following options together with --vision, combine roadmap records and regenerate the knowledge graph. No model/performance claim changes.

- [qkwxxvrp] Make external vision dependencies reproducible: pinned encoder-only build/tokenizer instructions, portable config/launch helper and schema-required no-model preflight. Eleven CPU-only setup tests and511 full Mac tests pass; actual generated preflight passes in a fresh server venv, including quoted paths and preserved Python venv symlinks. No model/decoder behavior or performance claim changes.

- [qkwxxvrp] Rebased setup onto merged main295c965: full M5 Pro suite548 tests plus22 subtests passed in41.40s, including combined vision/fusion/pressure option ordering. Selected production runtime restored with synthetic text/image checks, unchanged fresh photo aggregates and free locks. No new full-model performance measurement.
