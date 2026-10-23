# strata-mlx

Qwen3.8-Flash-Next on Apple Silicon: an MLX engine that reads the GGUF files
[Strata](https://github.com/Niko1221/Strata) uses and speaks Strata's line protocol, so Strata's server and
web page can drive it.

This is an independent, unofficial project. It is not affiliated with Strata or its authors, who have said
that macOS is out of scope for Strata itself. It is experimental and has been run on one machine.

## What it is

Strata runs a 125B-parameter mixture-of-experts model on a 12 GB graphics card by keeping most experts in
host memory. A Mac with unified memory does not need that half of it. This engine takes the other half:

- **The file as it is.** The GSQ-RCO GGUF files load without a weight being changed. Experts stay in the
  file's own blocks, nine block types, and Metal kernels multiply from them. The 28.8 GB n-gram table stays
  on disk.
- **Draft and verify.** The model's own draft head guesses the next tokens and one pass of the model checks
  them. The model decides every token that is written.
- **Prompts.** Read in chunks, the experts a tile at a time under Apple's matrix product for shaders. A
  conversation is read once: a follow-up reads only what is new, a new chat starts after the system prompt it
  shares with the last one, and a conversation can be saved and restored. A conversation that another request
  interrupts is parked in memory and taken back, not read again, by default where the model is held whole with
  4 GiB to spare (`--conversation-cache-mib`).
- **Long conversations**, measured to 128K tokens of context.
- **Less memory than the model.** Experts that do not fit are read from the SSD as tokens ask for them, and
  the dense tensors MLX would hold at twice their size stay the file's blocks, which leaves the experts more.
  A long prompt goes through the layers one at a time, so each layer's experts are read once for it.
- **Strata's line protocol** on stdin and stdout (`strata_mlx/serve.py`).
- **Opt-in image understanding** with `--vision`, using the official standalone encoder and HTTP server.

## Measured

M4 Max, 128 GB, macOS 27, MLX 0.32.3. Greedy decoding. Every number is from this one machine, which slows
down under sustained load, so read them as ranges. The two model files are Q2_0 (66.4 GB) and IQ3_S (83.6 GB).

Answers are written 1.6 to 2.9 times as fast as by llama.cpp or LM Studio, and a long prompt is read 4 to 36%
faster.

**Writing an answer**, tokens a second over four prompts of 256 tokens each (more is better):

| Engine | Q2_0 | IQ3_S |
| :--- | ---: | ---: |
| **strata-mlx (this project)** | **68-111** | **65-95** |
| strata-mlx, draft head off | 44-67 | 43-56 |
| llama.cpp `llama-server` 0.4.1 | 38-39 | 35-36 |
| LM Studio 0.4.25 (llama.cpp runtime) | 43 | 40 |

**Reading a prompt** of 1,180 tokens, tokens a second (more is better):

| Engine | Q2_0 | IQ3_S |
| :--- | ---: | ---: |
| **strata-mlx (this project)** | **570-578** | **510-517** |
| strata-mlx, draft head off | 595-612 | 504-515 |
| llama.cpp `llama-server` 0.4.1 | 494-495 | 488-489 |
| LM Studio 0.4.25 (llama.cpp runtime) | 426-446 | 421-424 |

With the draft head off the engine still copies guesses from earlier in the conversation where the answer
repeats it; one token a pass with no guesses at all is 46-48 tok/s (Q2_0) and 41-43 (IQ3_S).

Two things these tables do not show, both in [ISSUES.md](ISSUES.md): prompts of a few dozen tokens are read
slower than by llama.cpp (about 0.6 s to the first token against 0.3 s), and this engine's reading speeds
are from a later session than the other engines'. Ollama 0.35.1 loads neither file.

In 16-bit floats, the default, about one token in 250 is not the one the exact model would pick, at places
where two candidates are nearly tied; that is so with and without the draft head. `--dtype float32` writes
the exact model's tokens (checked on eight answers) and is slower. On an M5, MLX by default multiplies float32
matrices only about as accurately as float16 ones, so `--dtype float32` and the tests turn that off
(`MLX_ENABLE_TF32=0`); the tests pass on an M5 with it, and the model has not been run in float32 there
([issue 5](https://github.com/yibie/strata-mlx/issues/5)).

With part of the experts in memory and the rest on the SSD, limited on the same machine to stand for smaller
ones, Q2_0 only: answers at 11-12 tok/s with 3.3 GB of experts in memory, 18-27 with 11.9 GB, 23-38 with
20.5 GB; a prompt of 1,180 tokens read at 107-145 tok/s, since a chunk of a prompt reads 26 to 30 GB of
experts from the disk ([issue 4](https://github.com/yibie/strata-mlx/issues/4)). Those were what a 16, 24 and
32 GB machine kept when the engine left macOS 4.2 GB. A real 32 GB Mac, a MacBook Air M5, swapped with 20.5 GB
and wrote answers at 6-19 tok/s; with 16 GB it wrote them at 23-29
([issue 6](https://github.com/yibie/strata-mlx/issues/6)). So the engine now leaves macOS 7.9 GB, which gives
a 32 GB machine's experts 16 GB, and the three figures above have not been measured again with it.

A prompt of more than 2,048 tokens is read a layer at a time there, each layer's experts once and the next
layer's while this one is computed, to the same state bit for bit: as a 32 GB machine 8,192 tokens in 19 s
where chunk by chunk took 32, and 32,768 in 117 s for 193, with the model's n-gram table in the page cache.
With that dropped too, 26 s where chunk by chunk takes 35, and took 47 before the table's rows were read
ahead. On a real 32 GB MacBook Air M5, its owner's run: 31.5 s where chunk by chunk took 45.7, the same answer
every time ([issue 6](https://github.com/yibie/strata-mlx/issues/6)). A shorter prompt is read as before
([ROADMAP.md](ROADMAP.md), "A prompt read a layer at a time"). It holds 1 GB more while it reads, which
the experts give up: a 32 GB machine's are 15 GB where they were 16.

**Reported from another Mac, not measured here.** A Mac Studio with an M1 Ultra and 64 GB, the IQ3_XXS file,
tokens a second end to end (the prompt and the answer together):

| engine | tok/s |
| :--- | ---: |
| **strata-mlx** | **33.5-36.3** |
| llama.cpp with tarruda's optimizations | 32.2 |
| llama.cpp as released | 20.7 |

That is its owner's report as it was given: which prompts, how long, and with which settings is not known, and
the IQ3_XXS file has not been run by this engine's author.

**On an M5 Pro with 64 GB**, [@jccrypto007](https://x.com/jccrypto007)'s runs ([issue 8](https://github.com/yibie/strata-mlx/issues/8)), where
the IQ3_S file is read from the SSD in part (93% of its experts in memory by the rule then; 76% now, below) and Q2_0 is held whole:

| | IQ3_S | Q2_0 |
| :--- | ---: | ---: |
| the turn after a conversation of 10,000 tokens was interrupted, to its first token | 23.4 s | 13.8 s |
| the same with `--conversation-cache-mib 4096` | 0.68 s | 0.24 s |
| a prompt of 8,192 tokens, chunk by chunk | 37.9 s | |
| the same a layer at a time, with reads ahead | 19.1 s | |

All 27 turns after an interruption kept the conversation's facts, and each prompt gave the same tokens whichever
way it was read. It also shows what is still slow there: with IQ3_S a prompt of 1,180 tokens was read at 61-67
tok/s and answers came at 17-44: a chunk read a layer whole from the disk wherever it used 384 or more of its
512 experts, 30-31 layers of 48 for a prompt of code, the 93% that memory then held with the rest. Since
[pull request 7](https://github.com/yibie/strata-mlx/pull/7) it reads only what memory lacks: as a 64 GB machine
with IQ3_S, that prompt's 32 GB read became 1.4 and its 7.2-8.0 s became 3.3-3.9 (on an M5 Ultra, its author's
measurement; [issue 4](https://github.com/yibie/strata-mlx/issues/4)).

**A short prompt after a rest** ([issue 1](https://github.com/yibie/strata-mlx/issues/1), pull request 7). For
`--keep-awake` seconds after a request (600 by default) the engine gives the GPU a small addition a second and,
where it holds the model whole, keeps the model's arrays wired; it takes both. Q2_0 through the server, 8 s
between requests, time to read the prompt, on this M4 Max:

| prompt | before | after |
| :--- | ---: | ---: |
| 27-49 tokens | 618-835 ms | 148-215 ms |
| 340 tokens | 832-934 ms | 692-751 ms |
| 1,180 tokens | 2,113-2,630 ms | 1,829-1,874 ms |

Answers are as fast as before. Idle, the addition cost the GPU a median of 12 mW on an M5 Ultra; a laptop's
battery has not been measured.

The full account, with what was tried and did not help, is in [ROADMAP.md](ROADMAP.md) (English, as it went)
and [docs/OPTIMIZATION_RECORD.md](docs/OPTIMIZATION_RECORD.md) (Chinese, by topic).

## Which Mac

`--check` loads nothing and says what a Mac would do with each file. It works before anything is downloaded,
and for a Mac other than the one it runs on:

```sh
./strata-mlx --check                         # this Mac
./strata-mlx --check --machine-memory 48     # a Mac with 48 GB
./strata-mlx --check --native <shard 1>      # a file on the disk, by its own header
```

By the same arithmetic the engine starts with:

| Mac's memory | Q2_0 (66 GB on disk) | IQ3_S (84 GB on disk) |
| :--- | :--- | :--- |
| 16 GB | too small | too small |
| 24 GB | 19% of the experts in memory | 8% of the experts in memory |
| 32 GB | 44% of the experts in memory | 25% of the experts in memory |
| 36 GB | 54% of the experts in memory | 31% of the experts in memory |
| 48 GB | 82% of the experts in memory | 51% of the experts in memory |
| 64 GB | whole | 76% of the experts in memory |
| 96 GB and more | whole | whole |

"Whole" is the path the speeds under [Measured](#measured) are for, on an M4 Max; other chips have not been
run. Where only part of the experts is in memory the rest is read from the SSD as tokens ask for it, which is
the slow path above. IQ3_S has been run that way on one machine, [@jccrypto007](https://x.com/jccrypto007)'s 64 GB M5 Pro (above), and not by this engine's author. This table is arithmetic, not a test: the only
Mac its author has run the engine on has 128 GB. It leaves a quarter of the memory to macOS and other programs,
8.5 GB at least, and does not count the conversation itself, about 1 GB for 36,000 tokens. That is from two
machines that swapped with less: a MacBook Air M5 with 32 GB, which with 4.2 GB left to macOS swapped and with
8.5 hardly did and was about twice as fast ([issue 6](https://github.com/yibie/strata-mlx/issues/6)); and an
M4 Pro with 48 GB, which with 30.1 GB of Q2_0's experts swapped 5 GB out as it started and with 24 did not
(Strata's own Metal engine took 25.6 there; [@yuanhao](https://x.com/yuanhao/status/2110334086426337592)).
Other sizes are not known: a 16 GB Mac may do with less than is left here, and a 64 GB one, which by the old
rule kept 93% of IQ3_S's experts and did not swap, may be given less than it could take. `--expert-memory GB`
sets the experts' share by hand, and `--expert-memory available` holds it to what was free when it started.

## Running it

Needs an Apple Silicon Mac, Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/). Tested on macOS 27
only; the prompt kernel uses Metal 4's Metal Performance Primitives.

```sh
git clone https://github.com/yibie/strata-mlx && cd strata-mlx
uv sync
uv run pytest          # 256 tests, no model file needed
```

The model, about 66 GB for Q2_0: both shards of one size from
[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
side by side in one folder. Q2_0 and IQ3_S have been run; the other sizes have a path in the loader and have
not.

The draft head is not in those files. Its 31 tensors, 4.9 GB, are fetched out of the original checkpoint by
range requests, without downloading its 360 GB, and each is checked against a SHA-256. The script is Strata's
own, here unchanged and where Strata has it, in [tools](tools):

```sh
python tools/mtp_fetch.py fetch --out ~/.cache/strata-mlx/mtp     # resumable
```

Without the head the engine still runs, at the "draft head off" speeds above.

```sh
./strata-mlx --serve --native <shard 1 of the GGUF> --mtp ~/.cache/strata-mlx/mtp
```

The process then speaks Strata's line protocol: it prints `INFO ...` and `READY`, takes
`GEN <max new tokens> <id>,<id>,...` and answers with `T <id>` lines and a `DONE` line. Prompts are token
ids; tokenizing, chat templates and the HTTP APIs are the server's part. Strata's `serve/server.py`
(0.1.40) has driven it unchanged, with its config's `exe` set to the `strata-mlx` script and the options above
in `args`; Strata's `setup.py` does not run on macOS, so that config is written by hand.

Options: `--spec 1` (draft head off), `--dtype float32`, `--max-context N`, `--expert-memory GB` (how much of
the experts to keep in memory; by default all of them if the machine holds the model; `available` for no more
than the machine has free or in its file cache when the engine starts, where other programs hold a lot), `--machine-memory GiB`
(run as a machine of that size would), `--guess-rows off`, `--expert-profile off`,
`--conversation-cache-mib MiB` (park conversations that other requests interrupt, in that much memory; by default
4,096 where the model is held whole with that to spare and 0 where the memory would be taken from the experts', as
on a Mac that does not hold the model whole; `--conversation-cache-min-tokens`, 256, keeps short requests from
taking one of its 4 slots),
`--keep-awake SECONDS` (how long after a request the GPU is kept awake and the arrays wired; 0 for never),
`--wired off` (keep the GPU awake without wiring the arrays),
`--follow-pressure on` (where experts are read from the SSD: the experts' budget comes down while macOS compresses
the engine and grows back once it stops; off by default, being measured),
`--prompt-by-layer off` (where experts are read from the SSD: read a long prompt chunk by chunk, as before),
`--read-ahead off` (and do not read a layer's experts while the one before it is computed),
`--fusion-limit 6` or `8` (fuse the hyper-connections for windows up to that many tokens, in place of what the
GPU's generation chooses: a 20-core M5 Pro was faster at 6, an M5 Ultra at 8).

The benchmarks in `bench/` tokenize with llama.cpp's `llama-tokenize`, and `bench/compare_servers.py` starts
`llama-server` and LM Studio's `lms` if they are installed.

## Not there

Image generation, several conversations decoded together, and the files other than Q2_0 and IQ3_S. Conversation
caches are 16-bit floats (3.9 GB at 128K tokens). Nothing here has run on a Mac with less than 128 GB.

## License

MIT, see [LICENSE](LICENSE). The projects named below have their own; the one file taken from Strata,
`tools/mtp_fetch.py`, is Strata's and keeps Strata's MIT license, which is beside it.

## What is known, by concept

[docs/wiki](docs/wiki/index.md) holds what has been learned about this engine by concept rather than by date: a
page for each idea, its relations to the others (a [diagram](docs/wiki/graph.md)), and the experiment, change or
issue behind every claim. It starts with how answers are written; ROADMAP.md and ISSUES.md hold the rest, in the
order it was found.

## Thanks to

- [Strata](https://github.com/Niko1221/Strata) (MIT): the design this follows, the line protocol, the order
  of its sampler, the idea of its lookup drafter, which checkpoints of a conversation it keeps and how it
  parks conversations, all written anew here for MLX; and its script that fetches the draft head, included
  as it is.
- [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp) (MIT): the GGUF format and the block types,
  whose unpacking the kernels here follow; the codebook tables come from its `gguf` package at run time.
- [@jccrypto007](https://x.com/jccrypto007), for the runs on a 64 GB M5 Pro in issue 8: parked conversations, prompts read a layer at a time
  and IQ3_S read in part, on a real machine, and what swapped and when.
- [vendyluo](https://github.com/vendyluo), for pull request 7: why a short prompt after a rest was slow and
  how to keep it fast, the floats kept for prompts counted, and a layer's experts read only where memory lacks
  them; measured on an M5 Ultra.
- [MLX and mlx-lm](https://github.com/ml-explore/mlx-lm) (MIT), and the port of this model to mlx-lm in
  [pull request 1788](https://github.com/ml-explore/mlx-lm/pull/1788), which this engine builds on and pins.
- ISTA-DASLab for the GSQ-RCO files, and Qwen for the model. The model's weights have their own license and
  are not part of this repository.
