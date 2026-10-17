"""The engine as a process that speaks Strata's line protocol, so `serve/server.py` can drive it unchanged.

The server starts `<exe> --serve <args>` and talks over stdin and stdout. Prompts arrive as token ids.

    engine  INFO engine=... backend=mlx ...          once, then
            READY <max context> stop
    server  GEN <max new> [key=value ...] <id>,<id>,...
    engine  RESUME <tokens reused from the conversation held>
            PP <read> <total> <ms> <tokens/s>          after each chunk of the prompt; of one read a layer at a
                                                       time, after each layer, in proportion to the layers done
            T <id>                                     each token of the answer, the end token included
            DONE <tokens> <prompt tokens> <prompt ms> <decode ms> <stop|length|cancel> <guesses kept>
                 <guesses made> <reused> 0 <lookup passes> 0 0 0.0 <prompt tokens read> 0
            ERR <reason>                               instead, for a request that cannot run; no DONE follows
    server  STOP                                       ends the request under way; its DONE says cancel
            SAVE <path> | RESTORE <path>               between requests: the conversation held, to or from a file
    engine  SWAIT transfer <seconds>                   how long the one blocking step may take, then
            SAVED | RESTORED <tokens> <bytes> <ms>
            SERR <invalid|storage|memory|io> 0 <reason>  instead; the conversation held is as it was
    server  QUIT

    strata-mlx --serve --native <shard 1 of the GGUF> --mtp <folder of the draft head>

`--fusion-limit 6` or `8` fuses the hyper-connections for windows up to that many tokens, in place of what the
GPU's generation chooses (`auto`): a 20-core M5 Pro was faster at 6, an M5 Ultra at 8 (e54). INFO says which.

On a machine whose memory does not hold the model, only part of the experts stays in memory and the others are
read from the SSD as tokens ask for them (paged.py). How much is worked out from the machine's memory;
`--expert-memory <GB>` sets it by hand, and `--machine-memory <GiB>` runs as a machine of that size would.
Such a machine counts which experts its answers use, in `~/.cache/strata-mlx/expert-use-<model>.npy`
(`--expert-profile <path>` for another place, `off` for none), and starts with the most used ones in memory.

The engine holds one conversation, and a request that shares little with it cuts it back to what they share.
`--conversation-cache-mib <MiB>` (with `--conversation-cache-slots`, 4, and `--conversation-cache-min-tokens`, 0)
parks it instead, in that much memory, and takes it back when a prompt continues it (conversations.py); the
memory is taken from what the experts would have. Off by default. Two keys of a `GEN` line say which checkpoints
a request leaves: `ckpt=0`, that no later request continues it, and `pin=<tokens>`, that its first tokens are a
prefix other requests will start with. These are Strata's, as are the lines the cache writes to the log.

The draft head's guesses are worked out over the rows of the output layer that have been in sight (rows.py);
`--guess-rows off` has them go over all rows.

Between requests the GPU is not left idle for long (#1): for `--keep-awake <seconds>` after a request (600 by
default, 0 for never) the engine gives the GPU one addition a second, and meanwhile a machine that holds the model
keeps its arrays resident (MLX's wired limit; `--wired off` leaves them unwired). Unwired, a prompt read after two
seconds of nothing ran at about half speed or less; wired, the first work after four seconds of nothing waited
250-380 ms, as if the arrays were being made resident again (bench/experiments/e42, M5 Ultra: a piece of 64 tokens
in 250-350 ms unwired after a rest so and 420-445 wired, the wait included; 110-180 ms with both). It takes both,
on an M5 Ultra and on an M4 Max; wired with the GPU left alone is the slowest of all, so when the window ends the
arrays are let go of, and they are wired again once the model has run for the next request, while they are still
resident from it (KeepAwake, PR #7).

`strata-mlx --check [--native <shard 1>] [--machine-memory <GiB>]` loads nothing and says which of these a Mac
of that memory would do with a file (check.py).
"""

import errno
import hashlib
import os
import queue
import sys
import threading
import time

from . import bylayer
from .conversations import Parked, resume
from .engine import Engine, Stats, FUSE_UPTO
from .paged import available_memory, load_profile, machine_memory, save_profile
from .sampler import Sampling

VERSION = "0.0.1"
END_TOKENS = (248044, 248046)  # <|endoftext|>, <|im_end|>
_FLOATS = ("temperature", "top_p", "min_p", "penalty_repeat", "penalty_freq", "penalty_present")
_INTS = ("top_k", "penalty_last_n", "seed")
_CHECKPOINTS = ("ckpt", "pin")  # ckpt=0: no later request continues this one; pin=N: the first N tokens are a prefix
_IGNORED = ("cvec", "pcie_frac", "spec_min_p")  # for the CUDA engine's expert cache and its options
AWAKE_EVERY = 1.0  # seconds between the additions that keep the GPU awake: one in 2.5 s was enough as well (e42)


def serve_lines(commands: queue.Queue, handle, awake) -> None:
    """The lines from the server until QUIT. `handle(line)` says whether the model ran for it; between lines,
    `awake` (KeepAwake) keeps the GPU and the arrays ready."""
    while True:
        try:
            line = commands.get(timeout=awake.timeout())
        except queue.Empty:
            awake.idle()
            continue
        if line == "QUIT":
            return
        awake.heard(bool(handle(line)))


class KeepAwake:
    """#1, between requests: for `seconds` after the last line the GPU is given an addition every AWAKE_EVERY
    seconds, and the model's arrays are wired (`limit` bytes; 0: never). Wired and left alone, a prompt is read
    slower than with neither: the first work waits 250-380 ms, as if they were being made resident again, and
    wiring them when a request comes waits as long. So at the window's end they are let go of; the next request
    is read without them, and once the model has run for it they are wired again, in 10-30 ms after its DONE,
    while still resident from it (PR #7).

    `wire(bytes)` sets the limit, `wake()` gives the GPU its addition; `clock` is time.monotonic but in tests.
    """

    def __init__(self, seconds: float, limit: int, wire, wake, clock=time.monotonic):
        self.seconds, self.limit = seconds, limit if seconds > 0 else 0
        self.wire, self.wake, self.clock = wire, wake, clock
        self.wired, self.last = False, clock()
        if self.limit:
            self.wire(self.limit)
            self.wired = True

    def timeout(self) -> float | None:
        """How long to wait for the next line: past the window, once the arrays are let go of, nothing is woken
        for, the CPU no more than the GPU. While they are wired the wait ends, however late the window's last
        addition finished, so that `idle` lets them go."""
        return AWAKE_EVERY if self.wired or self.clock() - self.last < self.seconds else None

    def idle(self):
        """No line for a while."""
        if self.clock() - self.last < self.seconds:
            self.wake()
        elif self.wired:
            self.wire(0)
            self.wired = False

    def heard(self, ran: bool = False):
        """A line was handled, any line from the server, not only a request: someone is using it. `ran`: the
        model ran for it, which leaves its arrays resident to be wired again; after any other line they would
        wait as long as the next request would."""
        if ran and self.limit and not self.wired:
            self.wire(self.limit)
            self.wired = True
        self.last = self.clock()


class Session:
    """One conversation slot: turns protocol lines into engine calls and writes the answers."""

    def __init__(self, engine: Engine, write, max_context: int, stop: threading.Event | None = None,
                 parked: Parked | None = None, vision: bool = False):
        self.engine, self.write, self.max_context, self.parked = engine, write, max_context, parked
        self.stop = stop or threading.Event()
        self.ran = False
        self.vision = vision

    def handle(self, line: str) -> bool:
        """One line; whether the model ran for it (a DONE was written)."""
        self.ran = False
        self._handle(line)
        return self.ran

    def _handle(self, line: str):
        if line.startswith("GEN "):
            self.gen(line[4:])
        elif line.startswith("GENI "):
            if self.vision:
                self.gen(line[5:], image=True)
            else:
                self.write("ERR this engine was started without --vision")
        elif line.startswith(("SAVE ", "RESTORE ")):
            self.session_file(*line.split(" ", 1))
        elif line.startswith("VRAM"):
            self.write("ERR the MLX engine has no expert cache to resize")
        elif line.strip():
            self.write(f"ERR unknown command: {line[:40]}")

    def session_file(self, command: str, path: str):
        """`SAVE <path>` / `RESTORE <path>`: the conversation the engine holds, to or from a file."""
        e, t0 = self.engine, time.perf_counter()
        if not path:
            self.write("SERR invalid 0 missing path")
            return
        try:
            # one blocking step: tell the server how long it may take before it counts the engine as silent
            size = _state_bytes(e) if command == "SAVE" else os.path.getsize(path)
            self.write(f"SWAIT transfer {int(min(3600, 60 + size / 2e7))}")
            size = e.save(path) if command == "SAVE" else e.restore(path)
        except ValueError as err:
            self.write(f"SERR invalid 0 {' '.join(str(err).split())}")
        except MemoryError:
            self.write("SERR memory 0 not enough memory for the session")
        except OSError as err:
            kind = "storage" if err.errno in (errno.ENOSPC, errno.EDQUOT) else "io"
            self.write(f"SERR {kind} 0 {' '.join(str(err).split())}")
        else:
            done = "SAVED" if command == "SAVE" else "RESTORED"
            self.write(f"{done} {len(e.tokens)} {size} {(time.perf_counter() - t0) * 1e3:.1f}")

    def gen(self, rest: str, image: bool = False):
        try:
            fields = rest.split()
            max_new, options, marks = int(fields[0]), {}, {"ckpt": 1, "pin": 0}
            for f in fields[1:-2] if image else fields[1:-1]:
                key, _, value = f.partition("=")
                if key in _CHECKPOINTS:
                    marks[key] = int(value)
                elif key in _FLOATS:
                    options[key] = float(value)
                elif key in _INTS:
                    options[key] = int(value)
                elif key == "img_keys" and image:
                    pass  # identity is computed from the actual rows, never client-supplied hashes
                elif key not in _IGNORED:
                    raise ValueError(key)
            prompt = [int(t) for t in fields[-1].split(",")]
            vocab = self.engine.vocab
            if max_new < 0 or not prompt or len(fields) < 2 or any(not 0 <= t < vocab for t in prompt):
                raise ValueError
        except (ValueError, IndexError):
            self.write("ERR bad request")
            return
        if len(prompt) >= self.max_context:
            self.write(f"ERR the prompt ({len(prompt)} tokens) exceeds the context ({self.max_context})")
            return

        vision = None
        if image:
            from .vision import Vision
            try:
                vision = Vision.load(fields[-2], prompt, self.engine.m.args.hidden_size)
            except (ValueError, OSError) as err:
                self.write("ERR " + " ".join(str(err).split()))
                return
        e = self.engine
        e.sampling = Sampling(**options)
        t0 = time.perf_counter()
        keep, pin = marks["ckpt"] != 0, marks["pin"]
        reused = resume(e, self.parked, prompt, vision.key if vision else None)
        e.vision = vision
        self.write(f"RESUME {reused}")
        e.stats = Stats()

        def progress(done, total):
            ms = (time.perf_counter() - t0) * 1e3
            self.write(f"PP {done} {total} {ms:.0f} {(done - reused) / max(ms, 1e-3) * 1e3:.1f}")
            if self.stop.is_set():
                raise _Cancelled

        def step():
            if self.stop.is_set():
                raise _Cancelled

        produced, finish = 0, "length"
        try:
            token = e.read(prompt, on_chunk=progress, keep=keep, pin=pin, on_step=step)
            e.one_shot = not keep
        except _Cancelled:
            # the prompt is read as far as the last chunk, or the last piece of one read a layer at a time
            # (bylayer.py); what was read stays for the next request
            self._done(0, prompt, reused, (time.perf_counter() - t0) * 1e3, 0.0, "cancel", read=len(e.tokens) - reused)
            return
        t1 = time.perf_counter()
        budget = min(max_new, self.max_context - len(prompt))
        pending = [token]
        while produced < budget:
            if self.stop.is_set():
                finish = "cancel"
                break
            if not pending:
                pending = e.step(budget - produced, END_TOKENS)
            token = pending.pop(0)
            self.write(f"T {token}")
            produced += 1
            if token in END_TOKENS:
                finish = "stop"
                break
        self._done(produced, prompt, reused, (t1 - t0) * 1e3, (time.perf_counter() - t1) * 1e3, finish)

    def _done(self, produced, prompt, reused, prompt_ms, decode_ms, finish, read=None):
        # `read`: the prompt tokens read by this request, which is not all the others when it was stopped while
        # reading (Strata's #471: the server counts a stopped read as reused + read)
        s, self.ran = self.engine.stats, True
        read = len(prompt) - reused if read is None else max(0, read)
        self.write(f"DONE {produced} {len(prompt)} {prompt_ms:.1f} {decode_ms:.1f} {finish} {s.accepted} {s.drafted} "
                   f"{reused} 0 {s.lookup_steps} 0 0 0.0 {read} 0")


class _Cancelled(Exception):
    pass


def _state_bytes(e: Engine) -> int:
    """About what a session file of this conversation takes."""
    caches = [*e.attn.values(), *([e.mtp_cache] if e.mtp_cache is not None else [])]
    kv = sum(a.nbytes for c in caches if c.keys is not None for a in c.state)
    small = sum(a.nbytes for pair in e.gdn.values() for a in pair)
    return kv + small * (1 + len(e.checkpoints))


def model_identity(path: str, dtype: str) -> str:
    """Names a model file without reading it all: its size and a hash of its first megabytes, where a GGUF
    file keeps its metadata and its table of tensors."""
    with open(path, "rb") as f:
        head = hashlib.sha256(f.read(8 << 20)).hexdigest()[:32]
    return f"{os.path.getsize(path)}:{head}:{dtype}"


def fusion_limit(value: str, automatic: int) -> int:
    """Explicit measured override; auto retains the existing generation-based default."""
    if value == "auto":
        return automatic
    if value not in ("6", "8"):
        raise ValueError("--fusion-limit must be auto, 6, or 8")
    return int(value)


def _options(argv: list) -> dict:
    """The arguments this engine knows; the CUDA engine's others (expert cache, K/V placement, ...) are skipped."""
    known = {"--native": None, "--mtp": None, "--spec": "4", "--max-context": "131072", "--dtype": "float16",
             "--prefill": "2048", "--suffix-draft": "auto", "--expert-memory": "auto", "--machine-memory": None,
             "--expert-profile": "auto", "--guess-rows": "on", "--conversation-cache-mib": "0",
             "--conversation-cache-slots": "4", "--conversation-cache-min-tokens": "0",
             "--prompt-by-layer": "on", "--read-ahead": "on", "--follow-pressure": "off", "--keep-awake": "600", "--wired": "on",
             "--fusion-limit": "auto"}
    out, skipped, i = dict(known), [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--vision":
            out[a] = True
            i += 1
        elif a in known:
            if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
                raise ValueError(f"{a} requires a value")
            out[a] = argv[i + 1]
            i += 2
        else:
            skipped.append(a)
            i += 2 if a.startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--") else 1
    return out, skipped


def main():
    argv = [a for a in sys.argv[1:] if a != "--serve"]
    if "--check" in argv:
        from .check import main as check

        return check([a for a in argv if a != "--check"])
    # the protocol owns stdout: anything a library prints goes to the log instead
    wire = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)

    def write(line: str):
        wire.write(line + "\n")
        wire.flush()

    try:
        opts, skipped = _options(argv)
        selected_fusion = fusion_limit(opts["--fusion-limit"], FUSE_UPTO)
    except ValueError as error:
        write(f"ERR {error}")
        return 1

    if not opts["--native"]:
        write("ERR --native <shard 1 of the GGUF> is required")
        return 1
    if skipped:
        print("strata-mlx: arguments of the CUDA engine that do nothing here:", " ".join(skipped), file=sys.stderr)

    if opts["--dtype"] == "float32":
        # #5: on chips with Neural Accelerators (M5) MLX multiplies float32 matrices at reduced precision unless
        # told not to, before its first computation; float32 is asked for to get the exact model's tokens
        os.environ.setdefault("MLX_ENABLE_TF32", "0")

    import mlx.core as mx

    from .load import load
    from .mtp import load_mtp

    t0 = time.perf_counter()
    dtype = getattr(mx, opts["--dtype"])
    machine = int(float(opts["--machine-memory"]) * 2**30) if opts["--machine-memory"] else machine_memory()
    # parked conversations are memory the experts cannot have: the machine is that much smaller for them
    parked = Parked(int(opts["--conversation-cache-mib"]) << 20, int(opts["--conversation-cache-slots"]),
                    int(opts["--conversation-cache-min-tokens"]))
    vision_reserve = (512 << 20) if opts.get("--vision") else 0
    machine -= parked.budget + vision_reserve
    wanted = opts["--expert-memory"]
    # `available` (#6): what this machine has free or in its file cache, before the model's own files are read
    at_start = available_memory() if wanted == "available" and not opts["--machine-memory"] else None
    model, _ = load(opts["--native"], dtype, expert_memory="auto" if wanted in ("auto", "available") else float(wanted),
                    machine_memory=machine)
    store = model.expert_store
    paged = store is not None
    # #1: resident, the arrays are not let go of by the GPU after a few seconds of nothing; not where experts are
    # read from the SSD, whose machine may need macOS to be able to move memory

    first = True

    def set_wired(size: int):
        nonlocal first
        t = time.perf_counter()
        mx.synchronize()  # as mlx-lm does: the limit is not changed while something is being computed
        mx.set_wired_limit(size)
        if not first:  # what letting go and wiring again take: after a request's DONE, before the next line
            print(f"strata-mlx: the arrays {'wired again' if size else 'let go of'} in "
                  f"{(time.perf_counter() - t) * 1e3:.0f} ms", file=sys.stderr)
        first = False

    awake = KeepAwake(float(opts["--keep-awake"]), 0 if paged or opts["--wired"] == "off" else
                      mx.device_info()["max_recommended_working_set_size"], set_wired,
                      lambda: mx.eval(mx.array(1.0) + 1))
    if not paged and not model.keeps_floats:
        print("strata-mlx: the Q6_K layers' weights as floats do not fit beside the model; they are made anew for "
              "each chunk of a prompt, which reads prompts about a tenth slower", file=sys.stderr)
    mtp = load_mtp(opts["--mtp"], model.args.text, dtype) if opts["--mtp"] else None
    chunk = int(opts["--prefill"]) if opts["--prefill"].isdigit() else 2048
    # --spec is the CUDA engine's window; here anything above 1 turns the draft head on, and how many guesses it
    # makes for a pass is its own business (drafting.py)
    # Guesses copied from earlier text: by themselves when there is no draft head. With the head guessing as
    # many tokens as it is sure of they added nothing, measured (rewriting a file 100 against 101 tok/s, code 71
    # against 74), so they are then only on if asked for.
    head = mtp if int(opts["--spec"]) > 1 else None
    copies = opts["--suffix-draft"]
    engine = Engine(model, head, fuse_upto=selected_fusion, lookup=head is None if copies == "auto" else int(copies) > 0,
                    chunk=chunk, guess_rows=opts["--guess-rows"] != "off",
                    # where experts are read from the SSD (bylayer.py): `off` for a prompt read chunk by chunk
                    # as before, and for no layer's experts read while another is computed
                    piece=bylayer.PIECE if opts["--prompt-by-layer"] != "off" else 0,
                    ahead=opts["--read-ahead"] != "off")
    engine.identity = model_identity(opts["--native"], opts["--dtype"])
    if paged and wanted in ("auto", "available"):
        # now that everything else is in memory: what is left is the experts'. `available`: and no more than
        # what this machine had free or in its file cache, which says nothing of a machine it stands for
        ruled = store.fit(machine)
        if at_start is not None:
            store.fit(machine, sum(at_start))
            print(f"strata-mlx: {at_start[0] / 1e9:.1f} GB free and {at_start[1] / 1e9:.1f} GB of file cache at the start; "
                  f"the experts get {store.budget / 1e9:.1f} GB (by the rule alone {ruled / 1e9:.1f})", file=sys.stderr)
        elif wanted == "available":
            print("strata-mlx: --expert-memory available: " + ("not as another machine (--machine-memory)"
                  if opts["--machine-memory"] else "the counts could not be read") + "; the rule alone", file=sys.stderr)
    if paged and opts["--follow-pressure"] == "on":  # the budget comes down while macOS compresses the engine
        store.follow_pressure()
        print(f"strata-mlx: the experts' budget follows the memory pressure, up to {store.budget / 1e9:.1f} GB",
              file=sys.stderr)

    max_context = int(opts["--max-context"])
    # compile the kernels of every window size and of the sampler now, not during the first request
    engine.warm()
    for sampling in (Sampling(), Sampling(temperature=1.0, seed=1)):
        engine.generate(list(range(1000, 1024)), 12, sampling=sampling)
    # and a prompt long enough to go the way long prompts go, so that the kernels of that way are compiled too.
    # Not with experts read from the SSD: there it would read every layer whole.
    engine.reset()
    profile_path, before = None, None
    if not paged:
        engine.read(list(range(1000, 1600)))
        engine.reset()
    elif opts["--expert-profile"] != "off":
        # the experts this machine's answers used most, in memory before the first request: without them the
        # first answer after a start is the slowest. The warm-up's own experts and counts are let go first.
        profile_path = opts["--expert-profile"]
        if profile_path == "auto":
            name = hashlib.sha256(engine.identity.encode()).hexdigest()[:16]
            profile_path = os.path.expanduser(f"~/.cache/strata-mlx/expert-use-{name}.npy")
        before = load_profile(profile_path)
        store.clear()
        if before is not None:
            t1 = time.perf_counter()
            n = store.preload(before)
            print(f"strata-mlx: {n} experts ({store.held / 1e9:.1f} GB) read ahead in {time.perf_counter() - t1:.1f} s",
                  file=sys.stderr)
        store.reset_stats()
    print(f"strata-mlx: session is up in {time.perf_counter() - t0:.0f} s, "
          f"{mx.get_active_memory() / 2**30:.1f} GiB of weights", file=sys.stderr)

    stop, commands = threading.Event(), queue.Queue()

    def reader():
        for line in sys.stdin:
            line = line.rstrip("\n")
            if line == "STOP":
                stop.set()
            else:
                if line.startswith("GEN"):
                    stop.clear()  # a STOP that arrived after its request had ended
                commands.put(line)
        commands.put("QUIT")

    threading.Thread(target=reader, daemon=True).start()
    session = Session(engine, write, max_context, stop, parked, vision=opts.get("--vision", False))
    write(f"INFO engine={VERSION}-mlx backend=mlx vision={int(session.vision)} mlx={mx.__version__} context={max_context} "
          f"floats={opts['--dtype']} drafts={engine.drafts} lookup={int(engine.lookup is not None)} "
          f"guess_rows={int(engine.rows is not None)} fusion_limit={engine.fuse_upto}"
          + (f" conversation_cache_mib={parked.budget >> 20} conversation_cache_slots={parked.slots}"
             if parked.budget else "")
          + (f" expert_memory_gb={store.budget / 1e9:.1f}" if paged else "")
          + f" wired={int(awake.limit > 0)} keep_awake={opts['--keep-awake']}")
    write(f"READY {max_context} stop")
    awake.heard()  # the window from here, not from the load

    def handle(line: str) -> bool:
        ran = session.handle(line)
        if profile_path and line.startswith("GEN") and store.use:
            try:
                save_profile(profile_path, store.profile(before))
            except OSError as e:
                print(f"strata-mlx: the expert profile was not saved: {e}", file=sys.stderr)
        return ran

    serve_lines(commands, handle, awake)
    return 0


if __name__ == "__main__":
    sys.exit(main())
