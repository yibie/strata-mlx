"""e62: a pass with the dense Q3_K and IQ4_XS tensors at the file's bits (Q3_K as planes) against 8-bit arrays.

The model is loaded once with each of those layers held both ways; a class-wide switch says which is called.
Two engines (and a control of the arrays) are traced with the switch set their way, and before each of their
passes it is set again for the parts not compiled (attention's projections). Single passes of 1, 2, 4 and 8
tokens after a 300-token conversation, every (engine, size) shuffled, the lower quartile (e57's way).

    uv run python bench/experiments/e62_dense_planes_in_pass.py --model <Q2_0 shard 1> --json out.json
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx  # noqa: E402

from strata_mlx import load as load_mod  # noqa: E402
from strata_mlx.dense import AS_BLOCKS, as_blocks  # noqa: E402
from strata_mlx.engine import Engine, _arrays  # noqa: E402

SIZES = (1, 2, 4, 8)


class Both:
    """A dense layer held as 8-bit arrays and as the file's blocks; `Both.blocks` says which is called."""

    blocks = False

    def __init__(self, arrays, blocks):
        # not a module, so the loader's set_dtype does not reach in: float16 scales, as the engine holds them
        # (float32 scales cost MLX's product half its speed, e36)
        arrays.set_dtype(mx.float16)
        self.arrays, self.blocks_lin = arrays, blocks

    def __call__(self, x):
        return (self.blocks_lin if Both.blocks else self.arrays)(x)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--rounds", type=int, default=150)
    p.add_argument("--json")
    args = p.parse_args()

    real = load_mod.linear
    count = [0]

    def linear(t, rows=None, cols=None, as_blocks_=()):
        arrays = real(t, rows, cols)
        if as_blocks(t, AS_BLOCKS):
            count[0] += 1
            return Both(arrays, real(t, rows, cols, AS_BLOCKS))
        return arrays

    load_mod.linear = lambda t, rows=None, cols=None, as_blocks=(): linear(t, rows, cols)
    model, _ = load_mod.load(args.model, dense_blocks=())
    load_mod.linear = real
    print(f"{count[0]} dense layers held both ways", flush=True)

    settings = {"8-bit arrays": False, "file's bits": True, "8-bit arrays, control": False,
                "file's bits, control": True}
    engines = {}
    for name, blocks in settings.items():
        Both.blocks = blocks
        e = Engine(model)
        e.warm()
        e.read(list(range(1000, 1300)))
        engines[name] = e

    times = {(name, T): [] for name in engines for T in SIZES}
    order = list(times) * args.rounds
    random.Random(62).shuffle(order)
    for name, T in order:
        e = engines[name]
        Both.blocks = settings[name]
        t0 = time.perf_counter()
        logits, w = e._forward(mx.zeros((1, T), dtype=mx.int32) + 1000)
        mx.eval(logits, *_arrays([list(w.gdn), w.ple_conv, w.residual]))
        times[(name, T)].append(time.perf_counter() - t0)
        for cache in e.attn.values():
            cache.trim(T)
    rows = []
    for name in engines:
        q = {T: statistics.quantiles(times[(name, T)], n=4)[0] * 1e3 for T in SIZES}
        rows.append({"setting": name, "pass_ms": {T: round(v, 2) for T, v in q.items()}})
        print(f"{name:<22} " + "  ".join(f"{T} token(s) {q[T]:6.2f} ms" for T in SIZES), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
