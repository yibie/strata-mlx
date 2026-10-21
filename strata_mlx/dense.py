"""Dense layers of a type MLX has no exact form for: Q6_K, as a byte a weight under the file's own scales.

MLX holds a quantized weight as `code * scale + bias` with one scale to 32 weights or more; Q6_K scales every
16. Unpacked to floats, as this loader did at first, the Q6_K layers of the IQ3_S file (its delta-net and
attention projections and its output layer) are 4.6 GB of float16 for 1.9 GB of file, read by every token, the
output layer again by every guess of the draft head. A weight is `code * scale * d`: a code of six bits, a scale
of eight for 16 weights, a float16 `d` for 256. Kept as those three, a signed byte to a code, it is the same
number in 8.56 bits.

A chunk of a prompt (more than a window's tokens) multiplies as floats; those are made from the three parts at
the first prompt and kept beside them, so the file's Q6_K layers are then held twice, and answers read the
small form. Only where the machine has the memory for them (load.py, #3): elsewhere they are made anew for each
chunk and let go, and prompts are read a tenth slower.

A kernel on the file's blocks themselves reads 6.56 bits and was tried first (bench/experiments/e20, e21): a
block interleaves the codes' bits, and undoing that for every weight costs more than the bytes it spares once a
window has three tokens or more. In a pass of the IQ3_S model (e23), against float16, a byte a weight: 4.0 ms
less of 26.7 for one token, 3.2 of 42.9 for four, 1.4 of 54.5 for six, 1.7 more for eight.

The kernel is laid out as MLX's own quantized product: 32 threads share four rows, each takes eight weights of
each row at a time and goes along the rows; the activations come with a position's tokens side by side, so one
load brings a weight's activations for the whole window, once for the four rows. Sums are float32.

Below that, layers and an embedding kept as the file's blocks: what a machine that reads experts from the SSD
has in place of MLX's arrays of Q3_K and IQ4_XS tensors, which take more than twice the memory.
"""

import mlx.core as mx
import numpy as np

from .blocks import BLOCK, Q3K_PLANE_BYTES, dequantize, q3k_planes, q3k_planes_dequantize, q3k_planes_window, window_matvec

MAX_TOKENS = 8  # more than a window: a chunk of a prompt, multiplied as floats
_ROWS, _WEIGHTS = 4, 8  # to 32 threads; to a thread at a time
_VEC = {1: "float", 2: "float2", 3: "float3", 4: "float4"}
_PACKED = {1: "float", 2: "packed_float2", 3: "packed_float3", 4: "packed_float4"}


def _source(tokens: int) -> str:
    """x: (cols, tokens) float32, or for one token the activations as they are. For one token the source does
    not name the number of rows: a list of the output layer's rows (rows.py) changes it at every step."""
    chunks = [(o, min(4, tokens - o)) for o in range(0, tokens, 4)]  # (first token, how many) of each vector
    sums = "".join(f"{_VEC[n]} a{r}_{o} = 0.0f; " for r in range(_ROWS) for o, n in chunks)
    if tokens == 1:
        loads = "".join(f"float x{v}_0 = float(x[col + {v}]); " for v in range(_WEIGHTS))
    else:
        loads = "".join(f"{_VEC[n]} x{v}_{o} = {_VEC[n]}(*(const device {_PACKED[n]}*)(x + (col + {v}) * T + {o})); "
                        for v in range(_WEIGHTS) for o, n in chunks)
    rows = ""
    for r in range(_ROWS):
        rows += (f"        {{ uint row = {_ROWS} * group + {r}; const device int8_t* c = codes + row * COLS + col;\n"
                 f"          float s = float(d[row * (COLS / D_STEP) + col / D_STEP]) * float(scales[row * (COLS / 16) + col / 16]);\n")
        for o, n in chunks:
            rows += f"          a{r}_{o} += s * (" + " + ".join(f"float(c[{v}]) * x{v}_{o}" for v in range(_WEIGHTS)) + ");\n"
        rows += "        }\n"
    outs = ""
    for r in range(_ROWS):
        for o, n in chunks:
            at = f"{_ROWS} * group + {r}"
            writes = " ".join(f"out[{at if tokens == 1 else f'{o + j} * ROWS + {at}'}] = static_cast<OutT>({'s' if n == 1 else f's[{j}]'});"
                              for j in range(n))
            outs += f"    {{ {_VEC[n]} s = simd_sum(a{r}_{o}); if (lane == 0) {{ {writes} }} }}\n"
    return f"""
    uint lane = thread_position_in_grid.x % 32;
    uint group = thread_position_in_grid.x / 32;
    {sums}
    for (uint col = lane * {_WEIGHTS}; col < COLS; col += {32 * _WEIGHTS}) {{
        {loads}
{rows}    }}
{outs}"""


# every weight as a float, for a chunk of a prompt: a thread writes the 16 weights of one scale
_FLOATS = """
    uint group = thread_position_in_grid.x, row = thread_position_in_grid.y;
    float s = float(d[row * (COLS / D_STEP) + 16 * group / D_STEP]) * float(scales[row * (COLS / 16) + group]);
    const device int8_t* c = codes + row * COLS + 16 * group;
    auto o = out + row * COLS + 16 * group;
    for (uint i = 0; i < 16; ++i) {
        o[i] = static_cast<OutT>(s * float(c[i]));
    }
"""

_kernels = {}
_holding = []  # layers that hold their weights as floats as well, since a prompt was read


def release():
    """Let go of the float weights made for chunks of prompts. Nothing calls this by itself: made anew for
    every prompt they are 3.3 GB of fresh memory each time (the IQ3_S file), and that, not working them out
    (12 ms), made its prompts read a tenth slower (320 tok/s against 355 for 1,180 tokens)."""
    for layer in _holding:
        layer._floats = None
    _holding.clear()


def fits(rows: int, cols: int) -> bool:
    """Whether the kernel takes a layer of this shape."""
    return rows % _ROWS == 0 and cols % (32 * _WEIGHTS) == 0


def kept_floats(tensors: dict, itemsize: int = 2) -> int:
    """Bytes the weights of a file's Q6_K layers take as floats of `itemsize` once a prompt has been read, if they
    are kept: 3.3 GB in float16 for the IQ3_S file, 0.2 for Q2_0. The output layer never sees a chunk."""
    return sum(itemsize * t.shape[0] * t.shape[1] for name, t in tensors.items()
               if t.type == "Q6_K" and len(t.shape) == 2 and fits(*t.shape) and name != "output.weight")


class ScaledLinear:
    """A linear layer held as codes, scales and d; called like `nn.Linear`."""

    def __init__(self, codes: mx.array, scales: mx.array, d: mx.array):
        self.codes, self.scales, self.d = codes, scales, d
        self.output_dims, self.input_dims = codes.shape
        self.d_step = self.input_dims // d.shape[1]  # weights to a d: 256, or fewer where the inputs were reordered
        self._floats = None
        self.keep = True  # the floats made for a chunk of a prompt stay for the next ones (load.py decides)
        if not fits(*codes.shape):
            raise ValueError(f"a layer of {codes.shape} does not fit the kernel")

    @property
    def nbytes(self) -> int:
        return self.codes.nbytes + self.scales.nbytes + self.d.nbytes

    def floats(self, dtype: mx.Dtype) -> mx.array:
        """The weights (out, in), each worked out in float32 and rounded once to `dtype`. One kernel: as MLX
        operations (the codes as floats, the scales spread out, their product, the rounding) it took half a
        second for the Q6_K layers of the IQ3_S file, for every piece a prompt is read in, and that file's
        prompts were read at 246 tok/s where they had been at 440."""
        rows, cols = self.output_dims, self.input_dims
        key = ("floats", cols, self.d_step)
        if key not in _kernels:
            header = f"constant uint COLS = {cols};\nconstant uint D_STEP = {self.d_step};\n"
            _kernels[key] = mx.fast.metal_kernel(name="scaled_linear_floats", input_names=["codes", "scales", "d"],
                                                 output_names=["out"], source=_FLOATS, header=header)
        (out,) = _kernels[key](inputs=[self.codes, self.scales, self.d], template=[("OutT", dtype)],
                               grid=(cols // 16, rows, 1), threadgroup=(32, 1, 1),
                               output_shapes=[(rows, cols)], output_dtypes=[dtype])
        return out

    def __call__(self, x: mx.array) -> mx.array:
        rows, cols = self.output_dims, self.input_dims
        lead, dtype = x.shape[:-1], x.dtype
        tokens = int(np.prod(lead))
        if tokens > MAX_TOKENS:
            # Chunks of prompts multiply as floats, MLX's product being what it is at many tokens. The floats
            # are made at the first chunk and kept: 3.3 GB for the IQ3_S file beside the 1.8 GB that answers
            # read of the same layers (the output layer never sees a chunk), 0.2 GB for the Q2_0 file.
            if not self.keep:
                return x @ self.floats(dtype).T
            if self._floats is None or self._floats.dtype != dtype:
                self._floats = self.floats(dtype)
                _holding.append(self)
            return x @ self._floats.T
        x = x.reshape(tokens, cols)
        if tokens > 1:
            x = mx.contiguous(x.astype(mx.float32).T)
        key = (rows if tokens > 1 else 0, cols, tokens, self.d_step)
        if key not in _kernels:
            constants = {"COLS": cols, "T": tokens, "D_STEP": self.d_step} | ({"ROWS": rows} if tokens > 1 else {})
            header = "".join(f"constant uint {k} = {v};\n" for k, v in constants.items())
            _kernels[key] = mx.fast.metal_kernel(name=f"scaled_linear_{tokens}", input_names=["codes", "scales", "d", "x"],
                                                 output_names=["out"], source=_source(tokens), header=header)
        (out,) = _kernels[key](inputs=[self.codes, self.scales, self.d, x], template=[("OutT", dtype)],
                               grid=(32 * rows // _ROWS, 1, 1), threadgroup=(32, 1, 1),
                               output_shapes=[(tokens, rows)], output_dtypes=[dtype])
        return out.reshape(*lead, rows)


# ---- layers kept as the file's blocks, for a machine that needs its memory for experts
#
# MLX holds a Q3_K or an IQ4_XS weight at 8 bits, 2.6 and 2.1 times the file's bytes: with the Q2_0 file 2.9 GB
# of arrays for 1.2 GB of file, most of what its dense tensors take beyond their size in the file. The kernels
# the experts have read the blocks as they are (blocks.py).
#
# All of a pass's products of these types, the real model's 146 tensors (bench/experiments/e36), as blocks
# against MLX's arrays: 3.8 against 4.7 ms for one token, 4.3 / 4.8 for two, 5.7 / 4.8 for four, 9.3 / 7.4 for
# eight. A machine with every expert in memory answers four tokens a pass with its draft head and has no use
# for the memory, so it keeps the arrays.

AS_BLOCKS = ("Q3_K", "IQ4_XS")  # the dense types kept as the file's blocks (load.py)
PLANES = True  # Q3_K's blocks rearranged into planes (blocks.q3k_planes)
ARRAY_BYTES = 1.125  # a weight of these as MLX holds it: a byte, and a float16 scale and bias to every 32


def as_blocks(t, kinds: tuple = AS_BLOCKS) -> bool:
    """Whether a dense tensor can stay blocks: of one of `kinds`, and rows that end on a block."""
    return t.type in kinds and len(t.shape) == 2 and t.shape[1] % BLOCK[t.type][0] == 0


def spared(tensors: dict) -> int:
    """Bytes that the dense tensors of a file's first shard take less as blocks than as MLX's arrays. The output
    layer is not among them: it stays arrays whatever its type."""
    return sum(int(ARRAY_BYTES * int(np.prod(t.shape))) - (t.data.nbytes if name == "token_embd.weight" else held_bytes(t))
               for name, t in tensors.items()
               if as_blocks(t) and not name.endswith("_exps.weight") and name != "output.weight")


def held_bytes(t) -> int:
    """What a dense layer kept as blocks takes: the file's bytes, or for Q3_K those of its planes. (The embedding
    keeps the file's blocks: a token's row is unpacked when it comes.)"""
    return t.data.nbytes // 110 * Q3K_PLANE_BYTES if t.type == "Q3_K" and PLANES else t.data.nbytes


class BlockLinear:
    """A linear layer held as the file's blocks; called like `nn.Linear`.

    `blocks` are the rows in the order of the layer's outputs. `inputs`, if given, is where in the file's row
    each input is: the blocks cannot be reordered inside, so the activations are, before they multiply.
    """

    def __init__(self, blocks: mx.array, kind: str, shape: tuple, inputs: np.ndarray | None = None):
        self.blocks, self.kind = blocks, kind
        self.output_dims, self.input_dims = shape
        self.to_file = None if inputs is None else mx.array(np.argsort(inputs).astype(np.uint32))
        self.planes = None
        if kind == "Q3_K" and PLANES:  # held as planes instead (blocks.q3k_planes): faster for windows, e61
            self.planes, self.blocks = q3k_planes(blocks, *shape), None

    @property
    def nbytes(self) -> int:
        return self.blocks.nbytes if self.planes is None else sum(a.nbytes for a in self.planes.values())

    def __call__(self, x: mx.array) -> mx.array:
        rows, cols = self.output_dims, self.input_dims
        lead, dtype = x.shape[:-1], x.dtype
        tokens = int(np.prod(lead))
        x = x.reshape(tokens, cols)
        if self.to_file is not None:
            x = mx.take(x, self.to_file, axis=1)
        if self.planes is not None:
            out = (q3k_planes_window(self.planes, rows, cols, x) if tokens <= MAX_TOKENS
                   else x @ q3k_planes_dequantize(self.planes, rows, cols, dtype).T)
        elif tokens <= MAX_TOKENS:
            out = window_matvec(self.blocks, self.kind, rows, cols, x)
        else:
            # A chunk of a prompt: the weights unpacked for it and let go, at most 63 MB at a time. All of a
            # pass's products of 1,024 tokens (bench/experiments/e36): 290-307 ms so, 310-312 as MLX's 8-bit
            # arrays, 423-454 through the experts' tiles with the layer as a stack of one; and in a pass as a
            # 16 GB machine the 1,537-token prompt at 170 tok/s so, 154 through tiles, the peak of memory the same.
            out = x @ dequantize(self.blocks, self.kind, (rows, cols), dtype).T
        return out.reshape(*lead, rows)


class BlockEmbedding:
    """An embedding held as the file's blocks; a token's row is unpacked when the token comes."""

    def __init__(self, blocks: mx.array, kind: str, shape: tuple, dtype: mx.Dtype):
        self.num_embeddings, self.dims = shape
        self.rows, self.kind, self.dtype = blocks.reshape(self.num_embeddings, -1), kind, dtype

    @property
    def nbytes(self) -> int:
        return self.rows.nbytes

    def __call__(self, ids: mx.array) -> mx.array:
        taken = mx.take(self.rows, ids.reshape(-1), axis=0)
        return dequantize(taken.reshape(-1), self.kind, (ids.size, 1, self.dims), self.dtype).reshape(*ids.shape, self.dims)
