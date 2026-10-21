"""Experts computed on the GGUF file's own blocks, for the types MLX has no exact form for.

MLX holds a quantized weight as `code * scale + bias` with one scale to 32 weights or more. The codebook types
of the larger files do not fit that: IQ2_S, IQ2_XS and IQ1_M scale every 16 weights, and IQ3_XXS and the 4-bit
types fit only at 8 bits, twice their size. Unpacked for MLX the experts of the IQ3_S file would take 94 GB and
twelve minutes of conversion at every start; as they are in the file they take 50 GB and none.

So the blocks go to the GPU as bytes, and Metal kernels read them, each built from one small function per
type, `unpack32`, that turns a run of 32 weights into floats:

    matvec     out[pair, row] = W[expert of pair, row, :] . x     a few threads per row, for decoding
               (also with every pair's expert as an array of its own, for experts paged in from the SSD)
    window     the same for one layer and a window of tokens, its weights unpacked once for all of them
               (the dense layers of a machine that needs its memory for experts: dense.BlockLinear)
    tiles      the same for the many pairs of a chunk of a prompt, sorted by expert into tiles of 16: a
               threadgroup unpacks 64 rows by 64 columns of the tile's expert into its own memory and Apple's
               matrix product for shaders (Metal Performance Primitives) multiplies the tile's tokens by that
    dequant    every weight of the stack as float16 or float32    for shapes the tiles do not take

Q2_0, whose weights are a 2-bit code times a scale per 64, has a way of its own for answers, a table: two
weights make one of 16 sums of two inputs, so those sums are worked out once for the input and a row is then a
lookup for every two weights instead of a multiplication for every one (see `lookup_matvec`).

The tables are ggml's, taken from gguf-py. Sums are float32 whatever the type of the activations.
"""

import mlx.core as mx
import numpy as np
from gguf import quants
from mlx_lm.models.activations import swiglu
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

from .gguf import KVALUES_IQ4NL

# type -> (weights per block, bytes per block)
BLOCK = {"IQ4_NL": (32, 18), "IQ4_XS": (256, 136), "IQ3_S": (256, 110), "IQ3_XXS": (256, 98),
         "IQ2_S": (256, 82), "IQ2_XS": (256, 74), "IQ2_XXS": (256, 66), "IQ1_M": (256, 56), "Q2_0": (64, 18),
         "Q3_K": (256, 110)}

_COMMON = """
inline float half_at(const device uchar* b) {
    return float(as_type<half>(ushort(uint(b[0]) | (uint(b[1]) << 8))));
}
inline uint u16_at(const device uchar* b) { return uint(b[0]) | (uint(b[1]) << 8); }
inline uint u32_at(const device uchar* b) {
    return uint(b[0]) | (uint(b[1]) << 8) | (uint(b[2]) << 16) | (uint(b[3]) << 24);
}
// the k-th of a Q3_K block's sixteen 6-bit scales, less 32: their low four bits in eight bytes, their high two in four
inline float q3k_scale(const device uchar* s, uint k) {
    uint p = k % 4, word = k / 4;
    uint low = word < 2 ? (uint(s[4 * word + p]) & 15) : (uint(s[4 * (word - 2) + p]) >> 4);
    return float(int(low | (((uint(s[8 + p]) >> (2 * word)) & 3) << 4)) - 32);
}
"""

# each: the tables it needs, and unpack32(row, n, w): the n-th run of 32 weights of a row, scaled
_UNPACK = {
    "Q2_0": ((), """
    const device uchar* b = row + (n / 2) * 18;
    const device uchar* q = b + 2 + 8 * (n % 2);
    float d = half_at(b);
    for (uint i = 0; i < 8; ++i) {
        uint c = q[i];
        for (uint j = 0; j < 4; ++j) {
            w[4 * i + j] = d * (float((c >> (2 * j)) & 3) - 1.0f);
        }
    }
"""),
    "IQ4_NL": (("kvalues",), """
    const device uchar* b = row + n * 18;
    float d = half_at(b);
    for (uint i = 0; i < 16; ++i) {
        uchar q = b[2 + i];
        w[i] = d * float(kvalues[q & 15]);
        w[16 + i] = d * float(kvalues[q >> 4]);
    }
"""),
    "IQ4_XS": (("kvalues",), """
    const device uchar* b = row + (n / 8) * 136;
    uint ib = n % 8;
    uint ls = ((uint(b[4 + ib / 2]) >> (4 * (ib % 2))) & 15) | (((u16_at(b + 2) >> (2 * ib)) & 3) << 4);
    float d = half_at(b) * (float(ls) - 32.0f);
    const device uchar* q = b + 8 + 16 * ib;
    for (uint i = 0; i < 16; ++i) {
        w[i] = d * float(kvalues[q[i] & 15]);
        w[16 + i] = d * float(kvalues[q[i] >> 4]);
    }
"""),
    "IQ3_S": (("iq3s_grid",), """
    const device uchar* b = row + (n / 8) * 110;
    uint ib = n % 8;
    float d = half_at(b) * float(1 + 2 * ((uint(b[106 + ib / 2]) >> (4 * (ib % 2))) & 15));
    uint high = b[66 + ib];
    for (uint l = 0; l < 8; ++l) {
        constant uchar* g = iq3s_grid + 4 * (uint(b[2 + 8 * ib + l]) | (((high >> l) & 1) << 8));
        uint sign = uint(b[74 + 4 * ib + l / 2]) >> (4 * (l % 2));
        for (uint j = 0; j < 4; ++j) {
            float v = d * float(g[j]);
            w[4 * l + j] = (sign >> j) & 1 ? -v : v;
        }
    }
"""),
    "IQ3_XXS": (("iq3xxs_grid", "ksigns"), """
    const device uchar* b = row + (n / 8) * 98;
    uint ib = n % 8;
    uint aux = u32_at(b + 66 + 4 * ib);
    float d = half_at(b) * (0.5f + float(aux >> 28)) * 0.5f;
    for (uint l = 0; l < 8; ++l) {
        constant uchar* g = iq3xxs_grid + 4 * uint(b[2 + 8 * ib + l]);
        uint sign = uint(ksigns[(aux >> (7 * (l / 2))) & 127]) >> (4 * (l % 2));
        for (uint j = 0; j < 4; ++j) {
            float v = d * float(g[j]);
            w[4 * l + j] = (sign >> j) & 1 ? -v : v;
        }
    }
"""),
    "IQ2_S": (("iq2s_grid",), """
    const device uchar* b = row + (n / 8) * 82;
    uint ib = n % 8;
    float d = half_at(b) * 0.25f;
    uint high = b[66 + ib], scales = b[74 + ib];
    for (uint l = 0; l < 4; ++l) {
        constant uchar* g = iq2s_grid + 8 * (uint(b[2 + 4 * ib + l]) | (((high >> (2 * l)) & 3) << 8));
        uint sign = b[34 + 4 * ib + l];
        float dl = d * (0.5f + float(l < 2 ? scales & 15 : scales >> 4));
        for (uint j = 0; j < 8; ++j) {
            float v = dl * float(g[j]);
            w[8 * l + j] = (sign >> j) & 1 ? -v : v;
        }
    }
"""),
    "IQ2_XS": (("iq2xs_grid", "ksigns"), """
    const device uchar* b = row + (n / 8) * 74;
    uint ib = n % 8;
    float d = half_at(b) * 0.25f;
    uint scales = b[66 + ib];
    for (uint l = 0; l < 4; ++l) {
        uint q = u16_at(b + 2 + 2 * (4 * ib + l));
        constant uchar* g = iq2xs_grid + 8 * (q & 511);
        uint sign = ksigns[q >> 9];
        float dl = d * (0.5f + float(l < 2 ? scales & 15 : scales >> 4));
        for (uint j = 0; j < 8; ++j) {
            float v = dl * float(g[j]);
            w[8 * l + j] = (sign >> j) & 1 ? -v : v;
        }
    }
"""),
    "IQ2_XXS": (("iq2xxs_grid", "ksigns"), """
    const device uchar* b = row + (n / 8) * 66;
    const device uchar* q = b + 2 + 8 * (n % 8);
    uint aux = u32_at(q + 4);
    float d = half_at(b) * (0.5f + float(aux >> 28)) * 0.25f;
    for (uint l = 0; l < 4; ++l) {
        constant uchar* g = iq2xxs_grid + 8 * uint(q[l]);
        uint sign = ksigns[(aux >> (7 * l)) & 127];
        for (uint j = 0; j < 8; ++j) {
            float v = d * float(g[j]);
            w[8 * l + j] = (sign >> j) & 1 ? -v : v;
        }
    }
"""),
    # a weight is two bits and a third from a mask of the block's first 32 bytes, less 4 where that is unset;
    # the run's two halves each have a scale
    "Q3_K": ((), """
    const device uchar* b = row + (n / 8) * 110;
    uint r = n % 8;
    const device uchar* q = b + 32 + 32 * (r / 4);
    uint shift = 2 * (r % 4);
    float d = half_at(b + 108);
    float s0 = d * q3k_scale(b + 96, 2 * r), s1 = d * q3k_scale(b + 96, 2 * r + 1);
    for (uint i = 0; i < 32; ++i) {
        int v = int((uint(q[i]) >> shift) & 3) - (((uint(b[i]) >> r) & 1) ? 0 : 4);
        w[i] = (i < 16 ? s0 : s1) * float(v);
    }
"""),
    # the block's 16-bit scale is spread over the top four bits of its four scale words
    "IQ1_M": (("iq1_grid",), """
    const device uchar* b = row + (n / 8) * 56;
    uint ib = n % 8;
    uint packed = (u16_at(b + 48) >> 12) | ((u16_at(b + 50) >> 8) & 0x00f0) | ((u16_at(b + 52) >> 4) & 0x0f00)
        | (u16_at(b + 54) & 0xf000);
    float d = float(as_type<half>(ushort(packed)));
    uint scales = u16_at(b + 48 + 2 * (ib / 2)) >> (6 * (ib % 2));
    for (uint l = 0; l < 4; ++l) {
        uint high = (uint(b[32 + 2 * ib + l / 2]) >> (4 * (l % 2))) & 15;
        constant uchar* g = iq1_grid + 8 * (uint(b[4 * ib + l]) | ((high & 7) << 8));
        float dl = d * float(2 * ((scales >> (3 * (l / 2))) & 7) + 1);
        float delta = high & 8 ? -1.125f : -0.875f;  // the table holds the grid's -1, 0, 1 as 0, 1, 2
        for (uint j = 0; j < 8; ++j) {
            w[8 * l + j] = dl * (float(g[j]) + delta);
        }
    }
"""),
}

_MATVEC = """
    uint part = thread_position_in_grid.x % P;
    uint r = thread_position_in_grid.x / P;
    uint pair = thread_position_in_grid.y;
    const device uchar* row = blocks + (uint(idx[pair]) * ROWS + r) * ROW_BYTES;
    auto x_ = x + (pair / SHARE) * COLS;
    float w[32];
    float sum = 0.0f;
    for (uint n = part; n < RUNS; n += P) {
        unpack32(row, n, w);
        auto xr = x_ + 32 * n;
        float s = 0.0f;
        for (uint j = 0; j < 32; ++j) {
            s += w[j] * float(xr[j]);
        }
        sum += s;
    }
    float total = sum;
    for (ushort k = 1; k < P; ++k) {
        total += simd_shuffle_xor(sum, k);  // the other threads of this row
    }
    if (part == 0) {
        out[pair * ROWS + r] = static_cast<OutT>(total);
    }
"""

# down with silu(gate) * up worked out in each threadgroup's memory (a threadgroup is one pair's), as mlx-lm's
# swiglu works it out (sigmoid in float32 rounded to the activations' type, then two products in that type), so
# the same bits, and one kernel fewer to wait on
_MATVEC_GLU = """
    uint part = thread_position_in_grid.x % P;
    uint r = thread_position_in_grid.x / P;
    uint pair = thread_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    threadgroup OutT xs[COLS];
    for (uint c = tid; c < COLS; c += TG) {
        #pragma clang fp contract(off)  // here only: the product below is _MATVEC's, contracted as there
        OutT g = gate[pair * COLS + c];
        float y = 1 / (1 + metal::precise::exp(metal::abs(float(g))));
        OutT sig = static_cast<OutT>((g < 0) ? y : 1 - y);
        OutT a = g * sig;
        xs[c] = a * up[pair * COLS + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device uchar* row = blocks + (uint(idx[pair]) * ROWS + r) * ROW_BYTES;
    float w[32];
    float sum = 0.0f;
    for (uint n = part; n < RUNS; n += P) {
        unpack32(row, n, w);
        const threadgroup OutT* xr = xs + 32 * n;
        float s = 0.0f;
        for (uint j = 0; j < 32; ++j) {
            s += w[j] * float(xr[j]);
        }
        sum += s;
    }
    float total = sum;
    for (ushort k = 1; k < P; ++k) {
        total += simd_shuffle_xor(sum, k);
    }
    if (part == 0) {
        out[pair * ROWS + r] = static_cast<OutT>(total);
    }
"""

_DEQUANT = """
    uint n = thread_position_in_grid.x;
    uint r = thread_position_in_grid.y + ROWS * thread_position_in_grid.z;
    float w[32];
    unpack32(blocks + r * ROW_BYTES, n, w);
    auto o = out + r * COLS + 32 * n;
    for (uint j = 0; j < 32; ++j) {
        o[j] = static_cast<OutT>(w[j]);
    }
"""

_tables = {}


def _table(name: str) -> str:
    """A table of ggml's as Metal source."""
    if name not in _tables:
        if name == "kvalues":
            kind, values = "char", KVALUES_IQ4NL
        elif name == "ksigns":
            kind, values = "uchar", np.frombuffer(quants.IQ2_XXS.ksigns, dtype=np.uint8)
        else:
            cls = {"iq3s_grid": quants.IQ3_S, "iq3xxs_grid": quants.IQ3_XXS, "iq2s_grid": quants.IQ2_S,
                   "iq2xs_grid": quants.IQ2_XS, "iq2xxs_grid": quants.IQ2_XXS, "iq1_grid": quants.IQ1_M}[name]
            cls.init_grid()
            kind, values = "uchar", cls.grid.reshape(-1) + (1 if name == "iq1_grid" else 0)
        values = [int(v) for v in values]
        _tables[name] = f"constant {kind} {name}[{len(values)}] = {{{','.join(map(str, values))}}};\n"
    return _tables[name]


def _header(kind: str, rows: int, cols: int, **constants) -> str:
    per_block, block_bytes = BLOCK[kind]
    if cols % per_block:
        raise ValueError(f"{kind}: rows of {cols} weights do not end on a block of {per_block}")
    tables, body = _UNPACK[kind]
    constants |= {"ROWS": rows, "COLS": cols, "RUNS": cols // 32, "ROW_BYTES": cols // per_block * block_bytes}
    return ("".join(f"constant uint {k} = {v};\n" for k, v in constants.items()) + _COMMON
            + "".join(_table(t) for t in tables)
            + "inline void unpack32(const device uchar* row, uint n, thread float* w) {" + body + "}\n")


def _threads_per_row(runs: int) -> int:
    """As many as divide a row evenly and leave each at least four runs: a thread should not loop long."""
    p = 1
    while p < 32 and runs % (2 * p) == 0 and runs // (2 * p) >= 4:
        p *= 2
    return p


_kernels = {}


def _kernel(what: str, kind: str, rows: int, cols: int, share: int = 1, per_row: int = 1):
    key = (what, kind, rows, cols, share, per_row)
    if key not in _kernels:
        if what == "matvec":
            _kernels[key] = mx.fast.metal_kernel(
                name=f"matvec_{kind.lower()}", input_names=["blocks", "idx", "x"], output_names=["out"],
                source=_MATVEC, header=_header(kind, rows, cols, P=per_row, SHARE=share))
        else:
            _kernels[key] = mx.fast.metal_kernel(
                name=f"dequant_{kind.lower()}", input_names=["blocks"], output_names=["out"],
                source=_DEQUANT, header=_header(kind, rows, cols))
    return _kernels[key]


MAX_SEPARATE = 29  # Metal binds 31 buffers to a kernel: these, the activations and the output


def matvec_separate(experts: list, kind: str, rows: int, cols: int, x: mx.array, share: int) -> mx.array:
    """out[pair] = W[pair] . x, where every pair brings its expert's blocks as an array of its own, so that
    nothing has to be joined first. x (pairs / share, cols): one row to every `share` pairs. Returns (pairs, rows)."""
    n, per_row = len(experts), _threads_per_row(cols // 32)
    key = ("separate", kind, rows, cols, share, per_row, n)
    if key not in _kernels:
        pick = " ".join(f"pair == {k} ? b{k} :" for k in range(n - 1)) + f" b{n - 1}"
        source = _MATVEC.replace("blocks + (uint(idx[pair]) * ROWS + r) * ROW_BYTES", f"({pick}) + r * ROW_BYTES")
        assert source != _MATVEC
        _kernels[key] = mx.fast.metal_kernel(
            name=f"matvec_{kind.lower()}_{n}", input_names=[f"b{k}" for k in range(n)] + ["x"], output_names=["out"],
            source=source, header=_header(kind, rows, cols, P=per_row, SHARE=share))
    (out,) = _kernels[key](
        inputs=[*experts, x], template=[("OutT", x.dtype)], grid=(per_row * rows, n, 1), threadgroup=(32, 1, 1),
        output_shapes=[(n, rows)], output_dtypes=[x.dtype])
    return out


# One layer, not a stack, and a window of a few tokens: a run of weights is unpacked once and multiplies every
# token's activations. Each token's sum is taken in the same order whatever the window.
_WINDOW = """
    uint part = thread_position_in_grid.x % P;
    uint r = thread_position_in_grid.x / P;
    const device uchar* row = blocks + r * ROW_BYTES;
    float w[32];
    float sum[T];
    for (uint t = 0; t < T; ++t) {
        sum[t] = 0.0f;
    }
    for (uint n = part; n < RUNS; n += P) {
        unpack32(row, n, w);
        for (uint t = 0; t < T; ++t) {
            auto xr = x + t * COLS + 32 * n;
            float s = 0.0f;
            for (uint j = 0; j < 32; ++j) {
                s += w[j] * float(xr[j]);
            }
            sum[t] += s;
        }
    }
    for (uint t = 0; t < T; ++t) {
        float total = sum[t];
        for (ushort k = 1; k < P; ++k) {
            total += simd_shuffle_xor(sum[t], k);  // the other threads of this row
        }
        if (part == 0) {
            out[t * ROWS + r] = static_cast<OutT>(total);
        }
    }
"""


def window_matvec(blocks: mx.array, kind: str, rows: int, cols: int, x: mx.array, per_row: int | None = None) -> mx.array:
    """out[t] = W . x[t] for one layer `blocks` of (rows, cols) and x (tokens, cols). Returns (tokens, rows)."""
    tokens, per_row = x.shape[0], per_row or _threads_per_row(cols // 32)
    key = ("window", kind, rows, cols, tokens, per_row)
    if key not in _kernels:
        _kernels[key] = mx.fast.metal_kernel(
            name=f"window_{kind.lower()}_{tokens}", input_names=["blocks", "x"], output_names=["out"],
            source=_WINDOW, header=_header(kind, rows, cols, P=per_row, T=tokens))
    (out,) = _kernels[key](
        inputs=[blocks, x], template=[("OutT", x.dtype)], grid=(per_row * rows, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(tokens, rows)], output_dtypes=[x.dtype])
    return out


# Q3_K held as planes: a tensor's blocks rearranged once into aligned arrays of their parts, the same bits of
# weight; the 12 bytes of a block's 6-bit scales become 16 int8, so 114 bytes a block where the file has 110. A
# thread then reads its words where e36's kernel read bytes, eight times over: a 10,240 x 2,560 tensor of the
# Q2_0 file at 1 / 2 / 4 / 8 tokens in 67 / 64 / 72 / 109 us, MLX's 8-bit form of it in 80 / 81 / 83 / 116 (e61).
Q3K_PLANE_BYTES = 114  # a block's, against the file's 110

def q3k_planes(blocks: mx.array, rows: int, cols: int) -> dict:
    """A Q3_K tensor's blocks (rows, cols as the file has them) -> its planes."""
    b = np.array(blocks).reshape(rows, cols // 256, 110)
    sc12 = b[..., 96:108]
    sc = np.empty(b.shape[:-1] + (16,), dtype=np.int16)
    for k in range(16):  # as q3k_scale
        p, word = k % 4, k // 4
        low = (sc12[..., 4 * word + p] & 15) if word < 2 else (sc12[..., 4 * (word - 2) + p] >> 4)
        sc[..., k] = (low.astype(np.int16) | (((sc12[..., 8 + p] >> (2 * word)) & 3).astype(np.int16) << 4)) - 32
    return {"hm": mx.array(np.ascontiguousarray(b[..., 0:32]).reshape(-1).view(np.uint32)),
            "qs": mx.array(np.ascontiguousarray(b[..., 32:96]).reshape(-1).view(np.uint32)),
            "sc": mx.array(sc.astype(np.int8).reshape(-1)),
            "dd": mx.array(np.ascontiguousarray(b[..., 108:110]).reshape(-1).view(np.float16))}


# the weight `i` (0-31) of run `rr` (0-7) of a block, from that half's words: as unpack32 of Q3_K has it
_Q3K_WEIGHT = """
inline int q3k_value(uint qw, uint hw, uint k, uint j, uint rr) {
    return int((qw >> (8 * k + 2 * j)) & 3) - (((hw >> (8 * k + rr)) & 1) ? 0 : 4);
}
"""

_Q3K_PLANES_WINDOW = """
    uint part = thread_position_in_grid.x % P;
    uint r = thread_position_in_grid.x / P;
    float sum[T];
    for (uint t = 0; t < T; ++t) sum[t] = 0.0f;
    for (uint h = part; h < 2 * BLOCKS; h += P) {  // half a block, 128 weights, at a time
        uint blk = r * BLOCKS + h / 2, n = h % 2;
        const device uint* hmw = hm + blk * 8;
        const device uint* qsw = qs + blk * 16 + 8 * n;
        const device int8_t* scl = sc + blk * 16;
        float d = float(dd[blk]);
        uint hwords[8], qwords[8];
        for (uint w = 0; w < 8; ++w) { hwords[w] = hmw[w]; qwords[w] = qsw[w]; }
        uint base = (h / 2) * 256 + 128 * n;
        for (uint j = 0; j < 4; ++j) {
            uint rr = 4 * n + j;
            float acc0[T], acc1[T];
            for (uint t = 0; t < T; ++t) { acc0[t] = 0.0f; acc1[t] = 0.0f; }
            for (uint w4 = 0; w4 < 8; ++w4) {
                float4 v;
                for (uint k = 0; k < 4; ++k) v[k] = float(q3k_value(qwords[w4], hwords[w4], k, j, rr));
                uint xi = base + 32 * j + 4 * w4;
                for (uint t = 0; t < T; ++t) {
                    float4 xx = float4(*(const device metal::vec<InT, 4>*)(x + t * COLS + xi));
                    if (w4 < 4) acc0[t] += dot(v, xx); else acc1[t] += dot(v, xx);
                }
            }
            float s0 = d * float(scl[2 * rr]), s1 = d * float(scl[2 * rr + 1]);
            for (uint t = 0; t < T; ++t) sum[t] += s0 * acc0[t] + s1 * acc1[t];
        }
    }
    for (uint t = 0; t < T; ++t) {
        float total = sum[t];
        for (ushort k = 1; k < P; k <<= 1) total += simd_shuffle_xor(total, k);
        if (part == 0) out[t * ROWS + r] = static_cast<OutT>(total);
    }
"""

# the weights as floats, for a chunk of a prompt: d * scale * value, as dequantize gives them from the blocks
_Q3K_PLANES_DEQUANT = """
    uint blk = thread_position_in_grid.x;  // a thread to a run of 32: (row, block, run)
    uint rr = blk % 8;
    blk /= 8;
    uint n = rr / 4, j = rr % 4;
    const device uint* hmw = hm + blk * 8;
    const device uint* qsw = qs + blk * 16 + 8 * n;
    float d = float(dd[blk]);
    float s0 = d * float(sc[blk * 16 + 2 * rr]), s1 = d * float(sc[blk * 16 + 2 * rr + 1]);
    auto o = out + blk * 256 + 32 * rr;
    for (uint w4 = 0; w4 < 8; ++w4) {
        for (uint k = 0; k < 4; ++k) {
            o[4 * w4 + k] = static_cast<OutT>((w4 < 4 ? s0 : s1) * float(q3k_value(qsw[w4], hmw[w4], k, j, rr)));
        }
    }
"""


def q3k_planes_window(planes: dict, rows: int, cols: int, x: mx.array) -> mx.array:
    """out[t] = W . x[t] for a Q3_K tensor held as planes, x (tokens, cols). Returns (tokens, rows)."""
    T = x.shape[0]
    # Threads to a row by the tensor's shape, never by the window: a token must get the same bits alone as in a
    # window, which a different split of the row's sum would not give it. Two for 10,240 rows (e61); more where
    # rows are few, so that a tensor of 640 rows has threads enough to fill the GPU.
    per_row = 2
    while per_row < 16 and rows * per_row < 20480 and (2 * cols // 256) // (2 * per_row) >= 1:
        per_row *= 2
    key = ("q3k_planes", rows, cols, T, per_row, x.dtype)
    if key not in _kernels:
        header = ("".join(f"constant uint {k} = {v};\n" for k, v in
                          {"ROWS": rows, "COLS": cols, "BLOCKS": cols // 256, "P": per_row, "T": T}.items())
                  + _Q3K_WEIGHT)
        _kernels[key] = mx.fast.metal_kernel(name=f"q3k_planes_{T}", input_names=["hm", "qs", "sc", "dd", "x"],
                                             output_names=["out"], source=_Q3K_PLANES_WINDOW, header=header)
    return _kernels[key](inputs=[planes["hm"], planes["qs"], planes["sc"], planes["dd"], x],
                         template=[("InT", x.dtype), ("OutT", x.dtype)], grid=(per_row * rows, 1, 1),
                         threadgroup=(64, 1, 1),
                         output_shapes=[(T, rows)], output_dtypes=[x.dtype])[0]


def q3k_planes_dequantize(planes: dict, rows: int, cols: int, dtype: mx.Dtype = mx.float32) -> mx.array:
    key = ("q3k_planes_dequant",)
    if key not in _kernels:
        _kernels[key] = mx.fast.metal_kernel(name="q3k_planes_dequant", input_names=["hm", "qs", "sc", "dd"],
                                             output_names=["out"], source=_Q3K_PLANES_DEQUANT, header=_Q3K_WEIGHT)
    runs = rows * cols // 32
    return _kernels[key](inputs=[planes["hm"], planes["qs"], planes["sc"], planes["dd"]], template=[("OutT", dtype)],
                         grid=(runs, 1, 1), threadgroup=(min(256, runs), 1, 1),
                         output_shapes=[(rows, cols)], output_dtypes=[dtype])[0]


def dequantize(blocks: mx.array, kind: str, shape: tuple, dtype: mx.Dtype = mx.float32) -> mx.array:
    """The weights of `blocks` (uint8, a tensor of `shape` as the file holds it) as floats."""
    *lead, rows, cols = shape
    n = int(np.prod(lead)) if lead else 1
    (out,) = _kernel("dequant", kind, rows, cols)(
        inputs=[blocks], template=[("OutT", dtype)], grid=(cols // 32, rows, n), threadgroup=(32, 1, 1),
        output_shapes=[tuple(shape)], output_dtypes=[dtype])
    return out


class BlockSwitchLinear:
    """A stack of linear layers kept as the file's blocks; called like mlx-lm's `QuantizedSwitchLinear`."""

    # From this many (token, expert) pairs on, the stack is unpacked once and multiplied as floats: 1.7 GB for a
    # stack of the real model, 5.9 ms to unpack and 5.2 for MLX's gather_mm. Only for shapes `tile_matmul` does
    # not take; a prompt's pairs otherwise go there (switch_glu) before they get here.
    unpack_from = 5000

    def __init__(self, blocks: mx.array, kind: str, shape: tuple, per_row: int | None = None):
        self.blocks, self.kind = blocks, kind
        self.num_experts, self.output_dims, self.input_dims = shape
        self.per_row = per_row or _threads_per_row(self.input_dims // 32)

    def __call__(self, x: mx.array, indices: mx.array, sorted_indices: bool = False) -> mx.array:
        """x (..., 1, in), one row to each index or to each run of the last axis of `indices`."""
        batch, k = x.shape[:-2], indices.shape[-1] if indices.ndim else 1
        if batch == indices.shape:
            share = 1
        elif batch == (*indices.shape[:-1], 1):
            share = k
        else:
            x, share = mx.broadcast_to(x, (*indices.shape, 1, self.input_dims)), 1
        pairs = indices.size
        if pairs >= self.unpack_from:
            return self._as_floats(x, indices, sorted_indices)
        (out,) = _kernel("matvec", self.kind, self.output_dims, self.input_dims, share, self.per_row)(
            inputs=[self.blocks, indices.astype(mx.uint32).reshape(-1), x.reshape(-1, self.input_dims)],
            template=[("OutT", x.dtype)], grid=(self.per_row * self.output_dims, pairs, 1),
            threadgroup=(32, 1, 1), output_shapes=[(pairs, self.output_dims)], output_dtypes=[x.dtype])
        return out.reshape(*indices.shape, 1, self.output_dims)

    def _as_floats(self, x: mx.array, indices: mx.array, sorted_indices: bool) -> mx.array:
        w = dequantize(self.blocks, self.kind, (self.num_experts, self.output_dims, self.input_dims), x.dtype)
        return mx.gather_mm(x, w.swapaxes(-1, -2), rhs_indices=indices, sorted_indices=sorted_indices)


# ---- prompts: an expert's blocks a tile at a time, under Apple's matrix product
#
# A chunk of a prompt is thousands of pairs for a few hundred experts of a layer (1,024 tokens: 10,240 pairs for
# 340 to 395 experts, 7 to 13 at the median, some hundreds at most). The pairs are sorted by expert into tiles of
# TOKENS_TILE; a threadgroup takes a tile and ROWS_TILE rows of its expert, unpacks COLS_TILE columns of those
# rows into threadgroup memory (`unpack32`, so every type), and the primitive multiplies the tile's tokens by
# that; then the next columns, into the same memory.
#
# Measured on an M4 Max, layers of a real pass (bench/experiments/e31, e33), a product of 10,220 pairs:
# 4.5-5.4 ms whatever the type, 6.3-7.4 T operations a second. Before, Q2_0 went through the primitive's own
# 2-bit operand, which needs the stack split into codes and scales and the activations summed by 64 (1.4 + 0.8
# + 4.6 ms), and the other types were unpacked whole to 1.7 GB of floats for MLX's gather_mm (5.9 + 5.2 ms).
# A layer's three products: 16 ms, from 22 and from 43 to 79. Tiles of 32 tokens, or 128 columns at a time,
# are a fifth slower; 256 columns are all the memory a threadgroup has, and came out wrong.
#
# In a pass (e30), a piece of a prompt in tiles against a pair at a time (the table of sums for Q2_0, the matvec
# kernel for the rest), tok/s: Q2_0 128 tokens 283 / 309, 192 429 / 362, 256 476 / 368, 1,024 679 / 378; the
# IQ3_S file 128 tokens 166 / 178, 192 251 / 241, 256 337 / 277. A tile costs the same with one pair in it as
# with sixteen, and a short piece has few pairs to an expert.
TILES_FROM = 1600  # (token, expert) pairs from which a prompt goes this way
TOKENS_TILE, ROWS_TILE, COLS_TILE, _SIMD_GROUPS = 16, 64, 64, 2

_TILES = """
    uint tx = threadgroup_position_in_grid.x, ty = threadgroup_position_in_grid.y;
    uint expert = uint(tile_expert[ty]);
    if (expert >= EXPERTS) {
        return;  // a tile beyond those the pairs fill: there are as many as they could need at most
    }
    threadgroup InT weights[TN * TK];
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false);
    matmul2d<desc, execution_simdgroups<SIMD_GROUPS>> op;
    tensor<device InT, dextents<int32_t, 2>, tensor_inline> A((device InT *)a, dextents<int32_t, 2>(COLS, int(a_shape[0])));
    tensor<threadgroup InT, dextents<int32_t, 2>, tensor_inline> W((threadgroup InT *)weights, dextents<int32_t, 2>(TK, TN));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> C(c, dextents<int32_t, 2>(ROWS, int(a_shape[0])));
    auto tA = A.template slice<TK, TM>(0, ty * TM);  // `template`: A's type hangs on InT
    auto acc = op.get_destination_cooperative_tensor<decltype(tA), decltype(W), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            acc[i] = 0;
        }
    }
    const device uchar* rows = blocks + (expert * ROWS + tx * TN) * ROW_BYTES;
    float w[32];
    for (uint kt = 0; kt < COLS / TK; ++kt) {
        // the tile's weights: TN rows of TK / 32 runs, shared out among the threads
        for (uint u = thread_index_in_threadgroup; u < TN * (TK / 32); u += 32 * SIMD_GROUPS) {
            uint r = u / (TK / 32), n = u % (TK / 32);
            unpack32(rows + r * ROW_BYTES, kt * (TK / 32) + n, w);
            for (uint j = 0; j < 32; ++j) {
                weights[r * TK + 32 * n + j] = InT(w[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tA = A.template slice<TK, TM>(kt * TK, ty * TM);
        auto part = op.get_destination_cooperative_tensor<decltype(tA), decltype(W), float>();
        op.run(tA, W, part);  // sets, does not add
        for (uint16_t i = 0; i < part.get_capacity(); ++i) {
            if (part.is_valid_element(i)) {
                acc[i] += part[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // all have read the weights before the next are written
    }
    auto tC = C.slice<TN, TM>(tx * TN, ty * TM);
    acc.store(tC);
"""


def tileable(kind: str, rows: int, cols: int) -> bool:
    """Whether `tile_matmul` takes a stack of such layers."""
    return rows % ROWS_TILE == 0 and cols % COLS_TILE == 0 and cols % BLOCK[kind][0] == 0


def tile_matmul(blocks: mx.array, kind: str, shape: tuple, a: mx.array, tile_expert: mx.array) -> mx.array:
    """out[p] = W[expert of p's tile] . a[p] for a stack `blocks` of `shape` (experts, rows, cols).

    a (P, cols) float16 or float32, P a multiple of TOKENS_TILE, every TOKENS_TILE rows for one expert;
    tile_expert (P / TOKENS_TILE,) says which, or names no expert (the number of experts, or more) for a tile to
    leave out: its rows of the result are then whatever was in memory. The weights are rounded to the type of
    `a` before they multiply; sums are float32. Returns (P, rows) float32.
    """
    experts, rows, cols = shape
    key = ("tiles", kind, rows, cols, experts)
    if key not in _kernels:
        header = ("#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;\n"
                  + _header(kind, rows, cols, TM=TOKENS_TILE, TN=ROWS_TILE, TK=COLS_TILE, SIMD_GROUPS=_SIMD_GROUPS,
                            EXPERTS=experts))
        _kernels[key] = mx.fast.metal_kernel(name=f"tiles_{kind.lower()}", input_names=["blocks", "a", "tile_expert"],
                                             output_names=["c"], source=_TILES, header=header)
    width, p = 32 * _SIMD_GROUPS, a.shape[0]
    (out,) = _kernels[key](inputs=[blocks, a, tile_expert], template=[("InT", a.dtype)],
                           grid=(width * (rows // ROWS_TILE), p // TOKENS_TILE, 1), threadgroup=(width, 1, 1),
                           output_shapes=[(p, rows)], output_dtypes=[mx.float32])
    return out


def tiles_of(indices: mx.array, experts: int) -> tuple:
    """The pairs of `indices` (..., k) laid out in tiles of TOKENS_TILE, each tile one expert's.

    -> (the expert of each tile, or `experts` for a tile nothing fills; the token of each slot; the slot of
    each pair). All of it is worked out by the GPU, in a number of tiles that depends only on the number of
    pairs: as many as they could need, however the routers chose. Sorting on the CPU instead made a pass stop
    at every layer until the GPU had the routers' choice, and those stops were a third of the time a prompt
    took (bench/experiments/e32: 1,024 tokens in 2.6 s, in 1.8 with the choices known beforehand).
    """
    k, flat = indices.shape[-1], indices.reshape(-1).astype(mx.int32)
    pairs = flat.size
    order = mx.argsort(flat)
    expert = flat[order]  # of each pair, in order of expert
    counts = (flat[:, None] == mx.arange(experts, dtype=mx.int32)[None]).sum(axis=0)
    tiles = (counts + (TOKENS_TILE - 1)) // TOKENS_TILE
    tile_end, pair_end = mx.cumsum(tiles), mx.cumsum(counts)
    slot = ((tile_end - tiles)[expert] * TOKENS_TILE + mx.arange(pairs, dtype=mx.int32) - (pair_end - counts)[expert])
    most = pairs // TOKENS_TILE + min(experts, pairs)  # every expert's last tile may be all but empty
    token = mx.zeros((most * TOKENS_TILE,), dtype=mx.int32)
    token[slot] = order.astype(mx.int32) // k  # what fills a tile is token 0, computed and thrown away
    where = mx.zeros((pairs,), dtype=mx.int32)
    where[order] = slot
    # a tile's expert is the first whose tiles end after it: as many experts end at or before it
    tile_expert = (tile_end[None] <= mx.arange(most, dtype=mx.int32)[:, None]).sum(axis=1)
    return tile_expert.astype(mx.uint32), token.astype(mx.uint32), where.astype(mx.uint32)


def _tiled_glu(x: mx.array, indices: mx.array, up, gate, down) -> mx.array:
    """The three products of a prompt's pairs through `tile_matmul`, the pairs in tiles (`tiles_of`)."""
    tiles, token, where = tiles_of(indices, gate.num_experts)
    a = x.reshape(-1, x.shape[-1])[token]

    def product(lin, a):
        return tile_matmul(lin.blocks, lin.kind, (lin.num_experts, lin.output_dims, lin.input_dims), a, tiles)

    h = swiglu(product(gate, a), product(up, a)).astype(x.dtype)
    out = product(down, h)[where].astype(x.dtype)
    return out.reshape(*indices.shape, down.output_dims)


# ---- Q2_0 for answers: a table of sums instead of a multiplication for every weight
#
# The matvec kernel above is bound by arithmetic: about eight operations a weight, at all the operations this
# GPU has (bench/experiments/e02, e03, e10, e11: neither the inner loop's shape, nor the threads to a row, nor
# loading four values at once, nor sharing the activations between rows changed its time; leaving the
# arithmetic out did). Two Q2_0 weights are each -1, 0, 1 or 2 times their scale: of two inputs they make one
# of 16 sums. With those sums in a table, a row costs about three operations a weight. A table of the 256 sums
# of four weights would halve that again and was four times slower: 650 KB for a token do not stay in a cache,
# 41 KB do, and only as float16. Hence float16 activations only; a sum then carries about one more rounding of
# a float16 than before.
#
# In a pass, against the matvec kernel (e12): 0.4 ms less for one token, 1.2 for two, 3.4 for four, 9.4 for
# eight. Few pairs want many threads to a row, many pairs few.

LOOKUP_FROM = 1  # (token, expert) pairs from which Q2_0 experts over float16 go this way

_TABLE = """
    uint g = thread_position_in_grid.x;
    uint n = thread_position_in_grid.y;
    float a = float(x[n * COLS + 2 * g]), b = float(x[n * COLS + 2 * g + 1]);
    auto t = table + (n * (COLS / 2) + g) * 16;
    for (uint c = 0; c < 16; ++c) {
        t[c] = static_cast<OutT>((float(c & 3) - 1.0f) * a + (float(c >> 2) - 1.0f) * b);
    }
"""

# the same table for the experts' inner activations, silu(gate) * up, without those ever being an array
_TABLE_GLU = """
    uint g = thread_position_in_grid.x;
    uint n = thread_position_in_grid.y;
    float ga = float(gate[n * COLS + 2 * g]), gb = float(gate[n * COLS + 2 * g + 1]);
    float a = ga / (1.0f + metal::exp(-ga)) * float(up[n * COLS + 2 * g]);
    float b = gb / (1.0f + metal::exp(-gb)) * float(up[n * COLS + 2 * g + 1]);
    auto t = table + (n * (COLS / 2) + g) * 16;
    for (uint c = 0; c < 16; ++c) {
        t[c] = static_cast<OutT>((float(c & 3) - 1.0f) * a + (float(c >> 2) - 1.0f) * b);
    }
"""

_LOOKUP = """
    uint part = thread_position_in_grid.x % P;
    uint r = thread_position_in_grid.x / P;
    uint pair = thread_position_in_grid.y;
    const device uchar* row = blocks + (uint(idx[pair]) * ROWS + r) * ROW_BYTES;
    auto t_ = table + (pair / SHARE) * (COLS / 2) * 16;
    float sum = 0.0f;
    for (uint n = part; n < RUNS; n += P) {
        const device uchar* b = row + (n / 2) * 18;
        const device uchar* q = b + 2 + 8 * (n % 2);
        auto t = t_ + 256 * n;
        float s = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            uint c = q[i];
            s += float(t[32 * i + (c & 15)]) + float(t[32 * i + 16 + (c >> 4)]);
        }
        sum += half_at(b) * s;
    }
    float total = sum;
    for (ushort k = 1; k < P; ++k) {
        total += simd_shuffle_xor(sum, k);  // the other threads of this row
    }
    if (part == 0) {
        out[pair * ROWS + r] = static_cast<OutT>(total);
    }
"""


def lookable(lin, dtype: mx.Dtype) -> bool:
    return lin.kind == "Q2_0" and dtype == mx.float16


def lookup_table(x: mx.array) -> mx.array:
    """x (n, cols) float16 -> (n, cols / 2, 16): for every two inputs, the sums two weights can make of them."""
    n, cols = x.shape
    key = ("table", cols)
    if key not in _kernels:
        _kernels[key] = mx.fast.metal_kernel(name="q2_table", input_names=["x"], output_names=["table"],
                                             source=_TABLE, header=f"constant uint COLS = {cols};\n")
    (table,) = _kernels[key](inputs=[x], template=[("OutT", x.dtype)], grid=(cols // 2, n, 1), threadgroup=(32, 1, 1),
                             output_shapes=[(n, cols // 2, 16)], output_dtypes=[x.dtype])
    return table


def lookup_table_glu(gate: mx.array, up: mx.array) -> mx.array:
    """The table of silu(gate) * up, both (n, cols) float16."""
    n, cols = gate.shape
    key = ("table_glu", cols)
    if key not in _kernels:
        _kernels[key] = mx.fast.metal_kernel(name="q2_table_glu", input_names=["gate", "up"], output_names=["table"],
                                             source=_TABLE_GLU, header=f"constant uint COLS = {cols};\n")
    (table,) = _kernels[key](inputs=[gate, up], template=[("OutT", gate.dtype)], grid=(cols // 2, n, 1),
                             threadgroup=(32, 1, 1), output_shapes=[(n, cols // 2, 16)], output_dtypes=[gate.dtype])
    return table


def lookup_matvec(lin, idx: mx.array, table: mx.array, share: int) -> mx.array:
    """out[pair] = W[idx[pair]] . x[pair / share], x given as its `lookup_table`. Returns (pairs, rows)."""
    pairs = idx.shape[0]
    # Threads to a row (e44: the real model's 48 layers chained, the rule alternated with the one before, which
    # was 16 below 21 pairs and 4 from there): rows of 2,560 inputs want 8 up to 40 pairs (four tokens) and 2
    # beyond. The experts' down rows keep that rule: their table is built in `lookup_down_glu` since pull request
    # 9, which must give these bits, and with it the change measured 0.4 ms less a pass at four tokens and 3.0 at
    # eight on an M4 Max, in one process with the passes shuffled.
    per_row = _threads_per_row(lin.input_dims // 32)
    if lin.input_dims >= 2048:
        per_row = min(per_row, 8 if pairs <= 40 else 2)
    elif pairs > 20:
        per_row = min(per_row, 4)
    key = ("lookup", lin.output_dims, lin.input_dims, share, per_row)
    if key not in _kernels:
        _kernels[key] = mx.fast.metal_kernel(
            name="q2_lookup", input_names=["blocks", "idx", "table"], output_names=["out"], source=_LOOKUP,
            header=_header("Q2_0", lin.output_dims, lin.input_dims, P=per_row, SHARE=share))
    (out,) = _kernels[key](inputs=[lin.blocks, idx, table], template=[("OutT", table.dtype)],
                           grid=(per_row * lin.output_dims, pairs, 1), threadgroup=(32, 1, 1),
                           output_shapes=[(pairs, lin.output_dims)], output_dtypes=[table.dtype])
    return out


# gate and up in one kernel: they read the same table of the same input, and a kernel reading a layer's ten
# experts' 4.6 MB does not reach the bandwidth (e56: any reader of them takes 2.5 times the bytes' time), so two
# such kernels cost more than one of twice the rows. Each row is summed as `lookup_matvec` sums it: the same bits.
# A pass of 1 / 4 tokens 0.4-0.7 ms shorter on the M4 Max (e57).
GATE_UP_ONE_KERNEL = True
_LOOKUP_TWO = (_LOOKUP
               .replace("    uint pair = thread_position_in_grid.y;\n",
                        "    uint pair = thread_position_in_grid.y;\n    bool second = r >= ROWS;\n"
                        "    if (second) r -= ROWS;\n", 1)
               .replace("blocks + (uint(idx[pair])", "(second ? blocks_b : blocks_a) + (uint(idx[pair])", 1)
               .replace("        out[pair * ROWS + r] = static_cast<OutT>(total);",
                        "        if (second) out_b[pair * ROWS + r] = static_cast<OutT>(total);\n"
                        "        else out_a[pair * ROWS + r] = static_cast<OutT>(total);", 1))


def lookup_gate_up(gate, up, idx: mx.array, table: mx.array, share: int) -> tuple:
    """(`lookup_matvec(gate, ...)`, `lookup_matvec(up, ...)`) in one kernel; gate and up of one shape."""
    pairs = idx.shape[0]
    per_row = _threads_per_row(gate.input_dims // 32)  # as lookup_matvec chooses it
    if gate.input_dims >= 2048:
        per_row = min(per_row, 8 if pairs <= 40 else 2)
    elif pairs > 20:
        per_row = min(per_row, 4)
    key = ("lookup_two", gate.output_dims, gate.input_dims, share, per_row)
    if key not in _kernels:
        _kernels[key] = mx.fast.metal_kernel(
            name="q2_lookup_two", input_names=["blocks_a", "blocks_b", "idx", "table"],
            output_names=["out_a", "out_b"], source=_LOOKUP_TWO,
            header=_header("Q2_0", gate.output_dims, gate.input_dims, P=per_row, SHARE=share))
    shape = (pairs, gate.output_dims)
    g, u = _kernels[key](inputs=[gate.blocks, up.blocks, idx, table], template=[("OutT", table.dtype)],
                         grid=(per_row * gate.output_dims * 2, pairs, 1), threadgroup=(32, 1, 1),
                         output_shapes=[shape, shape], output_dtypes=[table.dtype, table.dtype])
    return g, u


# down with the table of silu(gate) * up built in each threadgroup's memory (a threadgroup is one pair's), not by
# a kernel of its own: the same table, the same bits, one kernel fewer for the layer to wait on. On an M5 Ultra,
# a pass of 1 / 4 / 8 / 10 tokens with Q2_0, 0.13 / 0.15 / 0.47 / 0.71 ms less (four runs alternated).
GLU_IN_DOWN = True
_DOWN_GLU = """
    uint part = thread_position_in_grid.x % P;
    uint r = thread_position_in_grid.x / P;
    uint pair = thread_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    threadgroup OutT tab[(COLS / 2) * 16];
    for (uint g = tid; g < COLS / 2; g += TG) {
        float ga = float(gate[pair * COLS + 2 * g]), gb = float(gate[pair * COLS + 2 * g + 1]);
        float a = ga / (1.0f + metal::exp(-ga)) * float(up[pair * COLS + 2 * g]);
        float b = gb / (1.0f + metal::exp(-gb)) * float(up[pair * COLS + 2 * g + 1]);
        for (uint c = 0; c < 16; ++c) {
            tab[g * 16 + c] = static_cast<OutT>((float(c & 3) - 1.0f) * a + (float(c >> 2) - 1.0f) * b);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device uchar* row = blocks + (uint(idx[pair]) * ROWS + r) * ROW_BYTES;
    float sum = 0.0f;
    for (uint n = part; n < RUNS; n += P) {
        const device uchar* b = row + (n / 2) * 18;
        const device uchar* q = b + 2 + 8 * (n % 2);
        const threadgroup OutT* t = tab + 256 * n;
        float s = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            uint c = q[i];
            s += float(t[32 * i + (c & 15)]) + float(t[32 * i + 16 + (c >> 4)]);
        }
        sum += half_at(b) * s;
    }
    float total = sum;
    for (ushort k = 1; k < P; ++k) {
        total += simd_shuffle_xor(sum, k);
    }
    if (part == 0) {
        out[pair * ROWS + r] = static_cast<OutT>(total);
    }
"""


def lookup_down_glu(lin, idx: mx.array, gate: mx.array, up: mx.array) -> mx.array | None:
    """`lookup_matvec(lin, idx, lookup_table_glu(gate, up), 1)` in one kernel, or None where the table would not
    fit a threadgroup's memory or the rows not divide into threadgroups."""
    pairs = idx.shape[0]
    per_row = _threads_per_row(lin.input_dims // 32)
    if pairs > 20:
        per_row = min(per_row, 4)
    width = per_row * lin.output_dims
    tg = next((t for t in (256, 128, 64, 32) if width % t == 0), None)
    if tg is None or lin.input_dims * 16 > 32768:  # (input / 2) * 16 float16 sums
        return None
    key = ("lookup_glu", lin.output_dims, lin.input_dims, per_row, tg)
    if key not in _kernels:
        _kernels[key] = mx.fast.metal_kernel(
            name="q2_lookup_glu", input_names=["blocks", "idx", "gate", "up"], output_names=["out"],
            source=_DOWN_GLU, header=_header("Q2_0", lin.output_dims, lin.input_dims, P=per_row, SHARE=1)
            + f"constant uint TG = {tg};\n")
    (out,) = _kernels[key](inputs=[lin.blocks, idx, gate, up], template=[("OutT", gate.dtype)],
                           grid=(width, pairs, 1), threadgroup=(tg, 1, 1),
                           output_shapes=[(pairs, lin.output_dims)], output_dtypes=[gate.dtype])
    return out


def _lookup_glu(x: mx.array, indices: mx.array, up, gate, down) -> mx.array:
    k = indices.shape[-1]
    idx = indices.reshape(-1).astype(mx.uint32)
    table = lookup_table(x.reshape(-1, x.shape[-1]))  # one for gate and up: they read the same input
    if GATE_UP_ONE_KERNEL and (gate.output_dims, gate.input_dims) == (up.output_dims, up.input_dims):
        g, u = lookup_gate_up(gate, up, idx, table, k)
    else:
        g, u = lookup_matvec(gate, idx, table, k), lookup_matvec(up, idx, table, k)
    out = lookup_down_glu(down, idx, g, u) if GLU_IN_DOWN else None
    if out is None:
        out = lookup_matvec(down, idx, lookup_table_glu(g, u), 1)
    return out.reshape(*indices.shape, down.output_dims)


# Experts of other types than Q2_0 in a window: the pairs are not sorted, and for a few pairs down takes
# silu(gate) * up itself. On an M5 Ultra with IQ3_S, side by side in one process: unsorted, a pass of 7 / 8 / 10
# tokens 1.50 / 1.63 / 1.51 ms shorter (the sort is for MLX's gather products; this kernel takes each pair alone);
# swiglu in down 0.24 ms shorter at one token, 0.2-0.4 longer from five to eight.
BLOCK_GLU = True
MATVEC_GLU_UPTO = 30  # pairs


def matvec_glu(lin, idx: mx.array, gate: mx.array, up: mx.array) -> mx.array | None:
    """`lin(swiglu(gate, up), idx)` for a BlockSwitchLinear, one pair to each row of gate and up, (pairs, out); None
    where the inner activations do not fit a threadgroup's memory or its rows do not divide into threadgroups."""
    pairs, width = idx.shape[0], lin.per_row * lin.output_dims
    tg = next((t for t in (256, 128, 64, 32) if width % t == 0), None)
    if tg is None or lin.input_dims * gate.dtype.size > 16384:
        return None
    key = ("matvec_glu", lin.kind, lin.output_dims, lin.input_dims, lin.per_row, tg)
    if key not in _kernels:
        _kernels[key] = mx.fast.metal_kernel(
            name=f"matvec_glu_{lin.kind.lower()}", input_names=["blocks", "idx", "gate", "up"], output_names=["out"],
            source=_MATVEC_GLU, header=_header(lin.kind, lin.output_dims, lin.input_dims, P=lin.per_row, SHARE=1)
            + f"constant uint TG = {tg};\n")
    (out,) = _kernels[key](inputs=[lin.blocks, idx, gate, up], template=[("OutT", gate.dtype)],
                           grid=(width, pairs, 1), threadgroup=(tg, 1, 1),
                           output_shapes=[(pairs, lin.output_dims)], output_dtypes=[gate.dtype])
    return out


def switch_glu(x: mx.array, indices: mx.array, up, gate, down) -> mx.array:
    """down(silu(gate(x)) * up(x)) through each token's experts: x (..., in), indices (..., experts per token)."""
    if (indices.size >= TILES_FROM and x.dtype in (mx.float16, mx.float32)
            and all(tileable(lin.kind, lin.output_dims, lin.input_dims) for lin in (up, gate, down))):
        return _tiled_glu(x, indices, up, gate, down)
    if indices.size >= LOOKUP_FROM and all(lookable(lin, x.dtype) for lin in (up, gate, down)):
        return _lookup_glu(x, indices, up, gate, down)
    if (BLOCK_GLU and x.dtype == mx.float16 and all(isinstance(lin, BlockSwitchLinear) for lin in (up, gate, down))
            and indices.size < min(lin.unpack_from for lin in (up, gate, down))):
        # each pair on its own: sorting them, as below for MLX's gather products, changes nothing here
        x5 = mx.expand_dims(x, (-2, -3))
        g, u = gate(x5, indices), up(x5, indices)
        flat = indices.reshape(-1).astype(mx.uint32)
        out = (matvec_glu(down, flat, g.reshape(flat.size, -1), u.reshape(flat.size, -1))
               if flat.size <= MATVEC_GLU_UPTO else None)
        if out is not None:
            return out.reshape(*indices.shape, down.output_dims)
        return down(swiglu(g, u), indices).squeeze(-2)
    # as mlx-lm's SwitchGLU
    x = mx.expand_dims(x, (-2, -3))
    do_sort = indices.size >= 64
    shape, inv_order = indices.shape, None
    if do_sort:
        x, indices, inv_order = _gather_sort(x, indices)
    x = down(swiglu(gate(x, indices, sorted_indices=do_sort), up(x, indices, sorted_indices=do_sort)), indices,
             sorted_indices=do_sort)
    if do_sort:
        x = _scatter_unsort(x, inv_order, shape)
    return x.squeeze(-2)


class BlockSwitchGLU:
    """A layer's routed experts, called like mlx-lm's `SwitchGLU`."""

    def __init__(self, gate: BlockSwitchLinear, up: BlockSwitchLinear, down: BlockSwitchLinear):
        self.gate_proj, self.up_proj, self.down_proj = gate, up, down

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        return switch_glu(x, indices, self.up_proj, self.gate_proj, self.down_proj)
