"""A Q6_K layer held as codes and scales must multiply as the weights gguf-py's dequantizer gives."""

import gguf
import mlx.core as mx
import numpy as np
import pytest

from strata_mlx.dense import MAX_TOKENS, ScaledLinear
from strata_mlx.gguf import Tensor, q6k_parts, to_float
from strata_mlx.load import linear

ROWS, COLS = 12, 1024


def q6k(seed: int = 0) -> Tensor:
    rng = np.random.default_rng(seed)
    n = ROWS * COLS // 256
    blocks = rng.integers(0, 256, (n, 210), dtype=np.uint8)
    blocks[:, 208:210] = rng.uniform(-0.01, 0.01, (n, 1)).astype("<f2").view(np.uint8)
    return Tensor("q6k", (ROWS, COLS), "Q6_K", blocks.reshape(-1))


def reference(t: Tensor) -> np.ndarray:
    return gguf.quants.dequantize(t.data.reshape(ROWS, -1), gguf.GGMLQuantizationType.Q6_K).reshape(t.shape)


def test_the_parts_are_the_weights():
    t = q6k()
    codes, scales, d = q6k_parts(t)
    assert codes.dtype == np.int8 and scales.dtype == np.int8 and d.dtype == np.float16
    assert codes.min() >= -32 and codes.max() <= 31
    w = codes.astype(np.float64) * np.repeat(scales, 16, 1) * np.repeat(d.astype(np.float64), 256, 1)
    want = reference(t)
    assert np.abs(w - want).max() <= 1e-6 * np.abs(want).max()
    assert np.abs(to_float(t) - want).max() <= 1e-6 * np.abs(want).max()


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 5, 7, 8, MAX_TOKENS + 4])
def test_a_window_multiplies_as_the_weights_do(tokens, dtype):
    t = q6k(tokens)
    lin = linear(t)
    assert isinstance(lin, ScaledLinear)
    x = mx.random.normal((1, tokens, COLS)).astype(dtype)
    got = np.array(lin(x).astype(mx.float32))
    want = np.array(x.astype(mx.float32)) @ reference(t).T
    assert got.shape == (1, tokens, ROWS)
    # float16: the output is rounded to it, and a chunk of a prompt multiplies weights rounded to it
    assert np.abs(got - want).max() <= (2e-5 if dtype == mx.float32 else 4e-3) * np.abs(want).max()


@pytest.mark.parametrize("tokens", [1, 4, MAX_TOKENS + 4])
def test_reordered_outputs_and_inputs(tokens):
    t = q6k(7)
    rng = np.random.default_rng(1)
    rows = rng.permutation(ROWS)
    cols = (rng.permutation(COLS // 128)[:, None] * 128 + np.arange(128)).reshape(-1)  # heads of 128 inputs change places
    lin = linear(t, rows=rows, cols=cols)
    assert lin.d_step == 128
    x = mx.random.normal((tokens, COLS))
    want = np.array(x) @ reference(t)[rows][:, cols].T
    got = np.array(lin(x))
    assert np.abs(got - want).max() <= 2e-5 * np.abs(want).max()
    with pytest.raises(ValueError, match="splits a run"):
        linear(t, cols=rng.permutation(COLS))


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
def test_the_weights_as_floats_are_the_weights(dtype):
    t = q6k(3)
    rng = np.random.default_rng(2)
    cols = (rng.permutation(COLS // 128)[:, None] * 128 + np.arange(128)).reshape(-1)
    for lin, want in ((linear(t), reference(t)), (linear(t, cols=cols), reference(t)[:, cols])):
        got = np.array(lin.floats(dtype).astype(mx.float32))
        if dtype == mx.float32:
            assert np.abs(got - want).max() <= 1e-6 * np.abs(want).max()
        else:
            np.testing.assert_array_equal(got, want.astype(np.float16).astype(np.float32))  # rounded once


def test_floats_made_for_a_prompt_are_kept():
    from strata_mlx import dense

    lin = linear(q6k(5))
    x = mx.random.normal((MAX_TOKENS + 2, COLS))
    first = lin(x)
    held = lin._floats
    assert held is not None and lin in dense._holding
    assert lin(x) is not None and lin._floats is held  # the second piece of the prompt: not made again
    lin(x[:2])  # a window: the kernel, which needs no floats
    dense.release()
    assert lin._floats is None and not dense._holding
    np.testing.assert_array_equal(np.array(lin(x)), np.array(first))


def test_floats_not_kept_are_made_for_each_piece_and_let_go():
    from strata_mlx import dense

    kept, made = linear(q6k(5)), linear(q6k(5))
    made.keep = False  # a machine without the memory for them (load.py, #3)
    x = mx.random.normal((MAX_TOKENS + 2, COLS))
    np.testing.assert_array_equal(np.array(made(x)), np.array(kept(x)))
    assert made._floats is None and made not in dense._holding
    dense.release()


def test_a_shape_the_kernel_does_not_take_stays_floats():
    blocks = np.zeros((6 * 2, 210), dtype=np.uint8)
    lin = linear(Tensor("odd", (6, 512), "Q6_K", blocks.reshape(-1)))  # six rows: not groups of four
    assert not isinstance(lin, ScaledLinear) and lin.weight.shape == (6, 512)


# ---- layers kept as the file's blocks (BlockLinear, BlockEmbedding)

from strata_mlx.dense import BlockEmbedding, BlockLinear, as_blocks, held_bytes  # noqa: E402
from strata_mlx.load import AS_BLOCKS, embedding  # noqa: E402
from test_gguf import random_tensor  # noqa: E402
from test_gguf import reference as weights  # noqa: E402


@pytest.mark.parametrize("kind", AS_BLOCKS)
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 8, MAX_TOKENS + 1, 40])
def test_a_layer_of_blocks_multiplies_as_its_weights_do(kind, tokens, dtype):
    t = random_tensor(kind, seed=tokens)
    lin = linear(t, as_blocks=AS_BLOCKS)
    assert isinstance(lin, BlockLinear) and lin.nbytes == held_bytes(t)
    x = mx.random.normal((1, tokens, t.shape[1])).astype(dtype)
    got = np.array(lin(x).astype(mx.float32))
    want = np.array(x.astype(mx.float32)).astype(np.float64) @ weights(t).astype(np.float64).T
    assert got.shape == (1, tokens, t.shape[0]) and lin(x).dtype == dtype
    # float16: the output is rounded to it, and a chunk of a prompt multiplies weights rounded to it
    assert np.abs(got - want).max() <= (2e-5 if dtype == mx.float32 else 4e-3) * np.abs(want).max()


@pytest.mark.parametrize("kind", AS_BLOCKS)
@pytest.mark.parametrize("tokens", [1, 4, MAX_TOKENS + 4])
def test_a_layer_of_blocks_with_outputs_and_inputs_reordered(kind, tokens):
    t = random_tensor(kind, seed=7)
    rng = np.random.default_rng(1)
    rows = rng.permutation(t.shape[0])
    cols = (rng.permutation(t.shape[1] // 128)[:, None] * 128 + np.arange(128)).reshape(-1)  # heads of 128 inputs change places
    lin = linear(t, rows=rows, cols=cols, as_blocks=AS_BLOCKS)
    x = mx.random.normal((tokens, t.shape[1]))
    want = np.array(x).astype(np.float64) @ weights(t).astype(np.float64)[rows][:, cols].T
    assert np.abs(np.array(lin(x)) - want).max() <= 2e-5 * np.abs(want).max()


@pytest.mark.parametrize("kind", AS_BLOCKS)
def test_a_window_over_blocks_gives_each_token_what_it_gets_alone(kind):
    t = random_tensor(kind, seed=3)
    lin = linear(t, as_blocks=AS_BLOCKS)
    x = mx.random.normal((5, t.shape[1])).astype(mx.float16)
    together = np.array(lin(x))
    for i in range(5):
        assert (np.array(lin(x[i : i + 1])) == together[i : i + 1]).all()


def test_a_type_not_asked_for_as_blocks_is_held_as_before():
    t = random_tensor("Q3_K")
    assert not isinstance(linear(t), BlockLinear)
    assert not isinstance(linear(t, as_blocks=("IQ4_XS",)), BlockLinear)


def test_rows_that_do_not_end_on_a_block_are_held_as_before():
    odd = Tensor("odd", (2, 4, 320), "Q3_K", np.zeros(2 * 4 * 320 // 256 * 110, dtype=np.uint8))  # not a layer
    short = Tensor("short", (4, 320), "Q3_K", np.zeros(4 * 320 // 256 * 110, dtype=np.uint8))  # 320 is not 256s
    assert not as_blocks(odd) and not as_blocks(short) and as_blocks(random_tensor("Q3_K"))
    assert not as_blocks(random_tensor("Q4_K"))


@pytest.mark.parametrize("kind", AS_BLOCKS)
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
def test_an_embedding_of_blocks_gives_the_rows_of_the_weights(kind, dtype):
    t = random_tensor(kind, seed=2)
    emb = embedding(t, dtype, AS_BLOCKS)
    assert isinstance(emb, BlockEmbedding) and emb.nbytes == t.data.nbytes
    ids = mx.array([[5, 0, 3, 3, 1]])
    got = emb(ids)
    want = weights(t)[np.array(ids)]
    assert got.shape == (1, 5, t.shape[1]) and got.dtype == dtype
    assert np.abs(np.array(got.astype(mx.float32)) - want).max() <= (1e-6 if dtype == mx.float32 else 1e-3) * np.abs(want).max()
    one = emb(mx.array([4]))
    assert one.shape == (1, t.shape[1])
    assert (np.array(one) == np.array(embedding(t, dtype, AS_BLOCKS)(mx.array([[4]])))[0]).all()
