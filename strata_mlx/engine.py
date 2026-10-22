"""Decoding with draft-and-verify, and a conversation that is read only once.

A step pushes a window of tokens through the trunk at once: the last token that is certain, then a few guesses
for the ones after it. The trunk's own choice after each row says whether the next guess was right; the run of
right guesses is kept, and the state is set to exactly what decoding those tokens one by one would have left.

What keeping a prefix of n of the window's T tokens takes, per kind of state:

    attention K/V and indexer keys   appended by the pass; drop the last T - n
    delta-net convolution history    a slice of the pass's convolution input
    delta-net recurrent state        the recurrence run again over the first n tokens, from inputs the pass kept
    n-gram convolution history       a slice, rebuilt from the state before and after the pass

The recurrent state cannot be cut back, so going back in a conversation needs a checkpoint: a copy of the small
states (113 MB) at a position. The attention caches are cut to that position, and reading resumes from there.
Checkpoints are taken while a prompt is read: before its last message boundaries and before its last token, which
is where a follow-up request usually parts from this one.
"""

import errno
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen4_exp
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.gated_delta import gated_delta_update

from . import bylayer
from .attention import AttnCache, install
from .conversations import reach, victim
from .drafting import FIRST_GUESSES, MAX_WINDOW, Lookup, Policy
from .fused import FusedResidual, PlainResidual, gated_norm, gdn_prep, install_routers
from .mtp import Mtp
from .rows import GuessRows
from .sampler import LIST, Sampling, history_counts, penalize, pick, uniform

TURN_TOKEN = 248045  # <|im_start|>
PERIODIC = 16384
ROOT = 2048  # a prompt read from its start is checkpointed where its first message ends, if that is this far in
SESSION_FORMAT = "strata-mlx session 1"


def _arrays(tree):
    if isinstance(tree, mx.array):
        yield tree
    elif isinstance(tree, (list, tuple)):
        for t in tree:
            yield from _arrays(t)


class _Slot:
    """The one slot of the port's layer cache that its n-gram layer reads and writes."""

    lengths = None

    def __init__(self, value):
        self.value = value

    def __getitem__(self, _):
        return self.value

    def __setitem__(self, _, value):
        self.value = value


@dataclass
class _Window:
    ids: mx.array
    residual: mx.array  # (1, T, hc * d): what the draft head reads
    gdn: list  # per delta-net layer: (convolution input, q, k, v, a, b, recurrent state after all T)
    ple_conv: mx.array | None


@dataclass
class _Checkpoint:
    at: int  # tokens decoded
    gdn: dict
    ple_conv: mx.array | None
    context: mx.array
    used: int = 0  # when a request last made it, landed on it or went on through it (conversations.victim)
    pinned: bool = False  # the end of a prefix a request named (`pin`): never dropped while it is the one named
    tail: bool = False  # the one before a prompt's last token: of use to that request asked again, the first to go


def _cut(cache, at: int):
    """A cache of the first `at` cells of this one. Its arrays are the other's until it is added to."""
    new = type(cache)()
    if at and cache.keys is not None:
        keys, values, index = cache.state
        new.state = keys[..., :at, :], values[..., :at, :], index[:, :at]
    return new


@dataclass
class Conversation:
    """A conversation's whole state, out of the engine (conversations.py)."""

    tokens: list
    gdn: dict
    ple_conv: mx.array | None
    context: mx.array
    attn: dict
    mtp_cache: object
    checkpoints: list
    image_key: tuple | None = None

    @property
    def pinned(self) -> bool:
        return any(c.pinned for c in self.checkpoints)

    @property
    def bytes(self) -> int:
        caches = [*self.attn.values(), *([self.mtp_cache] if self.mtp_cache is not None else [])]
        kv = sum(a.nbytes for c in caches for a in (c.keys, c.values, c.indexer.keys) if a is not None)
        small = sum(a.nbytes for pair in self.gdn.values() for a in pair)
        return kv + small * (1 + len(self.checkpoints)) + 4 * len(self.tokens)

    def prefix(self, cp: _Checkpoint) -> "Conversation":
        """The conversation as it was at one of its checkpoints, this one left as it is."""
        return Conversation(self.tokens[: cp.at], dict(cp.gdn), cp.ple_conv, cp.context,
                            {i: _cut(c, cp.at) for i, c in self.attn.items()},
                            _cut(self.mtp_cache, cp.at) if self.mtp_cache is not None else None,
                            [c for c in self.checkpoints if c.at <= cp.at], self.image_key)


@dataclass
class Stats:
    steps: int = 0
    tokens: int = 0
    drafted: int = 0
    accepted: int = 0
    lookup_steps: int = 0
    by_position: list = field(default_factory=lambda: [0] * MAX_WINDOW)


# Windows up to this many tokens take the fused hyper-connection kernels, longer ones MLX's operations. Measured in
# one process with the passes shuffled (e45), the best differs by GPU: on this M4 Max (generation 16) windows of 7
# and 8 tokens took 69.2 and 72.1 ms fused against 64.2 and 66.5 with MLX's operations; on an M5 Ultra (generation
# 17) fused was 2.0 ms a pass faster at 7 and 8, with Q2_0 and IQ3_S alike (pull request 9's author, against a
# control engine). Up to 6 tokens either way runs the same. Fused up to 6 also keeps 7- and 8-token windows' logits
# what they were before pull request 9; fused up to 8 they agree with a token at a time better.
def gpu_generation(architecture: str | None = None) -> int:
    """The Apple GPU's generation from MLX's name for it ("applegpu_g16s": 16, an M4; an M5 is 17), 0 if unknown."""
    if architecture is None:
        try:
            architecture = mx.device_info().get("architecture", "")
        except Exception:
            return 0
    m = re.match(r"applegpu_g(\d+)", architecture or "")
    return int(m.group(1)) if m else 0


def fuse_upto_for(generation: int) -> int:
    return MAX_WINDOW if generation >= 17 else 6


FUSE_UPTO = fuse_upto_for(gpu_generation())

# A window's pass is handed to the GPU in parts of this many layers while the rest is still being built (0: all
# at once, when it is waited for). MLX builds a graph without running it, so until then the GPU sits idle for the
# 1.8-2.1 ms the CPU takes to build a pass (e55); the graph and so the bits are the same either way. Every 6
# layers: a pass of 1 / 4 tokens 2.4-2.9 / 1.8-2.7 ms shorter on the M4 Max (e57: single passes shuffled across
# engines sharing the weights, lower quartile; every 3, 4, 8 within 0.5 ms of it). Not where experts are read
# from the SSD: such a pass already stops at every layer.
SUBMIT_EVERY = 6


class Engine:
    # What a pass leaves behind (caches, recurrent states) is computed without being waited for: the next thing
    # that needs it waits, and until then the CPU goes on with building that next thing.
    settle = staticmethod(mx.async_eval)

    def __init__(self, model: qwen4_exp.Model, mtp: Mtp | None = None, drafts: int = MAX_WINDOW - 1,
                 lookup: bool = False, chunk: int = 2048, checkpoints: int = 8, turn_token: int = TURN_TOKEN,
                 fused: bool = True, fuse_upto: int = FUSE_UPTO, as_sure: bool = True, guess_rows: bool = True,
                 root: int = ROOT, piece: int = bylayer.PIECE, ahead: bool = True, submit_every: int | None = None):
        """`drafts`: the most guesses the head makes for a pass. With `as_sure` it makes as many of them as it
        is sure enough of (drafting.py); without, always that many. With `guess_rows` a guess is worked out
        over the rows of the output layer that have been in sight, not over all of them (rows.py). Where
        experts are read from the SSD, a prompt of more than a chunk is read a layer at a time in pieces of
        `piece` tokens (bylayer.py; 0 for never), with `ahead` the next layer's experts read during this one."""
        self.model, self.m, self.mtp = model, model.model, mtp
        install(self.m.layers)
        install_routers(self.m.layers)
        if mtp is not None:
            install_routers(mtp.layers)
        self.drafts = min(drafts, MAX_WINDOW - 1) if mtp is not None else 0
        self.as_sure = as_sure
        vocab = self.m.args.vocab_size
        # kept through `reset`: what one conversation listed serves the next
        self.rows = GuessRows(model.lm_head, vocab) if self.drafts and guess_rows and vocab >= 100_000 else None
        self.chunk, self.max_checkpoints, self.turn_token, self.root = chunk, checkpoints, turn_token, root
        self.piece, self.ahead = piece, ahead
        self.clock = 0  # counts the uses of checkpoints
        self.one_shot = False  # the conversation held is of a request that said no other continues it
        self.vocab = self.m.args.vocab_size
        self.eos = self.m.args.eos_token_id
        self.gdn_layers = [i for i, l in enumerate(self.m.layers) if l.layer_type == "linear_attention"]
        self.lookup = Lookup() if lookup else None
        self.paged = getattr(model, "expert_store", None) is not None
        self.fuse_upto = fuse_upto  # windows up to this many tokens use the fused hyper-connection kernels
        self.submit_every = SUBMIT_EVERY if submit_every is None else submit_every
        self.fused = self._fuse() if fused else None
        self.policy = Policy()
        self.timed_sizes = set()
        self.sampling = Sampling()
        self.identity = ""  # names the model files; a session saved under another name is not restored
        self.reset()

    def reset(self):
        self.attn = {i: AttnCache() for i, l in enumerate(self.m.layers) if l.layer_type != "linear_attention"}
        self.gdn = {i: self._gdn_start(self.m.layers[i].linear_attn) for i in self.gdn_layers}
        self.ple_conv = None
        self.context = mx.full((1, self.m.args.ngram_size - 1), self.eos, dtype=mx.int32)
        self.mtp_cache = self.mtp.make_cache() if self.mtp is not None else None
        self.image_key = None
        self.vision = None
        self.tokens = []  # decoded into the state
        self.next_token = None  # decided, not decoded yet
        self.guesses = []
        self.guess_sure = ()  # how sure the head was of each run of them
        self.checkpoints = []
        self.stats = Stats()
        if self.lookup:
            self.lookup.reset()

    # ---- the trunk

    def _gdn_start(self, la):
        """(convolution history, recurrent state) of a delta-net layer that has read nothing."""
        dtype = la.conv1d.weight.dtype
        return (mx.zeros((1, la.conv_kernel_size - 1, la.conv_dim), dtype=dtype),
                mx.zeros((1, la.n_v, la.dv, la.dk), dtype=mx.float32))

    def _gdn(self, la, x, state):
        conv_state, recurrent = state
        B, S, _ = x.shape
        mixed = la.in_proj_qkv(x)
        z = la.in_proj_z(x).reshape(B, S, la.n_v, la.dv)
        b, a = la.in_proj_b(x), la.in_proj_a(x)
        prepared = gdn_prep(conv_state, mixed, la.conv1d.weight, la.key_dim, la.dk) if B == 1 else None
        if prepared is not None:  # the same as below in one kernel (fused.py)
            conv_input, q, k, v = prepared
            q = q.reshape(B, S, la.n_k, la.dk)
            k = k.reshape(B, S, la.n_k, la.dk)
            v = v.reshape(B, S, la.n_v, la.dv)
        else:
            conv_input = mx.concatenate([conv_state, mixed], axis=1)
            q, k, v = mx.split(nn.silu(la.conv1d(conv_input)), [la.key_dim, 2 * la.key_dim], axis=-1)
            q = q.reshape(B, S, la.n_k, la.dk)
            k = k.reshape(B, S, la.n_k, la.dk)
            v = v.reshape(B, S, la.n_v, la.dv)
            inv_scale = la.dk**-0.5
            q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        out, after = gated_delta_update(q, k, v, a, b, la.A_log, la.dt_bias, recurrent, None)
        normed = gated_norm(la.norm, out, z)  # one kernel for the two below (fused.py)
        y = la.out_proj((la.norm(out, z) if normed is None else normed).reshape(B, S, -1))
        return y, (conv_input, q, k, v, a, b, after)

    # A pass is thousands of small kernels. Two things make them fewer. The hyper-connections are three kernels
    # each (fused.py), with the write of the block before folded in: the residual travels as (before, out,
    # inject) and is only put together inside those kernels. And the parts of a layer that are plain functions
    # of arrays are compiled, which merges runs of elementwise operations into one kernel each. Attention is not
    # compiled: its caches grow with every token, and a compiled function is tied to the shapes it was traced
    # with.

    @staticmethod
    def _write(before, out, inject):
        return before + (out[..., None, :] * inject[..., None]).reshape(*out.shape[:-1], -1)

    def _hc(self, key, tokens: int):
        return self.hc[key][0 if tokens <= self.fuse_upto else 1]

    def _mix_block(self, i, before, out, inject, conv_state, recurrent):
        """A delta-net layer's first half -> (residual, the mixer's output, its inject, what _commit needs)."""
        layer = self.m.layers[i]
        x, residual, inject = self._hc((i, 0), before.shape[1])(before, out, inject)
        x, record = self._gdn(layer.linear_attn, x, (conv_state, recurrent))
        return (residual, x, inject, *record)

    def _ffn_block(self, i, before, out, inject):
        """Any layer's second half -> (residual, the experts' output, its inject)."""
        x, residual, inject = self._hc((i, 1), before.shape[1])(before, out, inject)
        return residual, self.m.layers[i].mlp(x), inject

    def _fuse(self):
        m = self.m
        parts = {(i, 0): l.attn_hyper_connection for i, l in enumerate(m.layers)}
        parts.update({(i, 1): l.mlp_hyper_connection for i, l in enumerate(m.layers)})
        parts["out"] = m.hyper_connection_mixer
        self.hc = {key: (FusedResidual(gr), PlainResidual(gr)) for key, gr in parts.items()}
        # experts read from the SSD (paged.py) stop at every layer for the router's choice: not a function to compile
        paged = self.paged
        return {i: (mx.compile(lambda *a, i=i: self._mix_block(i, *a)) if i in self.gdn_layers else None,
                    (lambda *a, i=i: self._ffn_block(i, *a)) if paged else mx.compile(lambda *a, i=i: self._ffn_block(i, *a)))
                for i in range(len(m.layers))}

    @property
    def rope(self):
        if self.vision is None:
            return self.m.rope
        from .vision import ImageRoPE
        return ImageRoPE(self.m.rope, self.vision)

    def embed(self, ids, start):
        return (self.m.embed_tokens(ids) if self.vision is None
                else self.vision.embed(self.m.embed_tokens, ids, start))

    def _forward(self, ids: mx.array, last_only: bool = False):
        """ids (1, T) -> (logits, the window). Attention caches grow by T; nothing else changes.

        Logits are (1, T, vocab), or (1, 1, vocab) for the last token when the others are not needed: a chunk
        of a prompt, which is always kept whole.
        """
        return self._forward_plain(ids, last_only) if self.fused is None else self._forward_fused(ids, last_only)

    def _record(self, i: int, record: tuple, whole: bool) -> tuple:
        """What a delta-net layer leaves for the commit. Kept whole, a window needs none of what cutting it back
        takes: the q, k and v of every token, 36 layers of them, were 3 GB for 1,500 tokens of prompt, held
        through the pass."""
        if not whole:
            return tuple(record)
        taps = self.m.layers[i].linear_attn.conv_kernel_size - 1
        tail = self._rows(record[0], record[0].shape[1] - taps, taps)
        if self.paged:  # the pass stops at every layer anyway: let go of the rest now, not at the end
            mx.eval(tail, record[6])
        return (tail, None, None, None, None, None, record[6])

    @staticmethod
    def _rows(x: mx.array, start: int, count: int) -> mx.array:
        """x[:, start : start + count] in memory of its own. A slice shares its source's memory and keeps all of
        it: the few rows of convolution history kept from a 1,500-token chunk held 31 MB a layer, 1.1 GB in all,
        for as long as the conversation stood there."""
        return mx.take(x, mx.arange(start, start + count), axis=1)

    def _forward_fused(self, ids: mx.array, last_only: bool = False):
        m, T = self.m, ids.shape[1]
        before = mx.tile(self.embed(ids, len(self.tokens)), (1, 1, m.hc))
        mask = create_attention_mask(before, next(iter(self.attn.values())))
        nothing = (mx.zeros((1, T, m.args.hidden_size), dtype=before.dtype), mx.zeros((1, T, m.hc), dtype=before.dtype))
        out, inject = nothing
        records, ple_conv = [], None
        # a compiled function is traced once for each shape it meets: for windows, not for a prompt's odd lengths
        compiled = T <= MAX_WINDOW
        for i, layer in enumerate(m.layers):
            if layer.ple is not None:
                slot = _Slot(self.ple_conv)
                h = self._write(before, out, inject)
                before, (out, inject) = h + layer.ple(h, ids, self.context, slot), nothing
                ple_conv = slot.value
            first, second = self.fused[i]
            if i in self.gdn:
                before, out, inject, *record = (first(before, out, inject, *self.gdn[i]) if compiled
                                                else self._mix_block(i, before, out, inject, *self.gdn[i]))
                records.append(self._record(i, record, last_only))
            else:
                x, before, inject = self._hc((i, 0), T)(before, out, inject)
                out = layer.self_attn(x, self.rope, mask, self.attn[i], self.attn[i].indexer)
            before, out, inject = (second(before, out, inject) if compiled
                                   else self._ffn_block(i, before, out, inject))
            if compiled and self.submit_every and not self.paged and (i + 1) % self.submit_every == 0 \
                    and i + 1 < len(m.layers):
                mx.async_eval(before, out, inject)
        x, residual, _ = self._hc("out", T)(before, out, inject)
        logits = self.model.lm_head(x[:, -1:] if last_only else x)
        return logits, _Window(ids, residual, records, ple_conv)

    def _forward_plain(self, ids: mx.array, last_only: bool = False):
        """The same pass in the model's own operations: the reference the fused one is tested against."""
        m = self.m
        h = mx.tile(self.embed(ids, len(self.tokens)), (1, 1, m.hc))
        mask = create_attention_mask(h, next(iter(self.attn.values())))
        records, ple_conv = [], None
        for i, layer in enumerate(m.layers):
            if layer.ple is not None:
                slot = _Slot(self.ple_conv)
                h = h + layer.ple(h, ids, self.context, slot)
                ple_conv = slot.value
            x, hyper, inject = layer.attn_hyper_connection(h)
            if i in self.gdn:
                x, record = self._gdn(layer.linear_attn, x, self.gdn[i])
                records.append(self._record(i, record, last_only))
            else:
                x = layer.self_attn(x, self.rope, mask, self.attn[i], self.attn[i].indexer)
            h = self._write(hyper, x, inject)
            x, hyper, inject = layer.mlp_hyper_connection(h)
            h = self._write(hyper, layer.mlp(x), inject)
        logits = self.model.lm_head(m.hyper_connection_mixer(h[:, -1:] if last_only else h))
        return logits, _Window(ids, h, records, ple_conv)

    def _commit(self, w: _Window, n: int):
        """Keep the first n tokens of the window."""
        T = w.ids.shape[1]
        for cache in self.attn.values():
            cache.trim(T - n)
        for i, (conv_input, q, k, v, a, b, after) in zip(self.gdn_layers, w.gdn):
            la = self.m.layers[i].linear_attn
            taps = la.conv_kernel_size - 1
            if n == T:
                recurrent = after
            elif q is None:
                raise ValueError("a chunk of a prompt can only be kept whole")
            else:
                recurrent = gated_delta_update(q[:, :n], k[:, :n], v[:, :n], a[:, :n], b[:, :n],
                                               la.A_log, la.dt_bias, self.gdn[i][1], None)[1]
            self.gdn[i] = (conv_input if q is None else self._rows(conv_input, n, taps), recurrent)
        if w.ple_conv is not None:
            taps = w.ple_conv.shape[1]
            if n == T:
                self.ple_conv = w.ple_conv
            elif T <= taps:
                before = self.ple_conv if self.ple_conv is not None else mx.zeros_like(w.ple_conv)
                self.ple_conv = mx.concatenate([before[:, n:], w.ple_conv[:, taps - T : taps - T + n]], axis=1)
            else:
                raise ValueError(f"a window of {T} tokens can only be kept whole")
        # the tokens are known here: worked out on the GPU, the next pass's n-gram rows would have to wait for them
        ids = w.ids.tolist()[0]
        ctx = self.context.shape[1]
        self.context = mx.array([(self.context.tolist()[0] + ids)[n : n + ctx]], dtype=self.context.dtype)
        self.settle(*_arrays([self.ple_conv, list(self.gdn.values()),
                          [(c.keys, c.values, c.indexer.keys) for c in self.attn.values()]]))
        kept = ids[:n]
        self.tokens += kept
        if self.lookup:
            self.lookup.extend(kept)

    def warm(self):
        """Run a pass of every window size once, so that no request waits for kernels to be compiled."""
        for T in range(1, MAX_WINDOW + 1):
            logits, w = self._forward(mx.zeros((1, T), dtype=mx.int32))
            mx.eval(logits, *_arrays([list(w.gdn), w.ple_conv, w.residual]))
            for cache in self.attn.values():
                cache.trim(T)
            self.timed_sizes.add(T)

    # ---- going back

    def _checkpoint(self, pinned: bool = False, tail: bool = False):
        self.clock += 1
        if pinned:
            for cp in self.checkpoints:
                cp.pinned = False  # one prefix is pinned at a time: a new one takes the place
        if self.checkpoints and self.checkpoints[-1].at == len(self.tokens):
            cp = self.checkpoints[-1]
            cp.used, cp.pinned, cp.tail = self.clock, cp.pinned or pinned, cp.tail and tail
            return
        self.checkpoints.append(_Checkpoint(len(self.tokens), dict(self.gdn), self.ple_conv, self.context,
                                            self.clock, pinned, tail))
        while len(self.checkpoints) > self.max_checkpoints:
            cps = self.checkpoints
            del cps[victim([c.used for c in cps], self.max_checkpoints, [c.tail for c in cps], [c.pinned for c in cps])]

    def _restore(self, cp: _Checkpoint):
        back = len(self.tokens) - cp.at
        for cache in self.attn.values():
            cache.trim(back)
        if self.mtp_cache is not None:
            self.mtp_cache.trim(back)
        self.gdn, self.ple_conv, self.context = dict(cp.gdn), cp.ple_conv, cp.context
        del self.tokens[cp.at :]
        self.checkpoints = [c for c in self.checkpoints if c.at <= cp.at]
        if self.lookup:
            self.lookup.truncate(cp.at)

    # ---- a conversation on disk

    def _small(self, prefix: str, gdn: dict, ple_conv, context) -> dict:
        """The states that cannot be cut back, as named arrays: what a checkpoint holds."""
        out = {f"{prefix}context": context}
        if ple_conv is not None:
            out[f"{prefix}ple_conv"] = ple_conv
        for i, (conv, recurrent) in gdn.items():
            out[f"{prefix}gdn.{i}.conv"], out[f"{prefix}gdn.{i}.state"] = conv, recurrent
        return out

    def save(self, path: str) -> int:
        """Write the conversation the state holds to `path`, replacing it only once it is whole. Returns its size."""
        if self.image_key is not None:
            raise ValueError("image conversations support RAM parking only, not SAVE")
        arrays = {"tokens": mx.array(self.tokens, dtype=mx.int32), **self._small("", self.gdn, self.ple_conv, self.context)}
        caches = {f"attn.{i}": c for i, c in self.attn.items()}
        if self.mtp_cache is not None:
            caches["mtp"] = self.mtp_cache
        for name, cache in caches.items():
            if cache.keys is not None:
                arrays[f"{name}.keys"], arrays[f"{name}.values"], arrays[f"{name}.index"] = cache.state
        for j, cp in enumerate(self.checkpoints):
            arrays.update(self._small(f"cp.{j}.", cp.gdn, cp.ple_conv, cp.context))
        meta = {"format": SESSION_FORMAT, "model": self.identity, "draft_head": str(int(self.mtp is not None)),
                "checkpoints": json.dumps([cp.at for cp in self.checkpoints])}
        tmp = path + ".tmp.safetensors"  # MLX adds the extension itself
        size = sum(a.nbytes for a in arrays.values())
        if shutil.disk_usage(os.path.dirname(os.path.abspath(path))).free < size + (64 << 20):
            raise OSError(errno.ENOSPC, f"a session of {size >> 20} MiB does not fit on the disk")
        try:
            mx.save_safetensors(path + ".tmp", arrays, metadata=meta)
        except RuntimeError as e:  # how MLX reports a file it could not write
            if os.path.exists(tmp):
                os.remove(tmp)
            raise OSError(errno.EIO, str(e)) from None
        os.replace(tmp, path)
        return os.path.getsize(path)

    def restore(self, path: str) -> int:
        """Take the conversation in `path` as the state. Returns its size; raises ValueError for a file that is
        not a session of this model, and leaves the state as it was."""
        size = os.path.getsize(path)
        try:
            arrays, meta = mx.load(path, format="safetensors", return_metadata=True)
        except (RuntimeError, ValueError) as e:
            raise ValueError(f"not a session file: {e}") from None
        if meta.get("format") != SESSION_FORMAT:
            raise ValueError("not a session file of this engine")
        if meta.get("model") != self.identity or meta.get("draft_head") != str(int(self.mtp is not None)):
            raise ValueError("the session was saved with other model files")
        try:
            tokens = arrays["tokens"].tolist()
            n = len(tokens)

            def small(prefix):
                gdn = {i: (arrays[f"{prefix}gdn.{i}.conv"], arrays[f"{prefix}gdn.{i}.state"]) for i in self.gdn_layers}
                for i, pair in gdn.items():
                    if [a.shape for a in pair] != [a.shape for a in self._gdn_start(self.m.layers[i].linear_attn)]:
                        raise ValueError("a state of another model's shape")
                return gdn, arrays.get(f"{prefix}ple_conv"), arrays[f"{prefix}context"]

            def cache(name, new):
                c = new()
                if n:
                    c.state = arrays[f"{name}.keys"], arrays[f"{name}.values"], arrays[f"{name}.index"]
                    if c.offset != n:
                        raise ValueError("a cache that does not match the tokens")
                return c

            gdn, ple_conv, context = small("")
            attn = {i: cache(f"attn.{i}", AttnCache) for i in self.attn}
            mtp_cache = cache("mtp", self.mtp.make_cache) if self.mtp is not None else None
            checkpoints = [_Checkpoint(at, *small(f"cp.{j}.")) for j, at in enumerate(json.loads(meta["checkpoints"]))]
            if any(not 0 <= cp.at <= n for cp in checkpoints):
                raise ValueError("a checkpoint past the end")
        except (KeyError, TypeError) as e:
            raise ValueError(f"an incomplete session file ({e})") from None
        mx.eval(*arrays.values())  # all of it now: the file may be written again before the state is used
        self.reset()
        self.tokens, self.gdn, self.ple_conv, self.context = tokens, gdn, ple_conv, context
        self.attn, self.mtp_cache, self.checkpoints = attn, mtp_cache, checkpoints
        if self.lookup:
            self.lookup.extend(tokens)
        return size

    def resume(self, prompt: list) -> int:
        """Bring the state to the longest prefix of `prompt` it can reach; returns how many tokens that is."""
        self.next_token, self.guesses, self.guess_sure = None, [], ()
        same, cp = reach(self.tokens, self.checkpoints, prompt)
        self.clock += 1
        for c in self.checkpoints:
            if c.at <= same:
                c.used = self.clock  # the request goes on through them
        if same == len(self.tokens):
            return same
        if cp is None:
            self.reset()
            return 0
        self._restore(cp)
        return cp.at

    def detach(self) -> Conversation:
        """The conversation held, taken out of the engine, which then holds none."""
        conv = Conversation(self.tokens, self.gdn, self.ple_conv, self.context, self.attn, self.mtp_cache,
                            self.checkpoints, self.image_key)
        self.reset()
        return conv

    def attach(self, conv: Conversation):
        """Hold a conversation that `detach` or `Conversation.prefix` gave, in place of the one held."""
        self.reset()
        self.tokens, self.gdn, self.ple_conv, self.context = conv.tokens, conv.gdn, conv.ple_conv, conv.context
        self.attn, self.mtp_cache, self.checkpoints = conv.attn, conv.mtp_cache, conv.checkpoints
        self.image_key = conv.image_key
        if self.lookup:
            self.lookup.extend(self.tokens)

    def _cuts(self, prompt: list, start: int, keep: bool = True, pin: int = 0) -> dict:
        """Where to checkpoint while reading prompt[start:], and as what (`_checkpoint`'s arguments): token counts
        at which a later prompt may part from this one. Where the last two messages begin, every PERIODIC
        tokens and before the last token; of a prompt read from its start, where its first message ends (the
        system prompt, which the client's next chat begins with); and where the request says a prefix ends
        (`pin`). A request that no other continues (`keep` false) leaves only the last two of those."""
        n = len(prompt)
        cuts = {}
        if keep:
            turns = [i for i in range(start + 1, n) if prompt[i] == self.turn_token][-2:]
            cuts = dict.fromkeys([*turns, *range((start // PERIODIC + 1) * PERIODIC, n, PERIODIC)], {})
            cuts.setdefault(n - 1, {"tail": True})
        if start == 0:
            first = next((i for i in range(1, n) if prompt[i] == self.turn_token), 0)
            if first >= self.root:
                cuts.setdefault(first, {})
        if start < pin < n:
            cuts[pin] = {"pinned": True}
        return {at: kind for at, kind in sorted(cuts.items()) if at not in (start, 0)}

    # ---- the draft head

    def _cells(self, residual: mx.array, following: mx.array):
        """Add the cells of positions just decoded: their residuals, and the token after each."""
        return self.mtp(residual, self.embed(following, self.mtp_cache.offset + 1), self.rope, self.mtp_cache)

    def _draft(self, residual: mx.array, following: mx.array) -> list:
        """The real cells of the positions just decoded, then guesses for the tokens after the pending one.

        The guesses' own cells are speculative and are dropped again; the real ones stay in the head's cache.
        When the request samples, each guess is drawn with the number the trunk will use for that token.
        """
        m, cache, s = self.m, self.mtp_cache, self.sampling
        first = len(self.tokens) + 1  # the index of the token the first guess stands for
        t0 = time.perf_counter()

        rows = self.rows if self.rows is not None and self.rows.ready and self.rows.count > LIST else None

        def guess(x, j):
            if rows is None:
                logits = self.model.lm_head(x)[0]
            else:
                logits, tokens = rows.logits(x[0])  # of the listed rows only
            at = pick(logits, mx.array([uniform(s.seed, first + j)]), s)[None]  # (1, 1): which of the logits
            g = at if rows is None else tokens[at]
            if not self.as_sure:
                return g, None
            logits = logits.astype(mx.float32)
            return g, mx.exp(mx.take_along_axis(logits, at.reshape(-1, 1), axis=-1) - mx.logsumexp(logits, axis=-1, keepdims=True))

        x, r = self._cells(residual, following)
        x, r = x[:, -1:], r[:, -1:]
        if not self.as_sure:
            guesses = []
            for j in range(self.drafts):
                if j:
                    x, r = self.mtp(r, self.embed(guesses[-1], cache.offset + 1), self.rope, cache)
                guesses.append(guess(x, j)[0])
            out = mx.concatenate(guesses, axis=1)
            mx.eval(out)
            cache.trim(self.drafts - 1)
            mx.eval(*_arrays([cache.keys, cache.values, cache.indexer.keys]))
            self.guess_sure = ()
            return out[0].tolist()
        # The first few guesses in any case, in one wait for the GPU; then two more at a time while the head is
        # sure enough of the run so far for them to be worth a longer pass, and of those the ones that are
        # (drafting.py).
        guesses, sure, so_far, made, g, policy = [], [], 1.0, 0, None, self.policy
        first = min(FIRST_GUESSES, self.drafts)
        while made < self.drafts and (made < first or policy.worth_guessing(so_far, len(guesses))):
            batch = []
            for _ in range(first if not made else min(2, self.drafts - made)):
                if made + len(batch):
                    x, r = self.mtp(r, self.embed(g, cache.offset + 1), self.rope, cache)
                g, p = guess(x, made + len(batch))
                batch.append((g, p))
            mx.eval(batch)
            refused = False
            for token, p in batch:
                made += 1
                so_far *= p.item()
                refused = refused or (made > first and not policy.worth_keeping(so_far, len(guesses)))
                if not refused:
                    guesses.append(token.item())
                    sure.append(so_far)
            if refused:
                break
        cache.trim(made - 1)  # the cells of the guesses themselves
        self.settle(*_arrays([cache.keys, cache.values, cache.indexer.keys]))
        self.guess_sure = tuple(sure)
        self.policy.observe_guesses(made, (time.perf_counter() - t0) * 1e3)
        return guesses

    # ---- decoding

    def _choose(self, logits: mx.array, window: list, history: list | None = None) -> list:
        """logits (T, vocab) -> the trunk's token after each row of the window, which follows `history`."""
        s = self.sampling
        history = self.tokens if history is None else history
        if s.penalties:
            logits = penalize(logits.astype(mx.float32),
                              history_counts(history, window, self.vocab, s.penalty_last_n), s)
        base = len(history) + 1  # row t decides the token at index base + t
        u = mx.array([uniform(s.seed, base + t) for t in range(len(window))])
        out = pick(logits, u, s)
        mx.eval(out)
        return out.tolist()

    def read(self, prompt: list, on_chunk=None, keep: bool = True, pin: int = 0, on_step=None) -> int:
        """Read prompt[len(self.tokens):] into the state; returns the first token of the answer. `keep` and `pin`
        say which checkpoints it leaves (`_cuts`). `on_chunk(tokens, of)` hears how far it is; `on_step()` is
        called more often where a prompt is read a layer at a time, and what either raises stops the read."""
        start, n = len(self.tokens), len(prompt)
        self.next_token, self.guesses, self.guess_sure = None, [], ()
        cuts = self._cuts(prompt, start, keep, pin)
        if pin == start and self.checkpoints and self.checkpoints[-1].at == start:
            self._checkpoint(pinned=True)  # the prefix was read by an earlier request: it is the one named now
        if self.paged and self.piece and self.fused is not None and n - start > self.chunk:
            return bylayer.read(self, prompt, cuts, on_chunk, on_step)
        a = start
        for b in sorted({*cuts, *range(start + self.chunk, n, self.chunk), n}):
            if b <= a:
                continue
            logits, w = self._forward(mx.array([prompt[a:b]]), last_only=True)
            if b == n:
                self.next_token = self._choose(logits[0], prompt[-1:], prompt[:-1])[0]
                if self.rows is not None:
                    self.rows.add(prompt)
                    self.rows.note(logits[0])
            else:
                mx.eval(logits)
            self._commit(w, b - a)
            if b == n:
                if self.drafts:
                    self.guesses = self._draft(w.residual, mx.array([prompt[a + 1 :] + [self.next_token]]))
            elif self.drafts:
                self._cells(w.residual, mx.array([prompt[a + 1 : b + 1]]))
                mx.eval(*_arrays([self.mtp_cache.keys, self.mtp_cache.values, self.mtp_cache.indexer.keys]))
            if b in cuts:
                self._checkpoint(**cuts[b])
            if on_chunk:
                on_chunk(b, n)
            a = b
        return self.next_token

    def prefill(self, prompt: list) -> int:
        """Read a prompt into an empty conversation; returns the first token of the answer."""
        return self.read(prompt)

    def step(self, limit: int = MAX_WINDOW, stop: tuple = ()) -> list:
        """One pass of the trunk. Returns the tokens it settled, the last of which is not decoded yet.

        At most `limit` tokens, and nothing after a token in `stop`: the state never runs ahead of the answer,
        so the next request, which continues from this answer, finds it as it left it.
        """
        t0 = time.perf_counter()
        guesses, from_lookup, match, sure = self.guesses, False, 0, self.guess_sure
        if self.lookup:
            found, match = self.lookup.propose([self.next_token], MAX_WINDOW - 1)
            if found and (not guesses or found[0] == guesses[0]):
                k = self.policy.choose(len(guesses) + 1, len(found), match)
                if k:
                    guesses, from_lookup = found[:k], True
        window = [self.next_token, *guesses][: max(1, limit)]
        T = len(window)
        logits, w = self._forward(mx.array([window]))
        choice = self._choose(logits[0], window)
        kept = 0
        while kept < T - 1 and window[kept + 1] == choice[kept] and choice[kept] not in stop:
            kept += 1
        n = kept + 1
        if self.rows is not None:
            self.rows.note(logits[0, :n])
        self._commit(w, n)
        settled = choice[:n]
        self.next_token = settled[-1]
        s = self.stats
        s.steps, s.tokens, s.drafted, s.accepted = s.steps + 1, s.tokens + n, s.drafted + T - 1, s.accepted + kept
        s.lookup_steps += from_lookup
        s.by_position[kept] += 1
        pass_ms = (time.perf_counter() - t0) * 1e3
        if self.drafts:
            self.guesses = self._draft(w.residual[:, :n], mx.array([settled]))
        ms = (time.perf_counter() - t0) * 1e3
        if T in self.timed_sizes:
            self.policy.observe(T, pass_ms, kept, from_lookup, match, () if from_lookup else sure[: T - 1])
            self.policy.observe_step(n, ms)
        self.timed_sizes.add(T)  # a size's first pass includes compiling its kernels: not a time to learn from
        return settled

    def generate(self, prompt: list, max_tokens: int, stop: tuple = (), sampling: Sampling | None = None) -> list:
        """A whole request, reusing what the state already holds of `prompt`."""
        self.sampling = sampling or Sampling()
        self.resume(prompt)
        self.stats = Stats()
        out = [self.read(prompt)]
        while len(out) < max_tokens and out[-1] not in stop:
            out += self.step(max_tokens - len(out), stop)
        return out
