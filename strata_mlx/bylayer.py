"""A prompt read a layer at a time, where experts are read from the SSD.

`Engine.read` takes a prompt in chunks, each through all the layers, and a chunk reads each layer's experts and
keeps none: a prompt of four chunks reads every layer four times, 99 GB for 8,192 tokens (e37). Here the layers
are outside and the chunks inside: a piece of the prompt, several chunks, goes through layer 1 chunk by chunk,
then through layer 2, and a layer's experts are read once and held until its last chunk (`ExpertStore.hold`);
with `ahead`, the next layer's are read by a thread while this one is computed. What a chunk computes at a
layer is `Engine._forward_fused`'s, operation for operation, so the state a prompt leaves is the same bit for
bit, checkpoints and all.

What travels from layer to layer is each chunk's residual as the fused pass has it, (before, out, inject).
The states that cannot be cut back (delta-net, the n-gram layer's convolution) move on chunk by chunk inside a
layer and are kept where a checkpoint falls; the conversation gets them at the piece's end. Until then a piece
has written only rows of the attention caches and cells of the draft head's, so a piece that is stopped is
undone by cutting those back, and the pieces before it stay read.
"""

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask

from .drafting import MAX_WINDOW

PIECE = 8192  # tokens of prompt through the layers at once: larger reads less and is no faster, and holds more
SAID_UPTO = 64  # Strata's server drops a line of progress this near the prompt's end


def pieces(subs: list, piece: int, chunk: int, periodic: int) -> list:
    """Chunks (from, to) into pieces of at most `piece` tokens, whole chunks and at least one each. A piece ends
    where a periodic checkpoint falls (every `periodic` tokens); a last one of less than a chunk goes with the
    one before it."""
    out, now = [], []
    for s in subs:
        if now and (s[1] - now[0][0] > piece or now[-1][1] % periodic == 0):
            out.append(now)
            now = []
        now.append(s)
    out.append(now)
    if len(out) > 1 and out[-1][-1][1] - out[-1][0][0] < chunk:
        out[-2:] = [out[-2] + out[-1]]
    return out


def read(e, prompt: list, cuts: dict, on_chunk=None, on_step=None) -> int:
    """What `Engine.read`'s loop does for prompt[len(e.tokens):], the checkpoints of `cuts` taken.

    `on_chunk(tokens, of)` is called after each layer of a piece with a count in proportion to the layers done,
    and at a piece's end with the tokens read; `on_step()` after every chunk through half a layer. If either
    raises inside a piece, the piece is undone before the exception goes on."""
    from .engine import PERIODIC, _arrays

    start, n = len(e.tokens), len(prompt)
    bounds = sorted({*cuts, *range(start + e.chunk, n, e.chunk), n})
    subs = [(a, b) for a, b in zip([start, *bounds[:-1]], bounds) if b > a]
    store = e.model.expert_store
    try:
        for part in pieces(subs, e.piece, e.chunk, PERIODIC):
            first = len(e.tokens)
            try:
                _piece(e, prompt, part, cuts, on_chunk, on_step)
            except BaseException:
                _undo(e, first)
                raise
            mx.eval(*_arrays([e.ple_conv, list(e.gdn.values())]))
            if on_chunk:
                on_chunk(len(e.tokens), n)  # the piece is read: stopped here, it stays
    finally:
        store.release()
    return e.next_token


def _undo(e, first: int):
    """Cut back what a piece wrote before it was given up: nothing else of the conversation has changed."""
    from .attention import AttnCache

    for i, cache in e.attn.items():
        if first == 0:
            e.attn[i] = AttnCache()  # cut to nothing it would still hold arrays, which a session file would get
        elif cache.offset > first:
            cache.trim(cache.offset - first)
    if e.mtp_cache is not None:
        if first == 0:
            e.mtp_cache = e.mtp.make_cache()
        elif e.mtp_cache.offset > first:
            e.mtp_cache.trim(e.mtp_cache.offset - first)


def _piece(e, prompt: list, subs: list, cuts: dict, on_chunk, on_step):
    from .engine import _arrays, _Slot

    m, n, first, last = e.m, len(prompt), subs[0][0], subs[-1][1]
    store, layers = e.model.expert_store, len(e.m.layers)
    ids = [mx.array([prompt[a:b]]) for a, b in subs]
    # the n-gram context each chunk finds: the tokens before it (Engine._commit works it out chunk by chunk)
    ctx, history, contexts = e.context.shape[1], e.context.tolist()[0], []
    for a, b in subs:
        contexts.append(mx.array([history[-ctx:]], dtype=e.context.dtype))
        history = history + prompt[a:b]
    contexts.append(mx.array([history[-ctx:]], dtype=e.context.dtype))
    before = [mx.tile(e.embed(i, a), (1, 1, m.hc)) for i, (a, _) in zip(ids, subs)]
    nothing = [(mx.zeros((1, i.shape[1], m.args.hidden_size), dtype=before[0].dtype),
                mx.zeros((1, i.shape[1], m.hc), dtype=before[0].dtype)) for i in ids]
    out, inject = [x[0] for x in nothing], [x[1] for x in nothing]
    # the states that cannot be cut back, as the chunks that end at a checkpoint leave them, and the last
    wanted = {c for c, (_, b) in enumerate(subs) if b in cuts or b == last}
    gdn_after, ple_after = [dict() for _ in subs], [e.ple_conv for _ in subs]

    def step():
        if on_step:
            on_step()

    if e.ahead:
        store.ahead(0)
    for i, layer in enumerate(m.layers):
        store.hold(i)
        if e.ahead and i + 1 < layers:
            store.ahead(i + 1)
        if layer.ple is not None:
            ple_conv = e.ple_conv
            for c in range(len(subs)):
                slot = _Slot(ple_conv)
                h = e._write(before[c], out[c], inject[c])
                before[c], (out[c], inject[c]) = h + layer.ple(h, ids[c], contexts[c], slot), nothing[c]
                ple_conv = ple_after[c] = slot.value
        mix, _ = e.fused[i]
        state = e.gdn.get(i)
        for c in range(len(subs)):
            T = ids[c].shape[1]
            if i in e.gdn:
                args = (before[c], out[c], inject[c], *state)
                before[c], out[c], inject[c], *record = mix(*args) if T <= MAX_WINDOW else e._mix_block(i, *args)
                record = e._record(i, record, True)
                state = (record[0], record[6])
                if c in wanted:
                    gdn_after[c][i] = state
            else:
                x, before[c], inject[c] = e._hc((i, 0), T)(before[c], out[c], inject[c])
                mask = create_attention_mask(before[c], e.attn[i])  # this layer's cache ends where the chunk begins
                out[c] = layer.self_attn(x, e.rope, mask, e.attn[i], e.attn[i].indexer)
            mx.eval(before[c], out[c], inject[c], *_arrays([state, e.attn.get(i) and e.attn[i].keys]))
            step()
        for c in range(len(subs)):
            before[c], out[c], inject[c] = e._ffn_block(i, before[c], out[c], inject[c])
            mx.eval(before[c], out[c], inject[c])
            step()
        said = first + (last - first) * (i + 1) // (layers + 1)
        if on_chunk and n - said > SAID_UPTO:
            on_chunk(said, n)
    store.kept = None  # the last layer's experts are let go
    if e.ahead and last < n:
        store.ahead(0)  # the next piece's first layer, behind what is left of this one
    # the draft head's cells of every chunk but the prompt's last, whose cells its guesses are made with
    residual = None
    for c, (a, b) in enumerate(subs):
        x, residual, _ = e._hc("out", b - a)(before[c], out[c], inject[c])
        before[c] = out[c] = inject[c] = None
        if b < n and e.drafts:
            e._cells(residual, mx.array([prompt[a + 1 : b + 1]]))
            mx.eval(*_arrays([e.mtp_cache.keys, e.mtp_cache.values, e.mtp_cache.indexer.keys]))
            step()
    # and the conversation moves on, chunk by chunk as `Engine.read` moves it: nothing stops it in here
    for c, (a, b) in enumerate(subs):
        e.tokens += prompt[a:b]
        e.context = contexts[c + 1]
        if e.lookup:
            e.lookup.extend(prompt[a:b])
        if c in wanted:  # elsewhere the states are not the conversation's, and nothing reads them there
            e.gdn, e.ple_conv = dict(gdn_after[c]), ple_after[c]
        if b == n:
            logits = e.model.lm_head(x[:, -1:])
            e.next_token = e._choose(logits[0], prompt[-1:], prompt[:-1])[0]
            if e.rows is not None:
                e.rows.add(prompt)
                e.rows.note(logits[0])
            if e.drafts:
                e.guesses = e._draft(residual, mx.array([prompt[a + 1 :] + [e.next_token]]))
        if b in cuts:
            e._checkpoint(**cuts[b])
