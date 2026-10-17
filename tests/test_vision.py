"""Tiny synthetic models: test vision state, shifted draft inputs and attention positions."""
import struct
import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models import qwen4_exp
from strata_mlx import serve
from strata_mlx.conversations import Parked
from strata_mlx.engine import Engine
from strata_mlx.mtp import Mtp
from strata_mlx.vision import Image, ImageRoPE, Vision
from test_engine import TEXT

PAD = 17
PROMPT = [9,20] + [PAD]*6 + [9,21,22,23,24,25,26,27,28,29,30]


@pytest.fixture
def model():
    mx.random.seed(3)
    # Force sparse selection even in this tiny fixture.
    model = qwen4_exp.Model(qwen4_exp.ModelArgs(text_config={**TEXT, "indexer_budget": 8}))
    mx.eval(model.parameters())
    model.eval()
    return model


def image(value=1):
    return Vision([Image(2, 3, 2, np.full((6,64), value, dtype="<f4"))])


def test_rope_matches_scalar_formula(model):
    v = image()
    rope = ImageRoPE(model.model.rope, v)
    cells = mx.arange(25)[None]
    c, s = rope(cells)
    pos = v.positions([list(range(25))])
    inv = np.array(model.model.rope.inv_freq)
    angle = pos[..., np.arange(len(inv)) % 3] * inv
    angle = np.concatenate([angle, angle], axis=-1)
    assert np.allclose(np.array(c), np.cos(angle), atol=1e-6)
    assert np.allclose(np.array(s), np.sin(angle), atol=1e-6)


def test_embedding_slice_and_shifted_mtp(model):
    e = Engine(model, Mtp(model.args.text), drafts=3, chunk=4)
    e.vision = image(2)
    ids = mx.array([PROMPT[1:6]])
    emb = e.embed(ids, 1)
    assert mx.all(emb[:, 1:] == 2).item()
    assert mx.array_equal(emb[:, :1], e.m.embed_tokens(ids[:, :1])).item()
    seen = []
    original = e.embed
    def spy(ids, start):
        seen.append(start)
        return original(ids, start)
    e.embed = spy
    e._cells(mx.zeros((1,4,256)), mx.array([PROMPT[1:5]]))
    assert seen == [1]  # MTP cell zero sees the embedding of physical token one


@pytest.mark.parametrize("fused", [False, True])
def test_draft_verification_matches_no_draft_with_sparse_images(model, fused):
    def answer(drafts):
        e = Engine(model, Mtp(model.args.text) if drafts else None, drafts=drafts, chunk=4, fused=fused)
        e.vision = image()
        first = e.read(PROMPT)
        out = [first]
        while len(out) < 12: out += e.step(12-len(out))
        return out
    assert answer(3) == answer(0)


def make_session(model, tmp_path, monkeypatch, value=1):
    monkeypatch.setattr("strata_mlx.vision.IMAGE_PAD", PAD)
    monkeypatch.setattr(serve, "END_TOKENS", (5,))
    path = tmp_path / "image.sve"
    path.write_bytes(struct.pack("<5i",0x31455653,6,3,2,64) + image(value).images[0].rows.tobytes())
    lines = []
    e = Engine(model, Mtp(model.args.text), drafts=3, chunk=4, turn_token=9)
    s = serve.Session(e, lines.append, 256, parked=Parked(1<<28, log=lambda _:None), vision=True)
    def ask(prompt=PROMPT, kind="GENI", limit=8):
        lines.clear()
        s.handle(f"{kind} {limit} " + (str(path)+" " if kind == "GENI" else "") + ",".join(map(str,prompt)))
        return list(lines)
    return s, path, ask


def test_changed_image_same_path_never_reuses_and_text_resets(model, tmp_path, monkeypatch):
    s, path, ask = make_session(model,tmp_path,monkeypatch)
    first = ask()
    assert first[0] == "RESUME 0"
    assert int(ask()[0].split()[1]) > 0
    path.write_bytes(path.read_bytes()[:20] + image(2).images[0].rows.tobytes())
    assert ask()[0] == "RESUME 0"
    assert ask(kind="GEN")[0] == "RESUME 0"
    assert s.engine.vision is None
    # Both image namespaces are parked and remain independently reusable.
    assert int(ask()[0].split()[1]) > 0
    with pytest.raises(ValueError, match="RAM parking"):
        s.engine.save(str(tmp_path / "forbidden.safetensors"))


def test_malformed_input_preserves_previous_state(model,tmp_path,monkeypatch):
    s,path,ask = make_session(model,tmp_path,monkeypatch)
    ask()
    tokens = list(s.engine.tokens)
    path.write_bytes(b"bad")
    assert ask()[0].startswith("ERR")
    assert s.engine.tokens == tokens
    assert ask([9,3,4],kind="GEN")[-1].startswith("DONE")


def test_cancel_during_prefill_and_resume(model,tmp_path,monkeypatch):
    s,path,ask = make_session(model,tmp_path,monkeypatch)
    write = s.write
    def stop(line):
        write(line)
        if line.startswith("PP "): s.stop.set()
    s.write = stop
    assert ask()[-1].split()[5] == "cancel"
    s.stop.clear()
    s.write = write
    assert ask()[-1].split()[5] in ("stop", "length")
    assert s.engine.vision.key == image().key


def test_layer_major_image_read_and_cancel_rollback(tmp_path):
    from test_engine import _paged
    model = _paged(tmp_path, 1 << 30)
    mtp = Mtp(model.args.text)
    engines = [Engine(model, mtp, drafts=3, chunk=4, piece=piece, turn_token=9) for piece in (0,8)]
    answers = []
    for e in engines:
        model.expert_store.clear()
        e.vision = image()
        first = e.read(PROMPT)
        out = [first]
        while len(out) < 8: out += e.step(8-len(out))
        answers.append(out)
    assert answers[0] == answers[1]
    e = engines[1]
    e.reset()
    e.vision = image()
    def stop(): raise RuntimeError("cancel piece")
    with pytest.raises(RuntimeError, match="cancel piece"):
        e.read(PROMPT, on_step=stop)
    assert e.tokens == [] and all(c.offset == 0 for c in e.attn.values())
    assert e.mtp_cache.offset == 0
    assert e.read(PROMPT) == answers[0][0]


def test_image_destination_is_taken_before_parking_evicts_it(model,tmp_path,monkeypatch):
    s,path,ask = make_session(model,tmp_path,monkeypatch)
    ask()
    conv = s.engine.detach()
    s.engine.attach(conv)
    s.parked.budget = conv.bytes
    ask([9,3,4], kind="GEN")
    assert len(s.parked.entries) == 1
    text = s.engine.detach()
    s.engine.attach(text)
    s.parked.budget = max(s.parked.bytes, text.bytes)
    # Each fits, but both do not: eviction must not destroy the destination before selection.
    assert int(ask()[0].split()[1]) > 0


@pytest.mark.parametrize("arguments,fusion,pressure", [
    (["--vision", "--fusion-limit", "6", "--follow-pressure", "on"], "6", "on"),
    (["--follow-pressure", "on", "--vision", "--fusion-limit", "8"], "8", "on"),
    (["--fusion-limit", "auto", "--vision", "--follow-pressure", "off"], "auto", "off"),
])
def test_vision_coexists_with_merged_fusion_and_pressure_options(arguments, fusion, pressure):
    opts, skipped = serve._options(arguments)
    assert opts["--vision"] is True
    assert opts["--fusion-limit"] == fusion
    assert opts["--follow-pressure"] == pressure
    assert skipped == []
