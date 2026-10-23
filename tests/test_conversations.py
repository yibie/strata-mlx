"""Which checkpoint leaves, and conversations parked while another is decoded: the answers are those of an engine
that never left the conversation, and only what is new is read."""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models import qwen4_exp

from strata_mlx import serve
from strata_mlx.conversations import Parked, reach, resume, victim
from strata_mlx.engine import Engine
from strata_mlx.mtp import Mtp
from test_engine import TEXT

TURN = 9
SYSTEM = [TURN, 20, 21, 22, 23, 24, 25, 26]  # what every chat of this client begins with
A = SYSTEM + [TURN, 30, 31, 32, 33, TURN, 50]
B = SYSTEM + [TURN, 40, 41, 42, TURN, 50]
OTHER = [TURN, 70, 71, 72, 73, 74, TURN, 50]  # a request that shares nothing but its first token


@pytest.fixture(scope="module")
def model():
    mx.random.seed(3)
    model = qwen4_exp.Model(qwen4_exp.ModelArgs(text_config=TEXT))
    mx.eval(model.parameters())
    model.eval()
    return model


def engine_of(model, **kw):
    mx.random.seed(4)
    mtp = Mtp(model.args.text)
    mx.eval(mtp.parameters())
    return Engine(model, mtp, drafts=3, lookup=True, chunk=4, turn_token=TURN, root=len(SYSTEM), **kw)


def quiet(budget=1 << 30, **kw):
    log = []
    parked = Parked(budget, log=log.append, **kw)
    parked.lines = log
    return parked


def ask(engine, parked, prompt, n=10, keep=True, pin=0):
    """A request as the server runs it. Returns (tokens not read, the answer)."""
    reused = resume(engine, parked, prompt)
    out = [engine.read(prompt, keep=keep, pin=pin)]
    engine.one_shot = not keep
    while len(out) < n:
        out += engine.step(n - len(out))
    return reused, out


def alone(model, *prompts, n=10):
    """The last prompt's answer from an engine that was asked these and nothing else."""
    engine = engine_of(model)
    return [engine.generate(p, n) for p in prompts][-1]


# ---- which checkpoint leaves

def test_the_root_stays_and_the_others_leave_by_least_recent_use():
    assert victim([1, 2, 3, 4], 3) == 1  # not the root, though it is the oldest
    assert victim([1, 5, 2, 4], 3) == 2
    assert victim([9, 5, 2, 4], 3) == 2
    assert victim([1, 2, 3], 1) == 0  # no room to keep root and leaf apart: the oldest
    assert victim([4], 3) == 0


def test_a_tail_goes_first_unless_it_is_the_newest_and_a_pinned_one_never():
    assert victim([1, 2, 3, 4], 3, tail=[False, False, True, False]) == 2
    assert victim([1, 2, 3, 4], 3, tail=[False, False, False, True]) == 1  # the one just made
    assert victim([1, 2, 3, 4], 3, pinned=[False, True, False, False]) == 2
    assert victim([1, 2, 3, 4], 3, tail=[False, True, False, False], pinned=[False, True, False, False]) == 2
    assert victim([1, 3, 2, 4], 3, pinned=[False, True, True, True]) == 2  # all pinned: the oldest of them


def test_reach_is_the_whole_conversation_or_its_last_checkpoint_in_common():
    cps = [SimpleNamespace(at=3), SimpleNamespace(at=6)]
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    assert reach(tokens, cps, tokens + [9, 10]) == (8, None)
    assert reach(tokens, cps, tokens) == (6, cps[1])  # the prompt's last token is read in any case
    assert reach(tokens, cps, [1, 2, 3, 4, 5, 0, 0]) == (3, cps[0])
    assert reach(tokens, cps, [1, 2, 0]) == (0, None)
    assert reach([], [], [1, 2]) == (0, None)


# ---- the conversations parked

def conv(tokens, size=100, pinned=False):
    return SimpleNamespace(tokens=tokens, checkpoints=[], bytes=size, pinned=pinned)


def test_the_oldest_leaves_when_slots_or_bytes_run_out_and_one_with_a_pinned_prefix_last():
    parked = quiet(budget=250, slots=3)
    a, b, c = conv([1, 1]), conv([2, 2]), conv([3, 3])
    assert parked.park(a) and parked.park(b) and parked.entries == [a, b]
    assert parked.park(c) and parked.entries == [b, c] and parked.evictions == 1  # 300 bytes do not fit
    parked = quiet(budget=10_000, slots=2)
    a = conv([1, 1], pinned=True)
    assert parked.park(a) and parked.park(b) and parked.park(c)
    assert parked.entries == [a, c]
    assert "parked 2 tokens" in parked.lines[-1] and parked.lines[-1].endswith("parked=2 bytes=200 evictions=1")


def test_a_parked_conversation_keeps_its_root_its_pinned_prefix_and_its_last_two_checkpoints():
    parked, c = quiet(), conv([1, 2, 3])
    c.checkpoints = [SimpleNamespace(at=at, pinned=at == 3) for at in range(1, 8)]
    parked.park(c)
    assert [cp.at for cp in c.checkpoints] == [1, 3, 6, 7]


def test_what_is_too_short_or_too_large_is_not_parked_and_evicts_nothing():
    parked = quiet(budget=250, slots=3, min_tokens=3)
    kept = conv([1, 2, 3])
    assert parked.park(kept)
    assert not parked.park(conv([4, 5])) and "skip parking (2 tokens, below the 3 minimum; parked=1)" in parked.lines[-1]
    assert not parked.park(conv([6, 7, 8], size=251)) and parked.lines[-1].startswith("skipped 3 tokens")
    assert not parked.park(conv([]))
    assert parked.entries == [kept] and parked.evictions == 0


def test_short_requests_between_two_turns_take_the_slots_unless_below_the_minimum():
    # #12: a chat beside an editor or an API client. Slots are given up oldest first whatever their size, so four
    # short conversations in a row left a long one out; the server's --conversation-cache-min-tokens (256)
    # keeps them from a slot
    def conv(tokens, size):
        return SimpleNamespace(tokens=tokens, bytes=size, checkpoints=[], pinned=False, image_key=None)

    long = conv(SYSTEM + list(range(100, 15_100)), 900 << 20)
    shorts = [conv(SYSTEM + [TURN, 200 + i, TURN, 50], 1 << 20) for i in range(4)]
    for minimum, kept in ((0, False), (256, True)):
        parked = quiet(4096 << 20, slots=4, min_tokens=minimum)
        parked.park(long)
        for s in shorts:
            parked.park(s)
        assert (long in parked.entries) is kept, (minimum, [len(c.tokens) for c in parked.entries])


def test_a_parked_conversation_that_a_new_one_continues_is_dropped():
    parked = quiet()
    parked.park(conv([1, 2]))
    parked.park(conv([7, 8]))
    longer = conv([1, 2, 3])
    parked.park(longer)
    assert [c.tokens for c in parked.entries] == [[7, 8], [1, 2, 3]]
    assert any(line.startswith("dropped 1 superseded") for line in parked.lines)


def test_the_best_is_the_one_that_spares_most_and_of_equals_the_one_parked_last():
    parked = quiet()
    parked.entries = [conv(tokens) for tokens in ([9, 9], [9, 9], [1, 2, 3, 4], [1, 2, 0], [5, 6])]
    assert parked.best([1, 2, 3, 4, 5, 6])[:2] == (2, 4)
    assert parked.best([9, 9, 1])[:2] == (1, 2)
    assert parked.best([8, 8]) == (None, 0, None)


# ---- with an engine

def test_two_conversations_in_turns_are_each_read_once_and_answer_as_if_alone(model):
    engine, parked = engine_of(model), quiet()
    _, a1 = ask(engine, parked, A)
    reused, o1 = ask(engine, parked, OTHER)
    assert reused == 0 and len(parked.entries) == 1  # A is parked, not cut back
    a_next = A + a1 + [TURN, 60, 61, TURN, 50]
    o_next = OTHER + o1 + [TURN, 62, TURN, 50]
    reused, a2 = ask(engine, parked, a_next)
    assert reused == len(A) + len(a1) - 1  # all of it but the answer's last token, which was not decoded
    assert a2 == alone(model, A, a_next)
    reused, o2 = ask(engine, parked, o_next)
    assert reused == len(OTHER) + len(o1) - 1 and o2 == alone(model, OTHER, o_next)
    assert [line.split()[0] for line in parked.lines] == ["parked", "parked", "restored", "parked", "restored"]


def test_a_new_chat_starts_from_the_system_prompt_and_the_chat_it_left_goes_on(model):
    engine, parked = engine_of(model), quiet()
    _, a1 = ask(engine, parked, A)
    assert engine.checkpoints[0].at == len(SYSTEM)  # where the first message ends
    reused, b1 = ask(engine, parked, B)
    assert reused == len(SYSTEM) and b1 == alone(model, B)
    assert [len(c.tokens) for c in parked.entries] == [len(A) + len(a1) - 1]
    a_next = A + a1 + [TURN, 60, 61, TURN, 50]
    reused, a2 = ask(engine, parked, a_next)
    assert reused == len(A) + len(a1) - 1 and a2 == alone(model, A, a_next)
    # and a third chat, while A is held and B parked, starts from the copy either has of the system prompt
    c = SYSTEM + [TURN, 45, 46, TURN, 50]
    reused, c1 = ask(engine, parked, c)
    assert reused == len(SYSTEM) and c1 == alone(model, c)
    assert len(parked.entries) == 2


def test_a_one_shot_request_leaves_no_checkpoints_and_is_not_parked(model):
    engine, parked = engine_of(model), quiet()
    _, a1 = ask(engine, parked, A)
    probe = SYSTEM + [TURN, 80, 81, 82, 83, 84, 85, TURN, 50]
    reused, _ = ask(engine, parked, probe, keep=False)
    assert reused == len(SYSTEM)
    assert [c.at for c in engine.checkpoints] == [len(SYSTEM)]  # only the one it started from
    a_next = A + a1 + [TURN, 60, 61, TURN, 50]
    reused, a2 = ask(engine, parked, a_next)
    assert reused == len(A) + len(a1) - 1 and a2 == alone(model, A, a_next)
    assert parked.entries == []  # the probe's conversation went nowhere


def test_an_answer_asked_again_or_rendered_anew_parks_nothing(model):
    engine, parked = engine_of(model), quiet()
    _, a1 = ask(engine, parked, A)
    ask(engine, parked, A)  # the same prompt again
    ask(engine, parked, A + [a1[0], 99, TURN, 60, TURN, 50])  # the answer as the template renders it now
    assert parked.entries == [] and parked.lines == []


def test_a_pinned_prefix_outlives_the_other_checkpoints_and_its_questions_are_not_parked(model):
    engine, parked = engine_of(model, checkpoints=3), quiet()
    document = [TURN] + list(range(20, 40))
    pin = 13  # inside the first message
    asked = []
    for q in range(4):
        prompt = document + [100 + q, 101, TURN, 50]
        reused, out = ask(engine, parked, prompt, pin=pin)
        asked.append(reused)
        assert out == alone(model, prompt)
        assert any(c.at == pin and c.pinned for c in engine.checkpoints) and len(engine.checkpoints) <= 3
    assert asked == [0, pin, pin, pin] and parked.entries == []


def test_without_a_budget_the_engine_holds_one_conversation_as_before(model):
    engine, off = engine_of(model), Parked(0)
    _, a1 = ask(engine, off, A)
    assert ask(engine, off, OTHER)[0] == 0 and off.entries == []
    assert ask(engine, off, A + a1 + [TURN, 60, TURN, 50])[0] == 0  # read again


# ---- over the line protocol

def test_the_server_parks_and_takes_back_and_reads_the_marks_of_a_request(model, monkeypatch):
    monkeypatch.setattr(serve, "END_TOKENS", (5,))
    lines, parked = [], quiet()
    session = serve.Session(engine_of(model), lines.append, 4096, parked=parked)

    def gen(prompt, marks=""):
        del lines[:]
        session.handle(f"GEN 8 {marks}{','.join(map(str, prompt))}")
        assert lines[-1].startswith("DONE"), lines
        return int(lines[0].split()[1]), [int(l.split()[1]) for l in lines if l.startswith("T ")]

    _, a1 = gen(A)
    assert gen(SYSTEM + [TURN, 80, 81, TURN, 50], "ckpt=0 ")[0] == len(SYSTEM)
    assert session.engine.one_shot
    kept = len(A) + len(a1) - (0 if a1[-1] == 5 else 1)
    assert gen(A + a1 + [TURN, 60, TURN, 50])[0] == min(kept, len(session.engine.tokens))
    assert parked.entries == []
    assert gen(A[:5] + [90, 91, 92, TURN, 50], "pin=5 ")[0] == 0
    assert any(c.pinned and c.at == 5 for c in session.engine.checkpoints)
