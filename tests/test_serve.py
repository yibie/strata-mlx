"""The line protocol, spoken to a small random model."""

import threading

import mlx.core as mx
import pytest
from mlx_lm.models import qwen4_exp

from strata_mlx import serve
from strata_mlx.engine import Engine
from strata_mlx.mtp import Mtp
from test_engine import TEXT

TURN = 9
PROMPT = [TURN, 30, 31, 32, 33, TURN, 40, 41, 42, 43, 44, TURN, 50]


@pytest.fixture(scope="module")
def model():
    mx.random.seed(3)
    model = qwen4_exp.Model(qwen4_exp.ModelArgs(text_config=TEXT))
    mx.eval(model.parameters())
    model.eval()
    return model


@pytest.fixture
def session(model, monkeypatch):
    monkeypatch.setattr(serve, "END_TOKENS", (5,))  # the small model's end token
    mx.random.seed(4)
    mtp = Mtp(model.args.text)
    mx.eval(mtp.parameters())
    lines = []
    s = serve.Session(Engine(model, mtp, drafts=3, lookup=True, chunk=4, turn_token=TURN), lines.append, 4096)
    s.lines = lines
    return s


def ask(session, line):
    del session.lines[:]
    session.handle(line)
    return list(session.lines)


def ids(tokens):
    return ",".join(map(str, tokens))


def tokens_of(lines):
    return [int(l.split()[1]) for l in lines if l.startswith("T ")]


def test_a_request_is_resume_progress_tokens_done(session, model):
    lines = ask(session, f"GEN 20 {ids(PROMPT)}")
    assert lines[0] == "RESUME 0"
    kinds = [l.split()[0] for l in lines]
    assert kinds == ["RESUME"] + ["PP"] * kinds.count("PP") + ["T"] * kinds.count("T") + ["DONE"]
    assert [int(l.split()[1]) for l in lines if l.startswith("PP ")][-1] == len(PROMPT)
    assert all(l.split()[2] == str(len(PROMPT)) for l in lines if l.startswith("PP "))
    answer = tokens_of(lines)
    want = Engine(model, turn_token=TURN).generate(PROMPT, 20, stop=(5,))
    assert answer == want
    done = lines[-1].split()
    assert done[:3] == ["DONE", str(len(answer)), str(len(PROMPT))]
    assert done[5] == ("stop" if answer[-1] == 5 else "length")
    assert len(done) == 16 and float(done[3]) > 0 and float(done[4]) >= 0


def test_the_end_token_is_written_and_nothing_after_it(session, model):
    answer = tokens_of(ask(session, f"GEN 200 {ids(PROMPT)}"))
    assert answer[-1] == 5 and 5 not in answer[:-1]
    assert session.lines[-1].split()[5] == "stop"
    # the state holds the prompt and the answer up to its last token, which the next prompt brings
    assert session.engine.tokens == PROMPT + answer[:-1]


def test_a_follow_up_resumes_where_the_answer_stopped(session, model):
    answer = tokens_of(ask(session, f"GEN 6 {ids(PROMPT)}"))
    assert len(answer) == 6
    follow = PROMPT + answer + [TURN, 60, 61, TURN, 50]
    lines = ask(session, f"GEN 8 {ids(follow)}")
    assert lines[0] == f"RESUME {len(PROMPT) + len(answer) - 1}"
    assert tokens_of(lines) == Engine(model, turn_token=TURN).generate(follow, 8, stop=(5,))
    assert lines[-1].split()[8] == str(len(PROMPT) + len(answer) - 1)  # reused


def test_sampling_keys_are_read_and_a_seed_repeats(session):
    line = f"GEN 12 temperature=0.9 top_p=0.95 top_k=20 min_p=0.01 penalty_repeat=1.1 penalty_last_n=64 seed=42 cvec=0 {ids(PROMPT)}"
    first = tokens_of(ask(session, line))
    assert tokens_of(ask(session, line)) == first
    assert tokens_of(ask(session, line.replace("seed=42", "seed=43"))) != first
    assert tokens_of(ask(session, f"GEN 12 {ids(PROMPT)}")) != first


@pytest.mark.parametrize("line", ["GEN", "GEN x 1,2", "GEN 5", "GEN 5 1,2,x", "GEN 5 bogus=1 1,2", "GEN 5 999999",
                                  "GEN -1 1,2", "GEN 5 temperature=hot 1,2"])
def test_a_bad_request_is_refused_and_the_engine_carries_on(session, line):
    assert ask(session, line) in (["ERR bad request"], [f"ERR unknown command: {line[:40]}"])
    assert ask(session, f"GEN 3 {ids(PROMPT)}")[-1].startswith("DONE 3 ")


def test_a_prompt_longer_than_the_context_is_refused(session):
    session.max_context = 10
    assert ask(session, f"GEN 3 {ids(PROMPT)}")[0].startswith("ERR the prompt (13 tokens) exceeds the context")


def test_stop_ends_the_answer_with_cancel(session):
    seen = []

    def write(line):
        seen.append(line)
        if line.startswith("T ") and len([l for l in seen if l.startswith("T ")]) == 3:
            session.stop.set()

    session.write = write
    session.handle(f"GEN 200 {ids(PROMPT)}")
    assert len(tokens_of(seen)) == 3
    assert seen[-1].split()[5] == "cancel" and seen[-1].split()[1] == "3"


def test_a_line_says_whether_the_model_ran_for_it(session):
    # KeepAwake wires the arrays again only after the model ran: they are resident then (serve_lines)
    assert session.handle(f"GEN 4 {ids(PROMPT)}") is True
    assert session.handle("GEN 4 x") is False  # ERR bad request
    assert session.handle("VRAM 1") is False and session.handle("NOPE") is False
    session.stop.set()
    assert session.handle(f"GEN 200 {ids(PROMPT + [60, 61])}") is True  # stopped: what it read, it computed
    assert session.lines[-1].split()[5] == "cancel"
    session.stop.clear()


def test_stop_while_the_prompt_is_read(session):
    seen = []

    def write(line):
        seen.append(line)
        if line.startswith("PP "):
            session.stop.set()

    session.write = write
    session.handle(f"GEN 200 {ids(PROMPT)}")
    assert [l.split()[0] for l in seen] == ["RESUME", "PP", "DONE"]
    assert seen[-1].split()[1] == "0" and seen[-1].split()[5] == "cancel"
    # it says how much of the prompt it read, not the prompt's length: what the next request finds read
    read = int(seen[-1].split()[14])
    assert 0 < read == len(session.engine.tokens) < len(PROMPT) and read == int(seen[1].split()[1])
    session.stop.clear()
    session.write = session.lines.append
    lines = ask(session, f"GEN 4 {ids(PROMPT)}")
    assert lines[0] == f"RESUME {read}" and tokens_of(lines)  # and the next request runs, from there
    assert int(lines[-1].split()[14]) == len(PROMPT) - read


def test_the_other_commands_answer_in_step(session):
    assert ask(session, "GENI 5 /tmp/x 1,2")[0].startswith("ERR this engine was started without --vision")
    assert ask(session, "VRAM 100")[0].startswith("ERR ")
    assert ask(session, "") == []


def test_arguments_of_the_other_engine_are_skipped():
    opts, skipped = serve._options(["--native", "m.gguf", "--expert-cache", "auto", "--stream-experts", "--spec", "4",
                                    "--kv", "int8", "--max-context", "65536"])
    assert opts["--native"] == "m.gguf" and opts["--spec"] == "4" and opts["--max-context"] == "65536"
    assert skipped == ["--expert-cache", "--stream-experts", "--kv"]
    assert opts["--prompt-by-layer"] == opts["--read-ahead"] == "on"  # where experts are read from the SSD
    opts, skipped = serve._options(["--prompt-by-layer", "off", "--read-ahead", "off"])
    assert (opts["--prompt-by-layer"], opts["--read-ahead"], skipped) == ("off", "off", [])
    # #1: the two things that keep a rest short, each with a switch of its own
    assert (opts["--wired"], opts["--keep-awake"]) == ("on", "600")
    opts, skipped = serve._options(["--wired", "off", "--keep-awake", "0"])
    assert (opts["--wired"], opts["--keep-awake"], skipped) == ("off", "0", [])


def test_conversations_are_parked_by_default_and_short_requests_get_no_slot():
    # #12: a chat page beside another client lost its conversation to every request of the other while the cache
    # was off; `auto` is 4 GiB where the model is held whole with that to spare (check.parking_budget)
    opts, _ = serve._options([])
    assert (opts["--conversation-cache-mib"], opts["--conversation-cache-slots"],
            opts["--conversation-cache-min-tokens"]) == ("auto", "4", "256")
    opts, skipped = serve._options(["--conversation-cache-mib", "0", "--conversation-cache-min-tokens", "0"])
    assert (opts["--conversation-cache-mib"], opts["--conversation-cache-min-tokens"], skipped) == ("0", "0", [])


def keep_awake(seconds: float, limit: int):
    """A KeepAwake on a clock of its own, and what it did: ("wire", bytes) and "wake"."""
    now, did = [0.0], []
    awake = serve.KeepAwake(seconds, limit, lambda size: did.append(("wire", size)), lambda: did.append("wake"),
                            clock=lambda: now[0])
    return awake, now, did


def test_the_arrays_are_wired_only_while_the_gpu_is_kept_awake():
    awake, now, did = keep_awake(600, 99)
    assert did == [("wire", 99)] and awake.timeout() == serve.AWAKE_EVERY
    now[0] = 599.5
    awake.idle()
    assert did[-1] == "wake" and awake.timeout() == serve.AWAKE_EVERY
    now[0] = 600.5  # the window's last wait ends past it: the arrays let go of, nothing woken for after that
    awake.idle()
    assert did[-1] == ("wire", 0) and awake.timeout() is None
    awake.idle()
    assert did[-1] == ("wire", 0) and len(did) == 3
    now[0] = 2000  # a request after a long rest, read without them; wired again after it, the window from there
    awake.heard(ran=True)
    assert did[-1] == ("wire", 99) and awake.timeout() == serve.AWAKE_EVERY
    awake.heard(ran=True)  # a line inside the window: nothing to wire again
    assert did.count(("wire", 99)) == 2
    now[0] = 2599.5
    assert awake.timeout() == serve.AWAKE_EVERY


@pytest.mark.parametrize("seconds, limit", [(0, 99), (600, 0), (0, 0)])
def test_never_kept_awake_or_never_wired(seconds, limit):
    awake, now, did = keep_awake(seconds, limit)
    for t in (0.5, 300, 700, 5000):
        now[0] = t
        awake.idle()
        awake.heard(ran=True)
        awake.idle()
    assert not any(isinstance(d, tuple) for d in did)  # --keep-awake 0 wires nothing: wired asleep is the slowest
    assert did.count("wake") == (7 if seconds else 0)  # each time but the one 4,300 s after the last line


def test_a_conversation_is_saved_and_restored_over_the_protocol(session, tmp_path):
    first = tokens_of(ask(session, f"GEN 8 {ids(PROMPT)}"))
    path = str(tmp_path / "slot 0.bin")  # the path runs to the end of the line
    wait, saved = ask(session, f"SAVE {path}")
    assert wait.startswith("SWAIT transfer ") and 0 < int(wait.split()[2]) <= 3600
    head, tokens, size, ms = saved.split()
    assert head == "SAVED" and int(tokens) == len(session.engine.tokens) and int(size) > 0 and float(ms) >= 0

    ask(session, f"GEN 4 {ids([TURN, 90, 91])}")  # the slot moves on to something else
    assert ask(session, f"RESTORE {path}")[-1].split()[:2] == ["RESTORED", tokens]
    follow_up = PROMPT + first + [TURN, 60, TURN, 50]
    lines = ask(session, f"GEN 6 {ids(follow_up)}")
    assert int(lines[0].split()[1]) >= len(PROMPT)  # RESUME: the restored conversation is reused


def test_a_session_file_that_cannot_be_used_is_refused_in_one_line(session, tmp_path):
    (tmp_path / "junk").write_bytes(b"\0" * 64)
    assert ask(session, f"RESTORE {tmp_path / 'junk'}")[-1].startswith("SERR invalid 0 ")
    assert ask(session, f"RESTORE {tmp_path / 'missing'}")[-1].startswith("SERR io 0 ")
    assert ask(session, f"SAVE {tmp_path / 'no such folder' / 'x'}")[-1].startswith("SERR io 0 ")
    assert ask(session, "SAVE ")[-1] == "SERR invalid 0 missing path"
    assert ask(session, f"GEN 3 {ids(PROMPT)}")[-1].startswith("DONE ")  # and the engine is still in step


def test_what_is_left_of_a_machines_memory_goes_to_the_experts():
    from strata_mlx import paged

    gib, q2_experts, q2_dense = 2**30, int(34.0e9), int(3.6e9)
    assert not paged.all_fit(16 * gib, q2_experts, q2_dense)
    assert not paged.all_fit(32 * gib, q2_experts, q2_dense)
    assert paged.all_fit(64 * gib, q2_experts, q2_dense)
    assert not paged.all_fit(64 * gib, q2_experts, q2_dense, 20 * gib)  # and the floats kept for prompts
    store = paged.ExpertStore(0)
    held = mx.get_active_memory()
    assert store.fit(32 * gib) == 32 * gib - paged.left_to_system(32 * gib) - paged.WORKING - held
    # and no more than what the machine had free or in its file cache before anything was loaded, less what is
    # loaded and the working memory (#6)
    ruled = store.budget
    assert store.fit(32 * gib, available=64 * gib) == ruled
    assert store.fit(32 * gib, available=13 * gib) == 13 * gib - held - paged.WORKING < ruled
    assert store.fit(32 * gib, available=2 * gib) == paged.LEAST
    free, cached = paged.available_memory()  # what `vm_stat` calls "Pages free" and "File-backed pages"
    assert 0 < free < paged.machine_memory() and 0 <= cached < paged.machine_memory()
    assert store.fit(4 * gib) == int(0.5e9)  # never nothing


def test_a_last_addition_that_ends_past_the_window_still_lets_the_arrays_go():
    awake, now, did = keep_awake(600, 99)
    now[0] = 599.99
    awake.wake = lambda: now.__setitem__(0, 600.01)  # the window ends while the GPU computes its addition
    awake.idle()
    assert awake.timeout() == serve.AWAKE_EVERY  # not a wait for ever with the arrays still wired
    awake.idle()
    assert did[-1] == ("wire", 0) and awake.timeout() is None


def test_only_a_line_the_model_ran_for_wires_the_arrays_again():
    awake, now, did = keep_awake(600, 99)
    now[0] = 700
    awake.idle()
    awake.heard(ran=False)  # an ERR, a SAVE: the arrays are not resident, wiring them would wait ~250 ms
    assert did == [("wire", 99), ("wire", 0)] and awake.timeout() == serve.AWAKE_EVERY
    awake.heard(ran=True)
    assert did[-1] == ("wire", 99)


class Lines:
    """A queue of the server's lines on a KeepAwake's clock: "" is a wait that ends with no line."""

    def __init__(self, lines, now):
        self.lines, self.now, self.waits = list(lines), now, []

    def get(self, timeout=None):
        self.waits.append(timeout)
        line = self.lines.pop(0)
        if line:
            return line
        assert timeout is not None, "a wait for ever with no line to come"
        self.now[0] += timeout
        raise serve.queue.Empty


def test_the_loop_keeps_the_gpu_awake_lets_the_arrays_go_and_wires_them_after_a_request():
    awake, now, did = keep_awake(3, 99)
    handled = []

    def handle(line):
        handled.append(line)
        return line.startswith("GEN")

    lines = Lines(["GEN 1 5", "", "", "", "VRAM 1", "", "GEN 1 6", "QUIT"], now)
    serve.serve_lines(lines, handle, awake)
    assert handled == ["GEN 1 5", "VRAM 1", "GEN 1 6"]
    # wired at the start; additions at 1 and 2 s, the window over at 3, let go of; the VRAM line starts a window
    # but wires nothing; an addition; the request, then wired again
    assert did == [("wire", 99), "wake", "wake", ("wire", 0), "wake", ("wire", 99)]
    assert lines.waits[3] == serve.AWAKE_EVERY and lines.waits[4] is None  # past the window, unwired: no wakes

