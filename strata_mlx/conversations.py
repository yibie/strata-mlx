"""strata_mlx/conversations.py - which checkpoint leaves, and conversations parked while another is decoded.

Both are Strata's, written anew for this engine (its `conv_cache.hpp` and `conversation_cache.hpp`, MIT).

The engine holds one conversation. Of its state the attention caches can be cut back to any length; the delta-net
layers' recurrent states cannot, so they are kept at a few places, the checkpoints (110 MiB each with this model),
and a request goes on from the last one its prompt still agrees with. The checkpoints kept are a chain, each a
prefix of the next, and the first is the deepest point every request so far has shared: in practice the end of
the system prompt, which a new chat of the same client starts with. So that one stays, and the others leave by
least recent use (`victim`).

A request that shares little with the conversation held cut that conversation back to what they share, and the
next turn of it read everything again. With `--conversation-cache-mib` the conversation is parked instead, whole,
and the request goes on from a copy of the part in common; when a prompt continues a parked conversation it is
taken back. Nothing is computed for either: arrays change hands. No session id is needed, only the tokens.
"""

import sys
import time

MIB = 1 << 20


def victim(stamps: list, cap: int, tail: list | None = None, pinned: list | None = None) -> int:
    """Which of a chain of checkpoints leaves once it holds more than `cap`: its index. `stamps` are their last
    uses. The first, the root, stays; a `pinned` one stays (a prefix that requests keep branching from); a `tail`
    one, taken near a prompt's end for a branch of that request only, is the first to go unless it is the newest;
    then the least recently used. With no room to keep root and leaf apart (`cap` under 2) the oldest leaves."""
    n = len(stamps)
    if cap < 2 or n < 2:
        return 0
    newest = max(range(n), key=lambda i: stamps[i])
    free = [i for i in range(1, n) if not (pinned and pinned[i])]
    for i in free:
        if tail and tail[i] and i != newest:
            return i
    # every other one pinned too: the oldest of them, so that the chain keeps to its size
    return min(free or range(1, n), key=lambda i: stamps[i])


def reach(tokens: list, checkpoints: list, prompt: list):
    """How much of `prompt` a conversation of `tokens` with these checkpoints spares a request: (tokens, the
    checkpoint to go back to, or None when the whole conversation is the prompt's start or nothing of it is)."""
    same, most = 0, min(len(prompt) - 1, len(tokens))  # the prompt's last token is always read: it gives the answer
    while same < most and prompt[same] == tokens[same]:
        same += 1
    if same == len(tokens):
        return same, None
    cp = max((c for c in checkpoints if c.at <= same), key=lambda c: c.at, default=None)
    return (cp.at, cp) if cp is not None else (0, None)


class Parked:
    """The conversations parked, the oldest first, within a budget of bytes and a number of slots."""

    def __init__(self, budget: int, slots: int = 4, min_tokens: int = 0, log=None):
        self.budget, self.slots, self.min_tokens = budget, slots, min_tokens
        self.entries, self.evictions = [], 0
        self.log = log or (lambda line: print(f"strata-mlx: conversation cache: {line}", file=sys.stderr))

    @property
    def bytes(self) -> int:
        return sum(c.bytes for c in self.entries)

    def _counts(self) -> str:
        return f"parked={len(self.entries)} bytes={self.bytes}"

    def best(self, prompt: list, image_key=None):
        """The parked conversation that spares `prompt` the most: (its index, tokens, checkpoint as `reach` gives
        it). Of equals the one parked last; (None, 0, None) when none spares anything."""
        found = (None, 0, None)
        for i in reversed(range(len(self.entries))):
            if getattr(self.entries[i], "image_key", None) != image_key:
                continue
            n, cp = reach(self.entries[i].tokens, self.entries[i].checkpoints, prompt)
            if n > found[1]:
                found = (i, n, cp)
        return found

    def take(self, i: int):
        return self.entries.pop(i)

    def park(self, conv) -> bool:
        """Keep a conversation the engine is leaving. False when it is not kept: too short to be worth a slot,
        or larger than the budget."""
        n = len(conv.tokens)
        if n < max(self.min_tokens, 1):
            if n:
                self.log(f"skip parking ({n} tokens, below the {self.min_tokens} minimum; parked={len(self.entries)})")
            return False
        # of its checkpoints the root, a pinned prefix and the last two are what a prompt comes back to: the root
        # for a new chat, the last ones for this chat's next turn rendered anew. The others are 110 MiB each.
        cps = conv.checkpoints
        conv.checkpoints = [c for j, c in enumerate(cps) if j == 0 or c.pinned or j >= len(cps) - 2]
        t0, size = time.perf_counter(), conv.bytes
        if size > self.budget or self.slots < 1:
            self.log(f"skipped {n} tokens ({size / MIB:.0f} MiB, the budget is {self.budget / MIB:.0f}); {self._counts()}")
            return False
        # a parked conversation this one continues can do nothing this one cannot
        older = [c for c in self.entries if not (getattr(c, "image_key", None) == getattr(conv, "image_key", None)
                                               and len(c.tokens) <= n and c.tokens == conv.tokens[: len(c.tokens)])]
        if len(older) < len(self.entries):
            dropped, self.entries = len(self.entries) - len(older), older
            self.log(f"dropped {dropped} superseded conversations; parked={len(self.entries)}")
        while self.entries and (len(self.entries) >= self.slots or self.bytes + size > self.budget):
            held = [i for i, c in enumerate(self.entries) if not c.pinned]
            self.entries.pop(held[0] if held else 0)  # the oldest; one that holds a pinned prefix only at need
            self.evictions += 1
        self.entries.append(conv)
        self.log(f"parked {n} tokens ({size / MIB:.0f} MiB) in {(time.perf_counter() - t0) * 1e3:.1f} ms; "
                 f"{self._counts()} evictions={self.evictions}")
        return True


def resume(engine, parked: Parked | None, prompt: list, image_key=None) -> int:
    """`engine.resume(prompt)` with parked conversations to choose from, and the one held parked when the prompt
    leaves it. Returns how many tokens of the prompt need not be read."""
    if getattr(engine, "image_key", None) != image_key:
        # Select/take the destination before parking the departing namespace: with room for just one
        # conversation, parking first could evict the very image conversation being resumed.
        destination = None
        if parked is not None and parked.budget > 0:
            i, there, cp = parked.best(prompt, image_key)
            if i is not None:
                destination = parked.take(i) if cp is None else parked.entries[i].prefix(cp)
        leaving = engine.detach()
        if parked is not None and parked.budget > 0 and leaving.tokens and not engine.one_shot:
            parked.park(leaving)
        if destination is not None:
            engine.attach(destination)
    if parked is None or parked.budget <= 0:
        n = engine.resume(prompt)
        engine.image_key = image_key
        return n
    here, here_cp = reach(engine.tokens, engine.checkpoints, prompt)
    i, there, there_cp = parked.best(prompt, image_key)
    if there > here:
        t0 = time.perf_counter()
        # a prompt that continues the parked conversation takes it; one that only starts like it takes a copy
        # of that start, and the conversation stays parked for its own next turn
        conv = parked.take(i) if there_cp is None else parked.entries[i].prefix(there_cp)
        leaving = engine.detach()
        if leaving.tokens and not engine.one_shot:
            parked.park(leaving)
        engine.attach(conv)
        parked.log(f"restored {there} tokens in {(time.perf_counter() - t0) * 1e3:.1f} ms; {parked._counts()}")
    elif _leaves(engine, here, here_cp):
        leaving = engine.detach()
        if here_cp is not None:
            engine.attach(leaving.prefix(here_cp))
        parked.park(leaving)
    engine.one_shot = False
    n = engine.resume(prompt)
    engine.image_key = image_key
    return n


def _leaves(engine, here: int, cp) -> bool:
    """Whether the prompt leaves the conversation held in a way worth parking it for: it parts from it before
    its last turn began. A prompt that parts inside the last turn is that turn asked again or rendered anew
    (an answer without its thinking), and what it cuts off is not come back to. Nor is a one-shot request's
    conversation, nor a question that branched from a pinned prefix."""
    tokens = engine.tokens
    if here >= len(tokens) or engine.one_shot or (cp is not None and cp.pinned):
        return False
    last_turn = next((j for j in reversed(range(len(tokens))) if tokens[j] == engine.turn_token), 0)
    return here < last_turn
