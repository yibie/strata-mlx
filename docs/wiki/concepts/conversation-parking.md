---
id: conversation-parking
kind: mechanism
area: conversations
status: current
---
# Parked conversations

> 中文摘要:`--conversation-cache-mib`:一段对话被另一个请求打断时,把它整体停放在内存里,回来时直接取回,不必重读。
> 一段 1 万 token 的对话下一轮出第一个字从 19.7 秒降到 0.26 秒。整个模型在内存且有 4 GiB 余量的机器上默认开 4,096 MiB,
> 预算从机器内存里先扣掉;专家要从 SSD 读的机器默认关(issue 12)。短请求(默认 256 token 以下)不占槽位。

## What it is

The engine holds one conversation; a request that shares little with it (a client's title request, another chat)
cut it back. Parked, a conversation a prompt leaves before its last turn began is kept whole, its arrays changing
hands; a prompt that continues it takes it back.

## In this engine

`strata_mlx/conversations.py:Parked`, `strata_mlx/conversations.py:resume`; the server's
`--conversation-cache-mib` (with `-slots`, 4, and `-min-tokens`, 256). By default (`auto`) 4,096 MiB where the model
and its floats are still held whole with that much less, nothing where it would come out of the experts'
(`strata_mlx/check.py:parking_budget`; `--check` says which); it was off by default until [xwlsutut].

## Relations

- depends-on → [[checkpoints]]: a parked conversation keeps the root, a pinned one and the last two of its checkpoints [yumvopms]
- costs → [[memory-rule]]: its budget comes out of the machine's memory before the experts' share is worked out; a conversation of 10,038 tokens was 732 MiB [yumvopms]
- depends-on → [[exactness]]: in float32 the same answers with and without parking; in float16 three of nine requests parted, a turn that goes on from a state against one that reads the conversation again [yumvopms]
- depends-on → [[image-understanding]]: a conversation with images is taken back only by a prompt with the same images at the same places (grid and a hash of the rows) [e63] [#13]

## What we know

| request | without | 4,096 MiB of parking |
| :--- | ---: | ---: |
| M4 Max, Q2_0: a chat's second turn after a title request | 19.7 s | 0.26 s |
| the same chat's third turn after another chat | 15.3 s | 0.25 s |
| a 64 GB M5 Pro, IQ3_S, after an unrelated request [#8] | 23.4 s | 0.68 s |

- Parking and taking back are 0.3 ms; 8,192 MiB was no better than 4,096 on the M5 Pro [#8].
- Off, the engine holds one conversation and any other client's request cuts it back: on a 96 GB M2 Max with
  Q2_0 (`2f9d1ab`, four hours before parking landed) a web chat beside another client read its 38,313-token
  prompt again every turn, 121 s; one 11-token request between two turns of a 15,013-token conversation left
  the next turn 42 tokens, 14,987 read again in 43.3 s. Their prompt reads: 358 tok/s at 15,019 tokens, 314 at
  39,940 [#12].
- The slots go oldest first whatever their size: with `-min-tokens` 0, four short requests from another client
  between two turns of a 15,000-token conversation took its slot (four slots, 4,096 MiB); at 256 they take none.
  Worked out on the CPU with `Parked` alone, a test holds it [xwlsutut].

- With images, the destination is taken before the departing conversation is parked: with room for one
  conversation, parking first evicted the very one being resumed (found on the M5 Pro, a test holds it) [e63] [#13].

## Superseded

- "Off by default": on where the model and its floats are held whole with 4 GiB to spare, since a user of a 96 GB
  Mac lost a long conversation to every request of another client and found no answer short of the source [#12]
  [xwlsutut].

## Open questions

- Strata's floor of free memory below which nothing is parked, which wants the reading of free memory of
  [[available-memory]]. Parked conversations on the SSD, to outlast a restart.
- The smallest slot leaving before the oldest: four requests above 256 tokens from other clients between two
  turns still take a longer conversation's slot.
- Whether 4,096 MiB is right on the smallest machine that gets it: Q2_0 on 64 GB has 2.1 GB to spare after it by
  the rule, and one real check, the 64 GB M5 Pro that did not swap with it [#8].
