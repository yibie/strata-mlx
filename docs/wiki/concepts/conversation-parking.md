---
id: conversation-parking
kind: mechanism
area: conversations
status: current
---
# Parked conversations

> 中文摘要:`--conversation-cache-mib`:一段对话被另一个请求打断时,把它整体停放在内存里,回来时直接取回,不必重读。
> 一段 1 万 token 的对话下一轮出第一个字从 19.7 秒降到 0.26 秒。默认关闭,预算从机器内存里先扣掉。

## What it is

The engine holds one conversation; a request that shares little with it (a client's title request, another chat)
cut it back. Parked, a conversation a prompt leaves before its last turn began is kept whole, its arrays changing
hands; a prompt that continues it takes it back.

## In this engine

`strata_mlx/conversations.py:Parked`, `strata_mlx/conversations.py:resume`; the server's
`--conversation-cache-mib` (with `-slots` and `-min-tokens`). Off by default.

## Relations

- depends-on → [[checkpoints]]: a parked conversation keeps the root, a pinned one and the last two of its checkpoints [yumvopms]
- costs → [[memory-rule]]: its budget comes out of the machine's memory before the experts' share is worked out; a conversation of 10,038 tokens was 732 MiB [yumvopms]
- depends-on → [[exactness]]: in float32 the same answers with and without parking; in float16 three of nine requests parted, a turn that goes on from a state against one that reads the conversation again [yumvopms]

- depends-on → [[image-understanding]]: image-bearing state is reusable only with matching positions, grids and embedding hashes [ROADMAP: Experimental image receiver]

## What we know

| request | without | 4,096 MiB of parking |
| :--- | ---: | ---: |
| M4 Max, Q2_0: a chat's second turn after a title request | 19.7 s | 0.26 s |
| the same chat's third turn after another chat | 15.3 s | 0.25 s |
| a 64 GB M5 Pro, IQ3_S, after an unrelated request [#8] | 23.4 s | 0.68 s |

- Parking and taking back are 0.3 ms; 8,192 MiB was no better than 4,096 on the M5 Pro [#8].

## Open questions

- Strata's floor of free memory below which nothing is parked, which wants the reading of free memory of
  [[available-memory]]. Parked conversations on the SSD, to outlast a restart.

- Synthetic full-model image/API validation and one-entry image-cache eviction regression are recorded separately from upstream-only tiny-model tests. [e55]
