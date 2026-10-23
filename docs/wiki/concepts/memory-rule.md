---
id: memory-rule
kind: mechanism
area: memory
status: current
---
# The memory rule

> 中文摘要:引擎启动时按机器的内存总量算出能给专家留多少:总量减去留给 macOS 和其他程序的、进程自己的工作内存、
> 稠密权重和草稿头,剩下的给专家。装得下全部专家就整个放进内存,否则一部分从 SSD 读。`--check` 用同一套算术,不加载模型。

## What it is

The arithmetic that decides, from a machine's memory, whether the model is held whole or its experts are read
from the SSD in part, and how many stay in memory: the machine's memory, less what is left to macOS and other
programs ([[system-reserve]]), less this process's [[working-memory]], the dense weights and the draft head; the
experts get the rest, 0.5 GB at the least.

## In this engine

`strata_mlx/paged.py:all_fit` (held whole or not), `strata_mlx/paged.py:fit` (the experts' budget),
`strata_mlx/paged.py:LEAST`, `strata_mlx/paged.py:DRAFT_HEAD`, `strata_mlx/paged.py:DENSE_AS_ARRAYS`;
`strata_mlx/check.py:plan` works out the same without loading anything (`--check`). `--expert-memory GB` sets the
budget by hand.

## Relations

- depends-on → [[system-reserve]]: the part that has moved most, 4.5 GB, then 8.5, then a quarter of the memory [zqxnwnnk] [xukwosuo]
- depends-on → [[working-memory]]: 4 GB, 5 since prompts are read a layer at a time [utpukvqp]
- depends-on → [[dense-blocks]]: where experts are read from the SSD two types of dense tensor stay the file's blocks, and the experts have 1.74 GB more [e36]
- limited-by → [[simulated-machine]]: every figure for a small machine but three was measured on a 128 GB one with the budget capped [ROADMAP: Macs with 16, 24 and 32 GB] [#6]

## What we know

By the rule now (`--check`, Q2_0 / IQ3_S, share of the experts in memory) [xukwosuo]:

| memory | Q2_0 | IQ3_S |
| :--- | :--- | :--- |
| 16 GB | too small | too small |
| 24 GB | 19% | 8% |
| 32 GB | 44% | 25% |
| 48 GB | 82% | 51% |
| 64 GB | whole | 76% |
| 96 GB and more | whole | whole |

- A real 32 GB Mac peaked at the experts' budget plus 8.4-8.6 GB of arrays in every run, as the rule counts; what
  it got wrong was what macOS held [#6].
- The rule does not count the conversation itself: 30 KB a token, about 1 GB for 36,000 tokens.
- Parked conversations ([[conversation-parking]]) are taken first, 4,096 MiB by default where the model and its
  floats are still held whole with that much less and nothing elsewhere: on 64 GB Q2_0 gets it (2.1 GB to spare
  after it) and IQ3_S, paged, does not [xwlsutut] [#12].

## Superseded

- "A 48 GB Mac holds Q2_0 whole", then "keeps 95% of it": a 48 GB M4 Pro swapped with that, and it keeps 82% now [xukwosuo].
- "A 64 GB Mac keeps 93% of IQ3_S's experts": 76% since a quarter is left to macOS, though the 64 GB M5 Pro did
  not swap at 93% [#8] [xukwosuo].

## Open questions

- A 16 GB Mac is cut to the least on no 16 GB machine's evidence.
- Whether the rule should count the conversation's caches.
