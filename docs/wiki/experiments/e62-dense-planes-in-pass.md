---
id: e62
script: bench/experiments/e62_dense_planes_in_pass.py
change: lqvqmrmr
date: 2026-10-21
---
# e62: the dense tensors at the file's bits, in a whole pass

> 中文摘要:Q3_K 用平面内核、IQ4_XS 用 e36 的块内核,放进整次前向和 8 位数组比(同一份权重两种存法,各一个引擎和对照组,
> 单次前向随机交替)。1 个 token 快 1.3 ms(7%),2 个 token 快 0.9 ms,但 4 个 token 慢 0.6 ms、8 个 token 慢 1.9 ms。
> 默认带草稿头时每次前向多是 3-5 个 token,所以整个模型装得下的机器仍用 8 位数组;从 SSD 读专家的机器本来就用文件块,
> 现在 Q3_K 换成平面内核(e61 里每种窗口都比原来的块内核快)。一致率:平面内核 378/384,和文件块相同(e60)。

## Question

Does holding the dense Q3_K and IQ4_XS tensors at the file's bits ([[dense-blocks]], Q3_K as planes [e61]) make
a whole pass faster than their 8-bit arrays, at the windows the draft head uses?

## Setup

M4 Max 128 GB, Q2_0 whole in memory, change lqvqmrmr. The model loaded once with each of the 146 such layers held
both ways and a class-wide switch for which is called; four engines traced with the switch their way (8-bit, the
file's bits, and a control of each), the switch set again before each pass for the parts not compiled. Single
passes of 1 / 2 / 4 / 8 tokens after a 300-token conversation, shuffled, 150 rounds, lower quartile. The GPU 75-95%
busy with other programs at the start. Q3_K's threads to a row set by its shape (more where rows are few, the
same for every window). Results `bench/results/e62_dense_planes_in_pass.json`; agreement with llama.cpp
`bench/results/e60/agreement_planes.txt`.

## Result

| setting | 1 token | 2 tokens | 4 tokens | 8 tokens |
| :--- | ---: | ---: | ---: | ---: |
| 8-bit arrays (the first engine made) | 25.34 | 27.80 | 35.30 | 49.46 |
| 8-bit arrays, control | 16.98 | 19.37 | 26.81 | 41.35 |
| file's bits | 15.75 | 18.63 | 27.44 | 43.05 |
| file's bits, control | 15.69 | 18.42 | 27.40 | 43.57 |

The first engine made was 5-8 ms slower than its control in this run and the two before it, whichever its
setting; not explained, and left out. Earlier runs before the layers kept their float16 scales (the switch is
not a module, so the loader's set_dtype missed them) and before the threads by shape were not comparisons.

Against llama.cpp after 1,537 / 3,161 / 6,945 tokens, the engine with Q3_K as planes (1 a pass / 4 a pass):
61 / 64, 63 / 64, 63 / 63, 378 of 384, as the file's blocks through e36's kernels (e60).

## Conclusion

- **At the file's bits a pass is 1.3 ms shorter at one token (7%) and 0.9 at two, 0.6 longer at four and 1.9 at
  eight.** The kernels are bound by decoding; from four tokens MLX's 8-bit product, which reads twice the bytes,
  does better.
- **Where the model is held whole the dense tensors stay 8-bit arrays**: with the draft head a pass is mostly of
  three to five tokens. Where experts are read from the SSD they were blocks already; their Q3_K now goes through
  the planes, faster than e36's kernel at every window (e61), 1.7 GB less than arrays as before.
- A window's logits are not the bits of the same tokens one by one in the real model either way (largest
  difference 0.19 as arrays, 0.46 as the file's bits): what agreement there is with llama.cpp is the measure.

## Concepts

[[dense-blocks]]
