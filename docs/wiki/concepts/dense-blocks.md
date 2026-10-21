---
id: dense-blocks
kind: component
area: memory
status: current
---
# The dense tensors as the file's blocks

> 中文摘要:Q3_K 和 IQ4_XS 两种稠密张量,MLX 只能按 8 位存,比文件大 1.74 GB。专家从 SSD 读的机器上它们保持文件原来的块格式,
> 省下的内存给专家。16 GB 机器上回答快了一半,24 和 32 GB 上差别在噪声里。

## What it is

Two types of the file's dense tensors (Q3_K and IQ4_XS) MLX can hold only at 8 bits: 5.49 GB as arrays against
3.64 in the file. Where experts are read from the SSD they stay the file's blocks, multiplied by kernels that
unpack them, and the experts have the 1.74 GB.

## In this engine

`strata_mlx/dense.py:BlockLinear`, `strata_mlx/dense.py:BlockEmbedding`, `strata_mlx/blocks.py:window_matvec`.

## Relations

- costs → [[decode-pass]]: 3.8 against 4.7 ms for one token, 5.6 against 4.8 for four: slower from four tokens [e36]
- trades-against → [[expert-store]]: 1.74 GB more for experts: as a 16 GB machine 3.3 GB of experts where 1.6, answers 10.9 tok/s where 7.2 [e36]

## What we know

- In a whole pass, Q3_K as planes and IQ4_XS through e36's kernel against the 8-bit arrays: 1.3 ms shorter at
  one token, 0.9 at two, 0.6 longer at four, 1.9 at eight; so held whole the model keeps the arrays, and where
  experts are read from the SSD Q3_K goes through the planes [e62].

- Q3_K rearranged at load into aligned planes is faster than MLX's 8-bit at 1 to 8 tokens (65-109 against
  80-116 us a 10,240 x 2,560 tensor); IQ4_XS is faster through e36's kernel at 1-2 tokens and as planes at 4-8.
  All are bound by decoding, far above the file's bytes (23-28 us) [e61].

- Against llama.cpp's answers after 1,537, 3,161 and 6,945 tokens of the same Q2_0 file, the engine as blocks
  agreed at 378 of 384 top tokens, as 8-bit arrays at 374: the file's blocks cost no agreement [e60].

- As a 24 GB machine 6% and 13% faster in two runs, less than a run differs from the next; as a 32 GB machine no
  difference [e36].
- Held whole, the engine keeps the arrays.
