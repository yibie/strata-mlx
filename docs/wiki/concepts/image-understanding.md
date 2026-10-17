---
id: image-understanding
kind: mechanism
area: prompts
status: current
---
# Image understanding

> 中文摘要:复用官方图像编码器,在 MLX 接收 SVE1、图像嵌入和三轴位置。尚需真实模型验证。

## What it is

The official standalone encoder writes projected rows. MLX inserts them at image_pad
cells and rotates attention/indexer/MTP with interleaved (t,h,w) positions.

## In this engine

`strata_mlx/vision.py:Vision`, `strata_mlx/vision.py:ImageRoPE`;
`strata_mlx/engine.py:Engine`, `strata_mlx/conversations.py:resume`.

## Relations

- depends-on → [[conversation-parking]]: image identities qualify both live and parked token reuse [ROADMAP: Experimental image receiver]

## What we know

- Source layout follows official Strata 296b59d's MIT SVE1/IMRoPE path. CPU framing tests cover malformed records and multiple image grids; bounded full-model results are retained in e55; broad production qualification remains open. [ROADMAP: Experimental image receiver]
- Entire image-sequence namespaces conservatively discard reuse on a changed/added image; disk SAVE of image state is refused. [ROADMAP: Experimental image receiver]

## Open questions

Broader visual accuracy, other memory profiles, long context and sustained/concurrent serving remain open.

- Synthetic full-model image/API validation and one-entry image-cache eviction regression are recorded separately from upstream-only tiny-model tests. [e55]

- Deployment requires external official HTTP/tokenizer code and standalone encoder, not the original language-model engine. Portable setup and a schema-required CPU-only preflight are documented; no weights are bundled or started by configuration. [qkwxxvrp]
