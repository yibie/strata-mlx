---
id: image-understanding
kind: mechanism
area: prompts
status: current
---
# Image understanding

> 中文摘要:图片由 Strata 自带的独立编码器和服务器处理,引擎只接收编码器输出的行,放到提示里图片 token 的位置,
> 注意力、indexer 和草稿头按 (t, h, w) 三轴旋转这些位置。带图的对话只被"同样的图在同样位置"的提示接回。
> 可选,`--vision`;不开时什么都不变。证据来自作者的 20 核 M5 Pro(PR 13)。

## What it is

Strata's standalone encoder turns a picture into rows of the model's width and its server writes them to the
engine as SVE1 records (`GENI`). The engine puts each row at its image token of the prompt and rotates the
positions of the image's cells by (t, h, w), as Strata's kernels do, in the attention, the indexer and the draft
head; the text before and after keeps its positions. Everything else of a pass is unchanged.

## In this engine

`strata_mlx/vision.py:Vision` (the records read and checked, the rows at the image tokens, a key of the images),
`strata_mlx/vision.py:ImageRoPE` (the positions), `strata_mlx/engine.py:Engine` (`embed`, `rope`),
`strata_mlx/conversations.py:resume` (the key); the server's `--vision`.

## Relations

- depends-on → [[conversation-parking]]: a conversation with images is taken back only by a prompt with the same images at the same places; with room for one conversation the destination is taken before the departing one is parked [e63] [#13]
- depends-on → [[attention-cache]]: past 2,048 cells the indexer's pooled keys carry the image positions too; run once, with Q2_0, on the author's M5 Pro [e64]

## What we know

- The layout of the records and the positions follow Strata (revision 296b59d, MIT; `THIRD_PARTY_NOTICES.md`).
  Tests hold the positions to the scalar formula, the draft head's inputs to their shift, the window to a token
  at a time, a changed image to no reuse, a bad record to the state as it was [#13].
- On this branch with Q2_0, the author's 20-core M5 Pro, 20 GB of experts in memory: 17 of 17 synthetic requests
  right, two 1024 x 768 images in a prompt of 4,442 tokens and a follow-up of 4,482 with 4,455 reused [e64].
- Decoding with an image in the context 42-62 tok/s against 63-67 without, a duplicate control 57-66, two
  shuffled triplets of 88 tokens: what an image costs a pass is not told apart from the noise and the longer
  prompt [e64].
- The author's first run was on an integration branch with an unsupported abliterated IQ3_XXS file, 12 of 12
  right, prompts under 846 tokens: history, not evidence for this branch [e63].
- Saving a conversation with images to disk (`SAVE`) is refused; parking in memory works.

## Open questions

- What `ImageRoPE` costs a pass: it works the positions out again in every attention layer of every pass. A
  comparison with the same prompt length and a control would say.
- Pictures with the IQ3_S file, prompts past 8,192 tokens, several images in a long conversation: not run.
- Only the author's machine has run pictures; nothing here has.
