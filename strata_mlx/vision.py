"""Receive official Strata SVE1 encoder records; no image decoder or HTTP fork.

Protocol/position layout adapted from Niko1221/strata 296b59d (MIT),
 tools/vision/strata_vision.cpp, src/program/generate.cpp and
 include/strata/kernels/mrope.hpp. See THIRD_PARTY_NOTICES.md.
"""
from dataclasses import dataclass
import hashlib
import struct

import numpy as np

IMAGE_PAD = 248056


@dataclass
class Image:
    start: int
    nx: int
    ny: int
    rows: np.ndarray


class Vision:
    def __init__(self, images):
        self.images = images
        self.key = tuple((im.start, im.nx, im.ny, hashlib.sha256(im.rows.tobytes()).digest()) for im in images)
        self.bytes = sum(im.rows.nbytes for im in images)

    @classmethod
    def load(cls, path, prompt, width):
        """Bound reads by the actual prompt, reject bad input before touching engine state."""
        images, cursor, total_bytes = [], 0, 0
        with open(path, "rb") as stream:
            while header := stream.read(20):
                if len(header) != 20:
                    raise ValueError("short SVE1 header")
                magic, n, nx, ny, d = struct.unpack("<5i", header)
                if magic != 0x31455653 or min(n, nx, ny) < 1 or nx * ny != n or d != width:
                    raise ValueError("bad SVE1 grid or embedding width")
                while cursor < len(prompt) and prompt[cursor] != IMAGE_PAD:
                    cursor += 1
                if n > len(prompt) - cursor or any(t != IMAGE_PAD for t in prompt[cursor:cursor+n]):
                    raise ValueError("image rows do not match image_pad tokens")
                total_bytes += n * d * 4
                if total_bytes > 128 << 20:
                    raise ValueError("image embeddings exceed the 128MiB receiver limit")
                raw = stream.read(n * d * 4)
                if len(raw) != n * d * 4:
                    raise ValueError("short SVE1 payload")
                rows = np.frombuffer(raw, dtype="<f4").reshape(n, d).copy()
                if not np.isfinite(rows).all():
                    raise ValueError("non-finite image embeddings")
                images.append(Image(cursor, nx, ny, rows))
                cursor += n
        if not images or IMAGE_PAD in prompt[cursor:] or prompt[-1] == IMAGE_PAD:
            raise ValueError("missing images or prompt ends in image_pad")
        return cls(images)

    def positions(self, cells):
        """Cell indices -> (t,h,w), including arbitrary decode/draft lookahead."""
        cells = np.asarray(cells)
        pos = np.repeat(cells[..., None], 3, axis=-1)
        delta = 0
        for im in self.images:
            n = len(im.rows)
            here = (cells >= im.start) & (cells < im.start + n)
            j = cells - im.start
            base = im.start + delta
            pos[..., 0] = np.where(here, base, pos[..., 0])
            pos[..., 1] = np.where(here, base + j // im.nx, pos[..., 1])
            pos[..., 2] = np.where(here, base + j % im.nx, pos[..., 2])
            shift = max(im.nx, im.ny) - n
            pos = np.where((cells >= im.start+n)[..., None], pos + shift, pos)
            delta += shift
        return pos

    def embed(self, embed_tokens, ids, start):
        import mlx.core as mx
        out = embed_tokens(ids)
        end = start + ids.shape[1]
        for im in self.images:
            a, b = max(start, im.start), min(end, im.start + len(im.rows))
            if a < b:
                out[:, a-start:b-start] = mx.array(im.rows[a-im.start:b-im.start], dtype=out.dtype)
        return out


class ImageRoPE:
    """Wrap the pinned port's rotary frequencies; causal masks still use cell indices."""
    def __init__(self, rope, vision):
        self.rope, self.vision = rope, vision

    def __call__(self, cells):
        import mlx.core as mx
        pos = mx.stack([cells, cells, cells], axis=-1)
        delta = 0
        for im in self.vision.images:
            n = len(im.rows)
            j = cells - im.start
            image = mx.stack([mx.full(cells.shape, im.start + delta),
                              im.start + delta + j // im.nx,
                              im.start + delta + j % im.nx], axis=-1)
            here = (cells >= im.start) & (cells < im.start + n)
            pos = mx.where(here[..., None], image, pos)
            shift = max(im.nx, im.ny) - n
            pos = mx.where((cells >= im.start+n)[..., None], pos + shift, pos)
            delta += shift
        pos = pos.astype(mx.float32)
        pairs = self.rope.inv_freq.shape[0]
        sectors = mx.arange(pairs) % 3
        selected = mx.take(pos, sectors, axis=-1)
        angles = selected * self.rope.inv_freq
        angles = mx.concatenate([angles, angles], axis=-1)
        return mx.cos(angles), mx.sin(angles)
