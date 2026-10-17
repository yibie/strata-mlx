"""CPU-only validation of official SVE1 framing and physical-to-rotary positions."""
import struct
import numpy as np
import pytest
from strata_mlx.vision import IMAGE_PAD, Image, Vision


def record(nx=2, ny=2, width=3, value=1):
    return struct.pack("<5i", 0x31455653, nx*ny, nx, ny, width) + np.full((nx*ny, width), value, dtype="<f4").tobytes()


def test_records_and_multi_image_positions(tmp_path):
    path = tmp_path / "image.sve"
    path.write_bytes(record() + record(nx=3, ny=1))
    prompt = [1] + [IMAGE_PAD]*4 + [2, 3] + [IMAGE_PAD]*3 + [4]
    vision = Vision.load(path, prompt, 3)
    assert vision.positions([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11]).tolist() == [
        [0,0,0], [1,1,1], [1,1,2], [1,2,1], [1,2,2], [3,3,3],
        [5,5,5], [5,5,6], [5,5,7], [8,8,8], [9,9,9]]


@pytest.mark.parametrize("raw,prompt,width", [
    (b"", [1], 3), (b"SVE1", [1], 3),
    (record()[:-1], [IMAGE_PAD]*4+[1], 3),
    (record(width=4), [IMAGE_PAD]*4+[1], 3),
    (record(), [IMAGE_PAD]*3+[1], 3),
    (record(), [IMAGE_PAD]*4, 3),
    (record(), [IMAGE_PAD]*5+[1], 3),
    (record(value=float("nan")), [IMAGE_PAD]*4+[1], 3),
    (struct.pack("<5i", 0x31455653, 2**30, 2**30, 1, 3), [IMAGE_PAD,1], 3),
])
def test_rejects_bad_records(tmp_path, raw, prompt, width):
    path = tmp_path / "bad.sve"
    path.write_bytes(raw)
    with pytest.raises(ValueError): Vision.load(path, prompt, width)


def test_identity_covers_rows_grid_and_location():
    rows = np.ones((4,3), dtype="<f4")
    a = Vision([Image(1,2,2,rows)])
    for im in [Image(2,2,2,rows), Image(1,4,1,rows), Image(1,2,2,rows*2)]:
        assert a.key != Vision([im]).key
