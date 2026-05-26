"""CPU-safe tests for chunk stitch utilities."""

from __future__ import annotations

import numpy as np

from arachne_x.runtime.chunk_stitch import (
    cosine_blend_weights,
    iter_chunk_frame_ranges,
    normalize_chunk_frames,
    stitch_chunk_videos,
)


def test_normalize_chunk_frames_4n_plus_1():
    assert normalize_chunk_frames(32) == 29  # floor to nearest 4n+1
    assert normalize_chunk_frames(33) == 33
    assert normalize_chunk_frames(34) == 33


def test_iter_chunk_ranges_cover_total():
    ranges = list(iter_chunk_frame_ranges(65, chunk_frames=33, overlap=8))
    assert ranges[0] == (0, 33, 33)
    assert ranges[-1][1] == 65
    covered = set()
    for start, end, _n in ranges:
        covered.update(range(start, end))
    assert covered == set(range(65))


def test_cosine_blend_weights_shape():
    w = cosine_blend_weights(8)
    assert w.shape == (8,)
    assert 0.0 <= w[0] < w[-1] <= 1.0


def test_stitch_single_chunk():
    c = np.zeros((10, 4, 4, 3), dtype=np.uint8)
    out = stitch_chunk_videos([c], overlap=0)
    assert out.shape == c.shape


def test_stitch_overlap_reduces_seam_jump():
    a = np.zeros((20, 8, 8, 3), dtype=np.uint8)
    a[:, :, :, 0] = 0
    b = np.zeros((20, 8, 8, 3), dtype=np.uint8)
    b[:, :, :, 0] = 255
    out = stitch_chunk_videos([a, b], overlap=8)
    assert out.shape[0] == 20 + 20 - 8
    mid = out[18:22, 0, 0, 0].astype(np.float32).mean()
    assert 0 < mid < 255
