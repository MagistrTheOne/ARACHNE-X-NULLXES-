"""Unit tests for avatar frame budget helpers."""

import numpy as np

from arachne_x.inference_frames import (
    duration_frames,
    max_sync_frames,
    normalize_ai2v_video_output,
    resolve_num_frames,
    round_to_4n_plus_1,
    suggest_embedding_fps,
)


def test_round_to_4n_plus_1():
    assert round_to_4n_plus_1(17) == 17
    assert round_to_4n_plus_1(18) == 17
    assert round_to_4n_plus_1(21) == 21


def test_max_sync_frames_elena_like():
    # ~6.24s @ 64 fps -> T_emb=399 -> sync 97
    sync = max_sync_frames(6.24, 64.0)
    assert sync == 97


def test_duration_frames_6s():
    assert duration_frames(6.24, 30) == 185


def test_resolve_sync_mode():
    n, info = resolve_num_frames("sync", 6.24, 64.0)
    assert n == 97
    assert info["mode"] == "sync"


def test_suggest_embedding_fps_boost():
    fps = suggest_embedding_fps(6.24, 185, base_fps=64.0)
    assert fps > 64.0
    assert fps <= 128.0


def test_normalize_ai2v_video_output_batch_dim():
    batched = np.zeros((1, 5, 8, 8, 3), dtype=np.uint8)
    out = normalize_ai2v_video_output(batched)
    assert out.shape == (5, 8, 8, 3)


def test_normalize_ai2v_video_output_no_batch():
    video = np.zeros((5, 8, 8, 3), dtype=np.uint8)
    out = normalize_ai2v_video_output(video)
    assert out.shape == (5, 8, 8, 3)


def test_normalize_ai2v_video_output_both_tuple():
    video = np.zeros((1, 3, 4, 4, 3), dtype=np.uint8)
    latents = np.zeros((1,))
    out = normalize_ai2v_video_output((video, latents))
    assert out.shape == (3, 4, 4, 3)
