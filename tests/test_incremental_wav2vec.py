"""Unit tests for incremental wav2vec sample budgeting (no GPU)."""

from __future__ import annotations

import pytest

from arachne_x.inference_audio import (
    incremental_wav2vec_enabled,
    min_audio_samples_for_frames,
    min_embedding_timesteps_for_frames,
)


def test_min_embedding_timesteps_first_chunk() -> None:
    t = min_embedding_timesteps_for_frames(9, audio_stride=4, audio_window=5)
    assert t >= 33


def test_min_audio_samples_operational_first_chunk() -> None:
    samples = min_audio_samples_for_frames(
        9,
        embedding_fps=64.0,
        sample_rate=16000,
        audio_stride=4,
        audio_window=5,
    )
    assert 4000 <= samples <= 12000


def test_incremental_enabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARACHNE_INCREMENTAL_WAV2VEC", raising=False)
    assert incremental_wav2vec_enabled() is True


def test_incremental_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARACHNE_INCREMENTAL_WAV2VEC", "0")
    assert incremental_wav2vec_enabled() is False


def test_min_ms_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARACHNE_INCREMENTAL_WAV2VEC_MIN_MS", "300")
    from arachne_x.inference_audio import _incremental_min_samples_override

    assert _incremental_min_samples_override() == 4800
