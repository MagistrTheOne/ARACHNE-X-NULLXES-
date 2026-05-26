"""CPU tests for audio motion gate."""

from __future__ import annotations

import torch

from arachne_x.runtime.audio_motion_gate import apply_audio_motion_gate


def test_silence_reduces_scale():
    silent = torch.zeros(1, 16, 32)
    eff, meta = apply_audio_motion_gate(silent, 5.0)
    assert eff < 5.0
    assert meta["silence_ratio"] > 0.5


def test_active_audio_keeps_scale():
    active = torch.randn(1, 16, 32) * 0.5
    eff, meta = apply_audio_motion_gate(active, 5.0)
    assert eff == 5.0
    assert meta["silence_ratio"] == 0.0
