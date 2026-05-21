"""Unit tests for experimental audio-conditioned I2V adapter."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

pytest.importorskip("triton")

from arachne_x.modules.audio_conditioning.adapter import (
    AudioConditioningAdapter,
    AudioConditioningAdapterConfig,
)
from arachne_x.modules.audio_conditioning.state_dict import (
    load_audio_conditioning_adapter,
    save_audio_conditioning_adapter,
)
from arachne_x.modules.audio_conditioning.wrapped_dit import AudioConditionedVideoDiTWrapper


def test_adapter_config_roundtrip():
    cfg = AudioConditioningAdapterConfig(block_indices=(24, 26, 28))
    restored = AudioConditioningAdapterConfig.from_dict(cfg.to_dict())
    assert restored.block_indices == (24, 26, 28)


def test_project_audio_embs_shape():
    adapter = AudioConditioningAdapter(
        AudioConditioningAdapterConfig(block_indices=(), context_tokens=32)
    )
    b, t, w, s, c = 1, 13, 5, 12, 768
    audio = torch.randn(b, t, w, s, c)
    projected = adapter.project_audio_embs(audio)
    assert projected.shape == (b * t, 32, 768)


def test_inject_block_scale_zero_is_identity():
    adapter = AudioConditioningAdapter(AudioConditioningAdapterConfig(block_indices=(24,)))
    hidden = torch.randn(1, 13 * 64, 4096)
    audio_tokens = torch.randn(13, 32, 768)
    out = adapter.inject_block(24, hidden, audio_tokens, (13, 8, 8), 1, scale=0.0)
    assert torch.allclose(out, hidden)


def test_inject_block_missing_index_is_identity():
    adapter = AudioConditioningAdapter(AudioConditioningAdapterConfig(block_indices=(24,)))
    hidden = torch.randn(1, 13 * 64, 4096)
    audio_tokens = torch.randn(13, 32, 768)
    out = adapter.inject_block(25, hidden, audio_tokens, (13, 8, 8), 1, scale=1.0)
    assert torch.allclose(out, hidden)


def test_adapter_is_noop_by_default():
    adapter = AudioConditioningAdapter(AudioConditioningAdapterConfig(block_indices=(24, 26)))
    assert adapter.is_noop()
    assert not adapter.has_active_injection()


def test_inject_block_scale_one_runs():
    adapter = AudioConditioningAdapter(AudioConditioningAdapterConfig(block_indices=(24,)))
    hidden = torch.randn(1, 13 * 64, 4096)
    audio_tokens = torch.randn(13, 32, 768)
    out = adapter.inject_block(24, hidden, audio_tokens, (13, 8, 8), 1, scale=1.0)
    assert out.shape == hidden.shape
    assert torch.allclose(out, hidden)


def test_wrapped_dit_delegates_when_adapter_noop():
    class _FakeBase(torch.nn.Module):
        def forward(self, hidden_states, **kwargs):
            return hidden_states + 2.0

    base = _FakeBase()
    adapter = AudioConditioningAdapter(AudioConditioningAdapterConfig(block_indices=(24,)))
    wrapper = AudioConditionedVideoDiTWrapper(base, adapter)
    x = torch.randn(1, 16, 13, 8, 8)
    out = wrapper(
        hidden_states=x,
        timestep=torch.zeros(1),
        encoder_hidden_states=torch.zeros(1, 1, 4, 4096),
        audio_embs=torch.randn(1, 13, 5, 12, 768),
        audio_conditioning_scale=1.0,
    )
    assert torch.allclose(out, x + 2.0)


def test_save_load_roundtrip():
    adapter = AudioConditioningAdapter(AudioConditioningAdapterConfig(block_indices=(24, 26)))
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "adapter.safetensors")
        save_audio_conditioning_adapter(adapter, path, metadata={"test": True})
        loaded = load_audio_conditioning_adapter(path, device="cpu", strict=True)
        assert loaded.block_indices == adapter.block_indices
        for key in adapter.state_dict():
            assert torch.allclose(adapter.state_dict()[key], loaded.state_dict()[key])


def test_wrapped_dit_delegates_when_scale_zero():
    class _FakeBase(torch.nn.Module):
        def forward(self, hidden_states, **kwargs):
            return hidden_states + 1.0

    base = _FakeBase()
    adapter = AudioConditioningAdapter(AudioConditioningAdapterConfig(block_indices=()))
    wrapper = AudioConditionedVideoDiTWrapper(base, adapter)
    x = torch.randn(1, 16, 13, 8, 8)
    out = wrapper(
        hidden_states=x,
        timestep=torch.zeros(1),
        encoder_hidden_states=torch.zeros(1, 1, 4, 4096),
        audio_embs=torch.randn(1, 13, 5, 12, 768),
        audio_conditioning_scale=0.0,
    )
    assert torch.allclose(out, x + 1.0)


def test_gate_parameters_start_at_zero():
    adapter = AudioConditioningAdapter(AudioConditioningAdapterConfig(block_indices=(24, 26)))
    for block in adapter.blocks.values():
        assert float(block.gate.detach().cpu()) == 0.0
