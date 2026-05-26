"""CPU-safe tests for sampling profile merge."""

from __future__ import annotations

import argparse

import pytest

from arachne_x.runtime.sampling_profiles import (  # noqa: PLC0415 — avoid runtime.__init__ torch pull
    apply_sampling_profile,
    cap_num_frames_for_profile,
    get_profile,
    resolve_use_distill,
)


def test_operational_profile_defaults():
    p = get_profile("operational")
    assert p.num_inference_steps == 12
    assert p.use_distill is True
    assert p.chunk_frames == 33
    assert p.chunk_overlap == 8
    assert p.use_chunked_denoise is True
    assert p.num_frames_cap == 65


def test_apply_profile_respects_explicit_cli():
    args = argparse.Namespace(
        runtime_profile="operational",
        num_inference_steps=25,
        use_distill=False,
        chunk_frames=33,
        chunk_overlap=8,
        use_chunked_denoise=True,
        num_frames_mode="explicit",
        resolution="480p",
        text_guidance_scale=4.0,
        audio_guidance_scale=4.0,
    )
    apply_sampling_profile(args, argv=["--num-inference-steps", "25"])
    assert args.num_inference_steps == 25
    assert args.use_distill is True  # profile applied (no --use-distill in argv)
    assert args._sampling_profile_name == "operational"


def test_cap_num_frames():
    args = argparse.Namespace()
    args._sampling_profile_num_frames_cap = 65
    assert cap_num_frames_for_profile(args, 113) == 65
    assert cap_num_frames_for_profile(args, 33) == 33


def test_resolve_use_distill_steps():
    args = argparse.Namespace(use_distill=False, num_inference_steps=12)
    assert resolve_use_distill(args) is True
    args.num_inference_steps = 35
    assert resolve_use_distill(args) is False


def test_unknown_profile_raises():
    with pytest.raises(ValueError, match="Unknown runtime_profile"):
        get_profile("invalid")
