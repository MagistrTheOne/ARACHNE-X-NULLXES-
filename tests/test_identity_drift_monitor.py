"""CPU tests for identity drift monitor."""

from __future__ import annotations

import numpy as np

from arachne_x.runtime.identity_drift_monitor import IdentityDriftMonitor


def test_drift_policy_triggers_on_low_cosine():
    mon = IdentityDriftMonitor(warn_threshold=0.99, critical_threshold=0.95)
    pol = mon.policy_for_next_chunk(0.80)
    assert pol["refresh_identity_tokens"] is True
    assert pol["audio_guidance_scale_multiplier"] < 1.0
    assert mon.corrective_actions


def test_score_chunk_records_cosine():
    mon = IdentityDriftMonitor()
    a = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
    mon.set_anchor_from_frame(a)
    b = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
    cos = mon.score_chunk_tail(np.stack([a, b]))
    assert -0.01 <= cos <= 1.01
    assert len(mon.per_chunk_cosine) == 1
