"""Virtual audio slots: PulseAudio null-sinks (Linux) + no-op stub (dev)."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


class AudioSink(Protocol):
    """Write PCM frames (float32 mono) to session output."""

    async def write_chunk(self, pcm_float32_mono: bytes) -> None: ...


class AudioSource(Protocol):
    """Read PCM from candidate / ingress (MVP: stub)."""

    async def read_chunk(self, max_samples: int) -> bytes: ...


@dataclass
class SlotRuntimeInfo:
    slot: int
    sink_name: str
    monitor_name: str
    pulse_module_id: Optional[str] = None
    present: bool = False


class MediaLayerBackend:
    """Creates N null sinks (nx_slot_0 ..) when ``pactl`` is available."""

    def __init__(self, num_slots: int = 10, sample_rate: int = 48000, channels: int = 2) -> None:
        self._num = num_slots
        self._rate = sample_rate
        self._ch = channels
        self._pactl = shutil.which("pactl")
        self._slots: List[SlotRuntimeInfo] = [
            SlotRuntimeInfo(
                slot=i,
                sink_name=f"nx_slot_{i}",
                monitor_name=f"nx_slot_{i}.monitor",
                present=False,
            )
            for i in range(num_slots)
        ]

    def _run_pactl(self, *args: str) -> Tuple[int, str, str]:
        if not self._pactl:
            return 127, "", "pactl not found"
        try:
            p = subprocess.run(
                [self._pactl, *args],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return p.returncode, p.stdout or "", p.stderr or ""
        except (subprocess.SubprocessError, OSError) as e:
            return 1, "", str(e)

    def _list_sinks(self) -> str:
        code, out, _ = self._run_pactl("list", "short", "sinks")
        return out if code == 0 else ""

    def _sink_exists(self, name: str) -> bool:
        # short format: index\tname\tdriver...
        for line in self._list_sinks().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1] == name:
                return True
        return False

    def _load_null_sink(self, slot: SlotRuntimeInfo) -> None:
        desc = f"NULLXES_slot_{slot.slot}"
        modargs = [
            "load-module",
            "module-null-sink",
            f"sink_name={slot.sink_name}",
            f"sink_properties=device.description={desc}",
            f"rate={self._rate}",
            f"channels={self._ch}",
        ]
        code, out, err = self._run_pactl(*modargs)
        if code != 0:
            logger.warning("pactl load-module failed for %s: %s %s", slot.sink_name, err, out)
            slot.present = False
            return
        # pactl prints module index as single line
        mid = out.strip().splitlines()[-1].strip() if out.strip() else None
        slot.pulse_module_id = mid
        slot.present = self._sink_exists(slot.sink_name)
        if slot.present:
            logger.info("Pulse null-sink ready: %s (module %s)", slot.sink_name, mid)

    def ensure_slots(self) -> List[SlotRuntimeInfo]:
        """
        Idempotent: load module-null-sink for each missing nx_slot_i.
        On non-Linux or without pactl, marks all slots logical-only (present=False).
        """
        if not self._pactl:
            logger.warning("pactl not found — media slots are logical names only (stub backend)")
            return list(self._slots)

        for s in self._slots:
            if self._sink_exists(s.sink_name):
                s.present = True
            else:
                self._load_null_sink(s)
        return list(self._slots)

    def snapshot(self) -> List[dict]:
        return [
            {
                "slot": s.slot,
                "sink_name": s.sink_name,
                "monitor_name": s.monitor_name,
                "pulse_sink_present": s.present,
                "pulse_module_id": s.pulse_module_id,
            }
            for s in self._slots
        ]


class StubMediaSink:
    """Discards audio; use when no device bound."""

    async def write_chunk(self, pcm_float32_mono: bytes) -> None:
        await asyncio.sleep(0)


class StubMediaSource:
    async def read_chunk(self, max_samples: int) -> bytes:
        await asyncio.sleep(0.05)
        return b"\x00" * (max_samples * 4)


def media_backend_from_env(num_slots: int) -> MediaLayerBackend:
    """``NULLXES_MEDIA_BACKEND=pulse`` (default) or ``stub``."""
    kind = os.environ.get("NULLXES_MEDIA_BACKEND", "pulse").lower().strip()
    if kind == "stub":
        return StubPulseBackend(num_slots=num_slots)
    return MediaLayerBackend(num_slots=num_slots)


class StubPulseBackend(MediaLayerBackend):
    """Logical slots without calling pactl (tests / Windows dev)."""

    def __init__(self, num_slots: int = 10) -> None:
        super().__init__(num_slots=num_slots)
        self._pactl = None

    def ensure_slots(self) -> List[SlotRuntimeInfo]:
        for s in self._slots:
            s.present = False
        return list(self._slots)
