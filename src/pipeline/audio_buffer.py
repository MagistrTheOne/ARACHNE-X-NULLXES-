import asyncio
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioWindow:
    pcm: np.ndarray
    start_timestamp: float
    end_timestamp: float


class AudioRingBuffer:
    """Fixed-size float32 mono ring buffer with timestamp tracking."""

    def __init__(self, max_seconds: float = 30.0, sample_rate: int = 16000):
        if max_seconds <= 0:
            raise ValueError("`max_seconds` must be positive.")
        if sample_rate <= 0:
            raise ValueError("`sample_rate` must be positive.")

        self.sample_rate = int(sample_rate)
        self.capacity = max(1, int(round(max_seconds * sample_rate)))
        self._pcm = np.zeros(self.capacity, dtype=np.float32)
        self._head = 0
        self._size = 0
        self._start_timestamp = 0.0
        self._lock = asyncio.Lock()

    async def push(self, pcm: np.ndarray, timestamp: float) -> None:
        pcm = self._normalize_pcm(pcm)
        if pcm.size == 0:
            return

        async with self._lock:
            original_size = pcm.size
            if original_size >= self.capacity:
                drop = original_size - self.capacity
                pcm = pcm[-self.capacity :]
                timestamp = float(timestamp) + drop / self.sample_rate

            overflow = max(0, self._size + pcm.size - self.capacity)
            if overflow:
                self._drop_oldest(overflow)

            write_pos = (self._head + self._size) % self.capacity
            first = min(pcm.size, self.capacity - write_pos)
            self._pcm[write_pos : write_pos + first] = pcm[:first]
            remaining = pcm.size - first
            if remaining > 0:
                self._pcm[:remaining] = pcm[first:]

            if self._size == 0:
                self._start_timestamp = float(timestamp)

            self._size = min(self.capacity, self._size + pcm.size)

    async def read_latest(self, duration_ms: int) -> AudioWindow:
        async with self._lock:
            n = min(self._duration_to_samples(duration_ms), self._size)
            pcm = self._slice_latest(n)
            end_ts = self._start_timestamp + self._size / self.sample_rate
            start_ts = end_ts - n / self.sample_rate
            return AudioWindow(pcm=pcm, start_timestamp=start_ts, end_timestamp=end_ts)

    async def pop_window(self, duration_ms: int) -> AudioWindow:
        async with self._lock:
            n = min(self._duration_to_samples(duration_ms), self._size)
            pcm = self._slice_oldest(n)
            start_ts = self._start_timestamp
            end_ts = start_ts + n / self.sample_rate
            self._drop_oldest(n)
            return AudioWindow(pcm=pcm, start_timestamp=start_ts, end_timestamp=end_ts)

    async def available_samples(self) -> int:
        async with self._lock:
            return self._size

    async def clear(self) -> None:
        async with self._lock:
            self._head = 0
            self._size = 0
            self._start_timestamp = 0.0

    def _duration_to_samples(self, duration_ms: int) -> int:
        if duration_ms < 0:
            raise ValueError("`duration_ms` must be >= 0.")
        return int(round(duration_ms * self.sample_rate / 1000.0))

    def _slice_latest(self, n: int) -> np.ndarray:
        if n <= 0 or self._size == 0:
            return np.empty(0, dtype=np.float32)
        start = (self._head + self._size - n) % self.capacity
        return self._slice(start, n)

    def _slice_oldest(self, n: int) -> np.ndarray:
        if n <= 0 or self._size == 0:
            return np.empty(0, dtype=np.float32)
        return self._slice(self._head, n)

    def _slice(self, start: int, length: int) -> np.ndarray:
        first = min(length, self.capacity - start)
        if first == length:
            return self._pcm[start : start + length].copy()
        return np.concatenate((self._pcm[start : start + first], self._pcm[: length - first])).astype(np.float32, copy=False)

    def _drop_oldest(self, n: int) -> None:
        if n <= 0 or self._size == 0:
            return
        n = min(n, self._size)
        self._head = (self._head + n) % self.capacity
        self._start_timestamp += n / self.sample_rate
        self._size -= n

    @staticmethod
    def _normalize_pcm(pcm: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(pcm, dtype=np.float32).reshape(-1))
