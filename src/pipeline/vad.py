from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch


class SpeechState(str, Enum):
    SILENCE = "silence"
    SPEECH = "speech"
    ENDED = "ended"


@dataclass(frozen=True)
class VADResult:
    state: SpeechState
    utterance_audio: Optional[np.ndarray]
    utterance_start_ts: Optional[float]
    utterance_end_ts: Optional[float]
    speech_probability: float


class SileroVAD:
    """Endpoint detector with hysteresis around a preloaded Silero model."""

    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        speech_threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 450,
        max_utterance_ms: int = 12000,
        pre_roll_ms: int = 150,
        post_roll_ms: int = 100,
        device: str = "cpu",
    ):
        self.sample_rate = int(sample_rate)
        self.speech_threshold = float(speech_threshold)
        self.min_speech_samples = self._ms_to_samples(min_speech_ms)
        self.min_silence_samples = self._ms_to_samples(min_silence_ms)
        self.max_utterance_samples = self._ms_to_samples(max_utterance_ms)
        self.pre_roll_samples = self._ms_to_samples(pre_roll_ms)
        self.post_roll_samples = self._ms_to_samples(post_roll_ms)
        self.device = torch.device(device)
        self.model = self._load_model(model_path).to(self.device)
        self.model.eval()

        self._state = SpeechState.SILENCE
        self._pre_roll = np.empty(0, dtype=np.float32)
        self._segments: List[np.ndarray] = []
        self._utterance_start_ts: Optional[float] = None
        self._speech_samples = 0
        self._silence_samples = 0
        self._utterance_samples = 0

    @torch.no_grad()
    def process_chunk(self, pcm: np.ndarray, timestamp: float) -> VADResult:
        chunk = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return VADResult(self._state, None, None, None, 0.0)

        prob = float(self.model(torch.from_numpy(chunk).to(self.device), self.sample_rate).item())
        is_speech = prob >= self.speech_threshold
        chunk_end_ts = float(timestamp) + chunk.size / self.sample_rate

        if self._state == SpeechState.SILENCE:
            self._append_pre_roll(chunk)
            if is_speech:
                self._start_utterance(timestamp, chunk)
                return VADResult(SpeechState.SPEECH, None, None, None, prob)
            return VADResult(SpeechState.SILENCE, None, None, None, prob)

        self._segments.append(chunk)
        self._utterance_samples += chunk.size
        if is_speech:
            self._speech_samples += chunk.size
            self._silence_samples = 0
            self._state = SpeechState.SPEECH
        else:
            self._silence_samples += chunk.size

        should_end = False
        if self._speech_samples >= self.min_speech_samples:
            if self._silence_samples >= self.min_silence_samples:
                should_end = True
            elif self._utterance_samples >= self.max_utterance_samples:
                should_end = True
        elif self._utterance_samples >= self.max_utterance_samples:
            self._reset()
            return VADResult(SpeechState.SILENCE, None, None, None, prob)

        if not should_end:
            return VADResult(SpeechState.SPEECH, None, None, None, prob)

        utterance = np.concatenate(self._segments) if self._segments else np.empty(0, dtype=np.float32)
        if self.post_roll_samples and self._silence_samples > self.post_roll_samples:
            trim = self._silence_samples - self.post_roll_samples
            if trim > 0 and utterance.size > trim:
                utterance = utterance[:-trim]
                chunk_end_ts -= trim / self.sample_rate

        start_ts = self._utterance_start_ts
        self._reset()
        return VADResult(SpeechState.ENDED, utterance.astype(np.float32, copy=False), start_ts, chunk_end_ts, prob)

    def reset(self) -> None:
        self._reset()

    def _start_utterance(self, timestamp: float, first_chunk: np.ndarray) -> None:
        pre_roll = self._pre_roll.copy()
        self._segments = [pre_roll, first_chunk] if pre_roll.size else [first_chunk]
        self._utterance_start_ts = float(timestamp) - pre_roll.size / self.sample_rate
        self._state = SpeechState.SPEECH
        self._speech_samples = first_chunk.size
        self._silence_samples = 0
        self._utterance_samples = pre_roll.size + first_chunk.size
        self._pre_roll = np.empty(0, dtype=np.float32)

    def _append_pre_roll(self, chunk: np.ndarray) -> None:
        if self.pre_roll_samples <= 0:
            return
        merged = np.concatenate((self._pre_roll, chunk)) if self._pre_roll.size else chunk
        self._pre_roll = merged[-self.pre_roll_samples :].astype(np.float32, copy=False)

    def _reset(self) -> None:
        self._state = SpeechState.SILENCE
        self._segments = []
        self._utterance_start_ts = None
        self._speech_samples = 0
        self._silence_samples = 0
        self._utterance_samples = 0

    def _ms_to_samples(self, value_ms: int) -> int:
        return max(0, int(round(value_ms * self.sample_rate / 1000.0)))

    @staticmethod
    def _load_model(model_path: str) -> torch.nn.Module:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Silero VAD model not found: {model_path}")
        return torch.jit.load(str(path), map_location="cpu")
