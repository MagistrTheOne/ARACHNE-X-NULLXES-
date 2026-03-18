import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from diffusers.utils import load_image

from arachne_x.loader import load_avatar_pipeline


@dataclass(frozen=True)
class AvatarFrame:
    frame: np.ndarray
    timestamp: float


class ArachneSession:
    """Long-lived avatar rendering session for one realtime dialog."""

    def __init__(
        self,
        checkpoint_dir: str,
        avatar_image: str,
        pipeline_cfg: Optional[Dict[str, Any]] = None,
        generation_cfg: Optional[Dict[str, Any]] = None,
        idle_frame_size: tuple[int, int] = (512, 512),
        render_window_ms: int = 800,
        max_frame_buffer_seconds: float = 3.0,
        sample_rate: int = 16000,
    ):
        self.pipeline_cfg = dict(pipeline_cfg or {})
        self.generation_cfg = dict(generation_cfg or {})
        self.checkpoint_dir = checkpoint_dir
        self.avatar_image_path = str(Path(avatar_image))
        self.sample_rate = int(sample_rate)
        self.render_window_samples = int(round(render_window_ms * self.sample_rate / 1000.0))
        self.frame_queue_maxsize = max(1, int(round(max_frame_buffer_seconds * 30)))
        self._audio_queue: asyncio.Queue[Optional[np.ndarray]] = asyncio.Queue(maxsize=32)
        self._audio_out_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=64)
        self._frame_queue: asyncio.Queue[AvatarFrame] = asyncio.Queue(maxsize=self.frame_queue_maxsize)
        self._render_task: Optional[asyncio.Task] = None
        self._running = False
        self._interrupt_generation = 0
        self._idle_frame = np.zeros((idle_frame_size[1], idle_frame_size[0], 3), dtype=np.uint8)
        self.last_frame = self._idle_frame
        self._pipeline = None
        self._avatar_image = None
        self._render_lock = asyncio.Lock()

    async def start(self) -> None:
        if self._running:
            return
        # Model/pipeline loading is heavy; do it off the event loop.
        await asyncio.to_thread(self._load_pipeline)
        self._running = True
        self._render_task = asyncio.create_task(self._render_loop())

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        await self._audio_queue.put(None)
        if self._render_task is not None:
            await self._render_task
        self._render_task = None

    async def push_audio_chunk(self, pcm: np.ndarray) -> None:
        chunk = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return
        await self._audio_queue.put(np.ascontiguousarray(chunk))
        await self._queue_audio_out(chunk)

    async def interrupt(self) -> None:
        self._interrupt_generation += 1
        if self._pipeline is not None:
            setattr(self._pipeline, "_interrupt", True)
        await self._drain_queue(self._audio_queue)
        await self._drain_queue(self._frame_queue)
        await self._drain_queue(self._audio_out_queue)

    async def pull_frame(self) -> Optional[np.ndarray]:
        try:
            item = self._frame_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        self.last_frame = item.frame
        return item.frame

    async def pull_audio(self) -> Optional[np.ndarray]:
        try:
            return self._audio_out_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def get_idle_frame(self) -> np.ndarray:
        return self.last_frame if self.last_frame is not None else self._idle_frame

    async def _render_loop(self) -> None:
        pending: list[np.ndarray] = []
        pending_samples = 0
        while self._running:
            try:
                chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                continue

            if chunk is None:
                break

            pending.append(chunk)
            pending_samples += chunk.size
            if pending_samples < self.render_window_samples:
                continue

            render_audio = np.concatenate(pending).astype(np.float32, copy=False)
            pending = []
            pending_samples = 0
            generation_id = self._interrupt_generation
            await self._render_audio_window(render_audio, generation_id)

    async def _render_audio_window(self, pcm: np.ndarray, generation_id: int) -> None:
        if self._pipeline is None or self._avatar_image is None:
            return

        async with self._render_lock:
            setattr(self._pipeline, "_interrupt", False)
            loop = asyncio.get_running_loop()
            started = time.perf_counter()
            frames = await loop.run_in_executor(None, self._generate_frames_sync, pcm)
            if generation_id != self._interrupt_generation:
                return
            for frame in frames:
                await self._queue_frame(frame)
            if frames:
                self.last_frame = frames[-1]
            latency_ms = (time.perf_counter() - started) * 1000.0
            metrics = getattr(self._pipeline, "metrics", None)
            if metrics is not None:
                try:
                    metrics.record("avatar_render_window_latency_ms", float(latency_ms))
                except Exception:
                    # Metrics must never break realtime generation.
                    # Keeping this at debug-level to avoid noise in production.
                    import logging
                    logging.getLogger(__name__).debug(
                        "metrics.record() failed for avatar_render_window_latency_ms", exc_info=True
                    )

    def _generate_frames_sync(self, pcm: np.ndarray) -> list[np.ndarray]:
        kwargs = dict(self.generation_cfg)
        kwargs.setdefault("prompt", "A person speaking naturally")
        kwargs.setdefault("negative_prompt", "")
        kwargs.setdefault("resolution", "480p")
        kwargs.setdefault("num_inference_steps", 8)
        kwargs.setdefault("text_guidance_scale", 4.0)
        kwargs.setdefault("audio_guidance_scale", 4.0)
        kwargs["audio_emb"] = self._pipeline.get_audio_embedding(
            pcm,
            fps=16 * max(int(getattr(self._pipeline, "vae_scale_factor_temporal", 4)), 1),
            device=self._pipeline.device,
            sample_rate=self.sample_rate,
        )
        try:
            output = self._pipeline.generate_ai2v(
                image=self._avatar_image,
                output_type="np",
                **kwargs,
            )
        except Exception as exc:
            # During realtime interruptions we abort diffusion denoising early.
            # The caller will discard the generation anyway, but returning [] avoids
            # decoding partial frames.
            if type(exc).__name__ == "GenerationInterrupted":
                return []
            raise
        if isinstance(output, torch.Tensor):
            array = output.detach().cpu().numpy()
        else:
            array = np.asarray(output)
        if array.ndim != 5:
            raise ValueError(f"Expected avatar output [B, T, H, W, C], got {tuple(array.shape)}")
        return [np.ascontiguousarray(frame) for frame in array[0]]

    def _load_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        pipe_cfg = dict(self.pipeline_cfg)
        device = pipe_cfg.pop("device", "cuda")
        torch_dtype = pipe_cfg.pop("torch_dtype", torch.bfloat16)
        variant = pipe_cfg.pop("variant", "single")
        self._pipeline = load_avatar_pipeline(
            checkpoint_dir=self.checkpoint_dir,
            variant=variant,
            device=device,
            torch_dtype=torch_dtype,
            **pipe_cfg,
        )
        self._avatar_image = load_image(self.avatar_image_path)

    async def _queue_frame(self, frame: np.ndarray) -> None:
        item = AvatarFrame(frame=np.ascontiguousarray(frame), timestamp=time.time())
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await self._frame_queue.put(item)

    async def _queue_audio_out(self, pcm: np.ndarray) -> None:
        if self._audio_out_queue.full():
            try:
                self._audio_out_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await self._audio_out_queue.put(np.ascontiguousarray(pcm))

    async def _drain_queue(self, queue_obj: asyncio.Queue) -> None:
        while True:
            try:
                queue_obj.get_nowait()
            except asyncio.QueueEmpty:
                break
