import asyncio
import fractions
from typing import Any, Dict

import av
import numpy as np
import soxr
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription

from src.pipeline.orchestrator import RealtimePipeline


class ArachneVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, session, fps: int = 30):
        super().__init__()
        self.session = session
        self.fps = int(fps)
        self._pts = 0
        self._time_base = fractions.Fraction(1, self.fps)

    async def recv(self) -> av.VideoFrame:
        await asyncio.sleep(1 / self.fps)
        frame = await self.session.pull_frame()
        if frame is None:
            frame = self.session.get_idle_frame()
        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = self._pts
        video_frame.time_base = self._time_base
        self._pts += 1
        return video_frame


class TTSAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, session, sample_rate: int = 16000, frame_ms: int = 20):
        super().__init__()
        self.session = session
        self.sample_rate = int(sample_rate)
        self.frame_samples = int(round(sample_rate * frame_ms / 1000.0))
        self._pts = 0
        self._time_base = fractions.Fraction(1, self.sample_rate)
        self._buffer = np.empty(0, dtype=np.float32)

    async def recv(self) -> av.AudioFrame:
        while self._buffer.size < self.frame_samples:
            chunk = await self.session.pull_audio()
            if chunk is None:
                break
            self._buffer = np.concatenate((self._buffer, chunk))

        if self._buffer.size >= self.frame_samples:
            pcm = self._buffer[: self.frame_samples]
            self._buffer = self._buffer[self.frame_samples :]
        else:
            pcm = np.pad(self._buffer, (0, self.frame_samples - self._buffer.size))
            self._buffer = np.empty(0, dtype=np.float32)

        pcm_i16 = np.clip(pcm * 32767.0, -32768, 32767).astype(np.int16, copy=False)
        frame = av.AudioFrame.from_ndarray(pcm_i16.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = self.sample_rate
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += pcm_i16.size
        await asyncio.sleep(self.frame_samples / self.sample_rate)
        return frame


async def _consume_audio_track(track: MediaStreamTrack, pipeline: RealtimePipeline) -> None:
    while True:
        frame = await track.recv()
        pcm = frame.to_ndarray()
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=0)
        pcm = pcm.astype(np.float32)
        if frame.format.name.startswith("s16"):
            pcm = pcm / 32768.0
        sample_rate = int(frame.sample_rate or 48000)
        if sample_rate != 16000:
            pcm = soxr.resample(pcm, sample_rate, 16000).astype(np.float32, copy=False)
        await pipeline.on_audio_chunk(pcm)


async def offer(request: web.Request) -> web.Response:
    payload = await request.json()
    app = request.app
    pipeline_cfg: Dict[str, Any] = app["pipeline_config"]
    pipeline = RealtimePipeline(pipeline_cfg)
    await pipeline.start()

    pc = RTCPeerConnection()
    app["pcs"].add(pc)
    app["pipelines"][pc] = pipeline
    pc.addTrack(ArachneVideoTrack(pipeline.session))
    pc.addTrack(TTSAudioTrack(pipeline.session))

    @pc.on("track")
    def on_track(track: MediaStreamTrack) -> None:
        if track.kind == "audio":
            asyncio.create_task(_consume_audio_track(track, pipeline))

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            await _cleanup_peer(app, pc)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=payload["sdp"], type=payload["type"]))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


async def health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def _cleanup_peer(app: web.Application, pc: RTCPeerConnection) -> None:
    pipeline = app["pipelines"].pop(pc, None)
    if pipeline is not None:
        await pipeline.stop()
    if pc in app["pcs"]:
        app["pcs"].remove(pc)
    await pc.close()


async def _on_shutdown(app: web.Application) -> None:
    peers = list(app["pcs"])
    for pc in peers:
        await _cleanup_peer(app, pc)


def create_app(pipeline_config: Dict[str, Any]) -> web.Application:
    app = web.Application()
    app["pipeline_config"] = pipeline_config
    app["pcs"] = set()
    app["pipelines"] = {}
    app.router.add_post("/offer", offer)
    app.router.add_get("/health", health)
    app.on_shutdown.append(_on_shutdown)
    return app
