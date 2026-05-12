"""Decode mp4 into JPEG base64 frames for WebSocket avatar.stream.chunk (line B)."""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from typing import Any, Tuple

logger = logging.getLogger(__name__)

_CACHE: dict[str, tuple[float, list[str], float]] = {}


def clear_frame_cache() -> None:
    """Test hook: drop decoded frames (e.g. after env/path changes)."""
    _CACHE.clear()


def _max_width() -> int:
    try:
        return max(64, min(1920, int(os.environ.get("NULLXES_WS_AVATAR_VIDEO_MAX_WIDTH", "480"))))
    except ValueError:
        return 480


def _max_frames() -> int:
    try:
        return max(1, min(300, int(os.environ.get("NULLXES_WS_AVATAR_VIDEO_MAX_FRAMES", "120"))))
    except ValueError:
        return 120


def _jpeg_quality() -> int:
    try:
        return max(30, min(95, int(os.environ.get("NULLXES_WS_AVATAR_VIDEO_JPEG_QUALITY", "80"))))
    except ValueError:
        return 80


def jpeg_frames_from_video_capture(cap: Any) -> Tuple[list[str], float]:
    """
    Read frames from an OpenCV VideoCapture until EOF. No caching.
    """
    import cv2

    if not cap.isOpened():
        return [], 30.0

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1.0 or fps > 120.0:
        fps = 30.0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_n = _max_frames()
    step = max(1, total // max_n) if total > max_n else 1

    max_w = _max_width()
    q = _jpeg_quality()
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), q]

    frames_b64: list[str] = []
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % step != 0:
            idx += 1
            continue
        idx += 1
        h, w = bgr.shape[:2]
        if w > max_w:
            scale = max_w / float(w)
            nw = max_w
            nh = max(1, int(round(h * scale)))
            bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        ok_enc, buf = cv2.imencode(".jpg", bgr, encode_params)
        if not ok_enc or buf is None:
            continue
        frames_b64.append(base64.b64encode(buf.tobytes()).decode("ascii"))
        if len(frames_b64) >= max_n:
            break

    if not frames_b64:
        return [], fps
    return frames_b64, fps


def load_jpeg_frames_from_mp4(path: str) -> Tuple[list[str], float]:
    """
    Decode mp4 from disk into JPEG base64 strings. Cached by resolved path + mtime.

    Returns ([], 30.0) on missing file, OpenCV failure, or zero frames.
    """
    import cv2

    try:
        abs_path = os.path.abspath(path)
        mtime = os.path.getmtime(abs_path)
    except OSError as e:
        logger.debug("avatar ws video: cannot stat %s: %s", path, e)
        return [], 30.0

    hit = _CACHE.get(abs_path)
    if hit is not None and hit[0] == mtime:
        return list(hit[1]), hit[2]

    cap = cv2.VideoCapture(abs_path)
    if not cap.isOpened():
        logger.warning("avatar ws video: OpenCV cannot open %s", abs_path)
        return [], 30.0
    try:
        frames_b64, fps = jpeg_frames_from_video_capture(cap)
    finally:
        cap.release()

    if not frames_b64:
        logger.warning("avatar ws video: zero frames from %s", abs_path)
        return [], fps

    _CACHE[abs_path] = (mtime, list(frames_b64), fps)
    return frames_b64, fps


def load_jpeg_frames_from_mp4_bytes(data: bytes) -> Tuple[list[str], float]:
    """
    Decode in-memory MP4 (e.g. NULLXES Inference Worker MP4 bytes). Not cached.
    """
    import cv2

    if not data:
        return [], 30.0
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                logger.warning("avatar ws: OpenCV cannot open temp mp4 (%d bytes)", len(data))
                return [], 30.0
            try:
                return jpeg_frames_from_video_capture(cap)
            finally:
                cap.release()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception:
        logger.exception("avatar ws: failed to decode mp4 bytes")
        return [], 30.0
