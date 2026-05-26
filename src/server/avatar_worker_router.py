"""
Route avatar GPU requests to a worker pool (session affinity via hash).

Env:
  NULLXES_AVATAR_WORKER_URLS — comma-separated base URLs (preferred)
  NULLXES_AVATAR_INFERENCE_URL — single-worker fallback
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

WORKER_URLS_ENV = "NULLXES_AVATAR_WORKER_URLS"
INFERENCE_URL_ENV = "NULLXES_AVATAR_INFERENCE_URL"
HEALTH_CACHE_TTL_SEC = 5.0

_health_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def worker_base_urls() -> list[str]:
    multi = (os.environ.get(WORKER_URLS_ENV) or "").strip()
    if multi:
        urls = [u.strip().rstrip("/") for u in multi.split(",") if u.strip()]
        if urls:
            return urls
    single = (os.environ.get(INFERENCE_URL_ENV) or "").strip().rstrip("/")
    return [single] if single else []


def route_worker_base_url(session_id: str) -> str:
    urls = worker_base_urls()
    if not urls:
        raise RuntimeError(
            f"Set {WORKER_URLS_ENV} or {INFERENCE_URL_ENV} for avatar GPU routing"
        )
    sid = str(session_id or "").strip() or "anonymous"
    digest = hashlib.sha256(sid.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(urls)
    return urls[idx]


def route_worker_index(session_id: str) -> tuple[int, str]:
    urls = worker_base_urls()
    if not urls:
        raise RuntimeError(
            f"Set {WORKER_URLS_ENV} or {INFERENCE_URL_ENV} for avatar GPU routing"
        )
    sid = str(session_id or "").strip() or "anonymous"
    digest = hashlib.sha256(sid.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(urls)
    return idx, urls[idx]
