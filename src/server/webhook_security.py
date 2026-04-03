"""HMAC verification for inbound webhooks (X-NULLXES-* headers)."""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

TIMESTAMP_HEADER = "X-NULLXES-Timestamp"
SIGNATURE_HEADER = "X-NULLXES-Signature"
MAX_SKEW_SEC = 300


def _timing_safe_equal(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def verify_webhook(
    secret: Optional[str],
    raw_body: bytes,
    timestamp_hdr: Optional[str],
    signature_hdr: Optional[str],
    *,
    now: Optional[float] = None,
) -> tuple[bool, str]:
    """
    Returns (ok, reason). If secret is None/empty, verification is skipped (dev only).
    Signature format: ``v1=<hex lowercase>`` over ``f"{ts}.".encode() + raw_body``.
    """
    if not secret:
        logger.warning("Webhook HMAC skipped: NULLXES_WEBHOOK_SECRET unset")
        return True, "no_secret_dev_mode"

    if not timestamp_hdr or not signature_hdr:
        return False, "missing_headers"

    try:
        ts = int(timestamp_hdr.strip())
    except ValueError:
        return False, "bad_timestamp"

    t = time.time() if now is None else now
    if abs(t - ts) > MAX_SKEW_SEC:
        return False, "timestamp_out_of_window"

    sig = signature_hdr.strip()
    if not sig.startswith("v1="):
        return False, "bad_signature_prefix"
    expected_hex = sig[3:].strip().lower()

    payload = str(ts).encode("ascii") + b"." + raw_body
    mac = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    if not _timing_safe_equal(mac, expected_hex):
        return False, "signature_mismatch"

    return True, "ok"


def sign_webhook(secret: str, raw_body: bytes, ts: Optional[int] = None) -> tuple[str, str]:
    """Test helper: return (timestamp_str, signature_header_value)."""
    t = int(time.time() if ts is None else ts)
    payload = str(t).encode("ascii") + b"." + raw_body
    mac = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return str(t), f"v1={mac}"
