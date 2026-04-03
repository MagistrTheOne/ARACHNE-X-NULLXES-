import json

from src.server.webhook_security import verify_webhook, sign_webhook


def test_sign_verify_roundtrip():
    secret = "test_secret"
    body = json.dumps({"event": "interview.session.created", "session_id": "a"}).encode("utf-8")
    ts, sig = sign_webhook(secret, body, ts=1_700_000_000)
    ok, reason = verify_webhook(secret, body, ts, sig, now=1_700_000_030)
    assert ok and reason == "ok"


def test_reject_bad_sig():
    body = b"{}"
    ok, reason = verify_webhook("s", body, "1700000000", "v1=deadbeef", now=1_700_000_000)
    assert not ok


def test_dev_mode_no_secret():
    ok, reason = verify_webhook(None, b"{}", None, None)
    assert ok and reason == "no_secret_dev_mode"
