"""Minimal OpenAPI 3.0 JSON for MVP routes."""

from __future__ import annotations

SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "NULLXES Session & Media API",
        "version": "0.1.0",
        "description": "MVP webhook, session control, media slots (NULLXES / ARACHNE-X).",
    },
    "paths": {
        "/health": {"get": {"summary": "Liveness", "responses": {"200": {"description": "OK"}}}},
        "/v1/openapi.json": {
            "get": {"summary": "This specification", "responses": {"200": {"description": "OpenAPI JSON"}}}
        },
        "/v1/webhooks/session": {
            "post": {
                "summary": "Inbound webhook (signed)",
                "parameters": [
                    {
                        "name": "X-NULLXES-Timestamp",
                        "in": "header",
                        "required": False,
                        "schema": {"type": "string"},
                    },
                    {
                        "name": "X-NULLXES-Signature",
                        "in": "header",
                        "required": False,
                        "schema": {"type": "string"},
                    },
                    {
                        "name": "Idempotency-Key",
                        "in": "header",
                        "required": False,
                        "schema": {"type": "string"},
                    },
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["event", "session_id"],
                                "properties": {
                                    "event": {"type": "string"},
                                    "session_id": {"type": "string"},
                                    "correlation_id": {"type": "string"},
                                    "config": {"type": "object"},
                                    "callback_url": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "202": {"description": "Accepted"},
                    "401": {"description": "Invalid signature"},
                    "507": {"description": "No free media slots"},
                },
            }
        },
        "/v1/sessions/{id}/start": {
            "post": {
                "summary": "Start session worker",
                "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
                "responses": {"200": {"description": "OK"}, "404": {"description": "Not found"}},
            }
        },
        "/v1/sessions/{id}/stop": {
            "post": {
                "summary": "Graceful stop",
                "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
                "responses": {"200": {"description": "OK"}},
            }
        },
        "/v1/sessions/{id}/status": {
            "get": {
                "summary": "Session status",
                "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
                "responses": {"200": {"description": "OK"}, "404": {"description": "Not found"}},
            }
        },
        "/v1/sessions/{id}/media": {
            "patch": {
                "summary": "Bind media devices / RTP",
                "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}],
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "input_device_id": {"type": "string"},
                                    "output_device_id": {"type": "string"},
                                    "rtp_ingress_port": {"type": "integer"},
                                    "rtp_egress_host": {"type": "string"},
                                },
                            }
                        }
                    }
                },
                "responses": {"200": {"description": "OK"}, "404": {"description": "Not found"}},
            }
        },
        "/v1/media/slots": {
            "get": {"summary": "List media slots", "responses": {"200": {"description": "OK"}}}
        },
    },
}
