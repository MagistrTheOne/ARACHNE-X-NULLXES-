"""Minimal OpenAPI 3.0 JSON for MVP routes."""

from __future__ import annotations

SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "NULLXES Session & Media API",
        "version": "0.2.0",
        "description": "MVP webhook, session control, media slots, dashboard realtime token/WebSocket/chat (NULLXES / ARACHNE-X). See Documentation/D_SAAS/.",
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
        "/v1/realtime/token": {
            "post": {
                "summary": "Mint opaque browser token for WebSocket (server-to-server only)",
                "description": "Requires X-NULLXES-Realtime-Service-Key or Authorization: Bearer when NULLXES_REALTIME_SERVICE_KEY is set.",
                "parameters": [
                    {
                        "name": "X-NULLXES-Realtime-Service-Key",
                        "in": "header",
                        "required": False,
                        "schema": {"type": "string"},
                    },
                    {
                        "name": "Authorization",
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
                                "required": ["sessionId"],
                                "properties": {
                                    "sessionId": {"type": "string"},
                                    "employeeId": {"type": "string"},
                                    "nullxesSessionId": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Token and WebSocket URL",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["token", "websocketUrl", "issuedAt", "expiresAt"],
                                    "properties": {
                                        "token": {"type": "string"},
                                        "websocketUrl": {"type": "string"},
                                        "issuedAt": {"type": "string", "format": "date-time"},
                                        "expiresAt": {"type": "string", "format": "date-time"},
                                    },
                                }
                            }
                        },
                    },
                    "400": {"description": "Validation error"},
                    "401": {"description": "Missing or invalid service key"},
                },
            }
        },
        "/v1/chat": {
            "post": {
                "summary": "Optional HTTP chat (stub; MVP chat is WebSocket)",
                "description": "Server-to-server; same auth as /v1/realtime/token. stream=true returns text/event-stream.",
                "parameters": [
                    {
                        "name": "X-NULLXES-Realtime-Service-Key",
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
                                "required": ["sessionId", "messages"],
                                "properties": {
                                    "sessionId": {"type": "string"},
                                    "employeeId": {"type": "string"},
                                    "stream": {"type": "boolean"},
                                    "messages": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "role": {"type": "string"},
                                                "content": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "200": {"description": "JSON message or SSE stream"},
                    "400": {"description": "Validation error"},
                    "401": {"description": "Missing or invalid service key"},
                },
            }
        },
        "/v1/ws": {
            "get": {
                "summary": "Dashboard WebSocket (protocol v1)",
                "description": "Upgrade to WebSocket. Auth: query ?token=... from /v1/realtime/token, or first text frame {\"type\":\"auth\",\"token\":\"...\",\"protocolVersion\":1}. JSON text frames per Documentation/D_SAAS/WIRE_EXAMPLES.md. Close code 4401 on auth failure.",
                "parameters": [
                    {
                        "name": "token",
                        "in": "query",
                        "required": False,
                        "schema": {"type": "string"},
                    }
                ],
                "responses": {
                    "101": {"description": "Switching Protocols (WebSocket)"},
                    "401": {"description": "Not applicable to HTTP; auth errors use WS close 4401"},
                },
            }
        },
    },
}
