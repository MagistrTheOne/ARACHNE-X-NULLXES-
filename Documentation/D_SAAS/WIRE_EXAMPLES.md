# D_SAAS ↔ ARACHNE-X: примеры провода (token, WebSocket, chat, avatar preview stub)

Машиночитаемая спека также в `GET /v1/openapi.json` ([openapi_spec.py](../../src/server/openapi_spec.py)).

## Базовые URL (подставьте свой стенд)

| Окружение | HTTP | WebSocket |
|-----------|------|-----------|
| Локально (пример) | `http://127.0.0.1:8080` | `ws://127.0.0.1:8080` |
| Stage / prod | `https://<arachne-host>` | `wss://<arachne-host>` |

Для корректного `websocketUrl` в ответе токена задайте `NULLXES_PUBLIC_HTTP_BASE` или `NULLXES_PUBLIC_WS_BASE` на сервере (см. [TRUST_AND_ENV.md](./TRUST_AND_ENV.md)).

---

## 1. Mint токена (server-to-server)

**`POST /v1/realtime/token`**

Заголовок (если задан `NULLXES_REALTIME_SERVICE_KEY`):

```http
X-NULLXES-Realtime-Service-Key: <your-service-key>
```

или `Authorization: Bearer <your-service-key>`.

**Тело:**

```json
{
  "sessionId": "ui_sess_demo_1",
  "employeeId": "66",
  "nullxesSessionId": "nx_from_webhook_optional"
}
```

**Ответ 200:**

```json
{
  "token": "opaque-urlsafe-string",
  "websocketUrl": "ws://127.0.0.1:8080/v1/ws?token=opaque-urlsafe-string",
  "issuedAt": "2026-04-03T12:00:00Z",
  "expiresAt": "2026-04-03T12:15:00Z"
}
```

### curl (локально, без service key в dev)

```bash
curl -sS -X POST "http://127.0.0.1:8080/v1/realtime/token" \
  -H "Content-Type: application/json" \
  -d "{\"sessionId\":\"ui_sess_demo_1\",\"employeeId\":\"66\"}"
```

### curl (со service key)

```bash
curl -sS -X POST "https://arachne.example.com/v1/realtime/token" \
  -H "Content-Type: application/json" \
  -H "X-NULLXES-Realtime-Service-Key: $NULLXES_REALTIME_SERVICE_KEY" \
  -d "{\"sessionId\":\"ui_sess_demo_1\"}"
```

---

## 2. WebSocket `GET /v1/ws`

### Вариант A: токен в query

Подключиться к `websocketUrl` из ответа токена.

### Вариант B: первый кадр `auth`

Подключиться к `/v1/ws` без query, затем одно текстовое сообщение:

```json
{"type":"auth","token":"<opaque>","protocolVersion":1}
```

### Исходящие от сервера после успешной аутентификации

```json
{"type":"session.connecting","at":1712140800123}
```

```json
{"type":"session.connected","at":1712140800456}
```

### Входящие от клиента (примеры)

```json
{"type":"chat.send","id":"client-uuid-1","text":"Привет"}
```

```json
{"type":"voice.mute","muted":true}
```

```json
{"type":"session.disconnect"}
```

### Исходящие от сервера (примеры)

```json
{"type":"chat.message.received","at":1712140801200,"message":{"id":"reply_client-uuid-1","from":"assistant","text":"[stub] Привет"}}
```

```json
{"type":"session.error","at":1712140800999,"message":"auth_failed"}
```

```json
{"type":"session.disconnected","at":1712140810000,"reason":"client_close"}
```

Поле `at` — Unix time в **миллисекундах**.

### Коды закрытия WebSocket

| Код | Ситуация |
|-----|----------|
| `4401` | Невалидный или просроченный токен (до или вместо рабочей сессии) |
| `1008` | Допустима как альтернатива policy violation (см. контракт) |

Перед закрытием сервер может отправить кадр `session.error` с `message`: `auth_failed`, `session_expired`, `internal_error`.

### protocolVersion

MVP: **`protocolVersion`: `1`** в кадре `auth` (опционально, если не передан — принимается). Несовпадение при явной передаче → отказ в auth.

---

## 3. Опциональный HTTP chat `POST /v1/chat`

Тот же server-to-server ключ, что и для токена.

**Тело:**

```json
{
  "sessionId": "ui_sess_demo_1",
  "employeeId": "66",
  "stream": false,
  "messages": [{ "role": "user", "content": "Hello" }]
}
```

**Ответ 200 (stream: false, MVP-заглушка):**

```json
{
  "message": {
    "id": "chat_1712140800000",
    "role": "assistant",
    "content": "[stub] Hello"
  }
}
```

**stream: true** — ответ `Content-Type: text/event-stream`, чанки:

```
data: {"delta":"..."}

```

### curl

```bash
curl -sS -X POST "http://127.0.0.1:8080/v1/chat" \
  -H "Content-Type: application/json" \
  -d "{\"sessionId\":\"ui_sess_demo_1\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}"
```

---

## 4. Avatar preview stub `POST /v1/avatar/preview`

**Назначение:** отдать **публичный URL mp4** для UI / поля `employees.config.videoPreviewUrl` **без** запуска `scripts/infer.py`. Позже тот же путь можно заменить на реальный at2v / очередь.

**Авторизация:** как у mint — `X-NULLXES-Realtime-Service-Key` или `Authorization: Bearer`, если задан `NULLXES_REALTIME_SERVICE_KEY`.

**Сервер:** задайте `NULLXES_AVATAR_PREVIEW_VIDEO_URL` (HTTPS, прямой mp4). Иначе **503** `preview_not_configured`.

**Тело (все поля опциональны, для будущего пайплайна):**

```json
{
  "employeeId": "66",
  "sessionId": "ui_sess_demo_1",
  "imageUrl": "https://cdn.example.com/face.png",
  "speakText": "Короткая фраза для будущего TTS"
}
```

**Ответ 200:**

```json
{
  "videoPreviewUrl": "https://cdn.example.com/demos/avatar_stub.mp4",
  "status": "ready",
  "pipelineMode": "at2v_stub",
  "arachneOutputProfile": "gpt-realtime-arachne-v1-mvp"
}
```

`arachneOutputProfile` переопределяется env `NULLXES_ARACHNE_OUTPUT_PROFILE` (по умолчанию строка выше).

**Интеграция `dai_saas`:** `ARACHNE_AVATAR_PREVIEW_URL` на стороне Next обычно указывает на **этот** эндпоинт относительно `ARACHNE_HTTP_BASE`, например `https://<pod-proxy>/v1/avatar/preview` — Next проксирует server-to-server с ключом.

### curl

```bash
curl -sS -X POST "http://127.0.0.1:8080/v1/avatar/preview" \
  -H "Content-Type: application/json" \
  -H "X-NULLXES-Realtime-Service-Key: $NULLXES_REALTIME_SERVICE_KEY" \
  -d "{\"employeeId\":\"66\",\"sessionId\":\"ui_sess_1\"}"
```

---

## Bruno

Готовые запросы: папка [bruno/](./bruno/) (импорт коллекции в Bruno через Open Folder).
