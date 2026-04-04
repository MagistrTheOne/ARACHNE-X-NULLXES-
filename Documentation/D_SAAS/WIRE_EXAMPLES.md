# D_SAAS ↔ ARACHNE-X: примеры провода (token, WebSocket, chat, avatar preview, bootstrap)

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

**Вариант B (оживление по WS):** сразу после `chat.message.received` сервер (если не выключено env) шлёт асинхронно `avatar.state.changed` → `speaking` → серия `avatar.stream.chunk` → `avatar.state.changed` → `idle`.

**Прод (LongCat / [LongCat-Video](https://huggingface.co/meituan-longcat/LongCat-Video)):** задайте **`NULLXES_AVATAR_INFERENCE_URL`** на HTTP-воркер (шаблон в репозитории: `services/longcat-worker`). После `chat.send` ARACHNE-X вызывает воркер с текстом пользователя как `prompt`, получает **MP4**, декодирует в JPEG и шлёт те же `avatar.stream.chunk` с `jpeg_base64`.

**Локальный файл без воркера:** если задан **`NULLXES_AVATAR_PREVIEW_ASSET_PATH`** и режим `video`, кадры из этого mp4. Если URL воркера не задан и файла нет — по умолчанию **цепочка не шлётся** (`off`). Режим **`stub`** — только метаданные (`seq`, `kind`) для dev.

```json
{"type":"avatar.state.changed","at":1712140801201,"state":"speaking"}
```

```json
{"type":"avatar.stream.chunk","at":1712140801245,"kind":"video","seq":1,"encoding":"jpeg_base64","data":"<base64...>"}
```

Режим только метаданных (пример без `encoding`/`data`):

```json
{"type":"avatar.stream.chunk","at":1712140801245,"kind":"video","seq":1}
```

Полностью отключить цепочку: `NULLXES_WS_AVATAR_STREAM_STUB=0`.

| Env | По умолчанию | Смысл |
|-----|--------------|--------|
| `NULLXES_WS_AVATAR_STREAM_STUB` | `1` | `0` — не слать avatar.* после `chat.send` |
| `NULLXES_WS_AVATAR_STREAM_MODE` | (авто) | `inference` \| `video` \| `stub` \| `off`. Авто: `inference` при `NULLXES_AVATAR_INFERENCE_URL`; иначе `video` при файле превью; иначе `off` |
| `NULLXES_AVATAR_INFERENCE_URL` | — | База воркера (prod) |
| `NULLXES_AVATAR_INFERENCE_PATH` | `/v1/longcat/generate` | POST generate |
| `NULLXES_AVATAR_INFERENCE_TASK` | `text-to-video` | Задача для воркера (T2V / I2V / continuation) |
| `NULLXES_WS_AVATAR_STREAM_NUM_CHUNKS` | `5` | Для **stub**: число `avatar.stream.chunk` (1…60) |
| `NULLXES_WS_AVATAR_STREAM_CHUNK_MS` | `40` | Для **stub**: пауза между чанками, мс (0…500) |
| `NULLXES_WS_AVATAR_VIDEO_MAX_FRAMES` | `120` | Для **video**: максимум кадров за один проход (субсэмпл из длинного mp4) |
| `NULLXES_WS_AVATAR_VIDEO_MAX_WIDTH` | `480` | Для **video**: масштаб по ширине перед JPEG |
| `NULLXES_WS_AVATAR_VIDEO_JPEG_QUALITY` | `80` | Для **video**: качество JPEG (30…95) |

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

**Сервер (один из вариантов):**

1. **Тот же хост, что RunPod proxy (рекомендуется):**  
   `NULLXES_PUBLIC_HTTP_BASE=https://1qs8mciim8zovo-8080.proxy.runpod.net`  
   `NULLXES_AVATAR_PREVIEW_ASSET_PATH=/workspace/demo.mp4` (любой локальный файл mp4 на поде)  
   Тогда `POST` вернёт  
   `videoPreviewUrl` = `https://1qs8mciim8zovo-8080.proxy.runpod.net/v1/avatar/preview/asset.mp4`,  
   а сам файл отдаётся **`GET /v1/avatar/preview/asset.mp4`** (без service key, для `<video src>`).

2. **Внешний CDN:** `NULLXES_AVATAR_PREVIEW_VIDEO_URL=https://…/file.mp4` (перекрывает вариант 1, если задано).

Если ни URL, ни файл не настроены — **503** `preview_not_configured`.

**Аудио:** в текущем стенде голос идёт через **GPT Realtime** (или иной канал), **не** через загрузку wav в этот POST.

**Тело (все поля опциональны; для будущего at2v можно расширить):**

```json
{
  "employeeId": "66",
  "sessionId": "ui_sess_demo_1"
}
```

**Ответ 200:**

```json
{
  "videoPreviewUrl": "https://cdn.example.com/demos/avatar_stub.mp4",
  "status": "ready",
  "pipelineMode": "static_preview",
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

## 5. Один вызов: превью + WebSocket `POST /v1/avatar/bootstrap`

**Назначение:** одна server-to-server команда для дашборда: то же, что **`POST /v1/realtime/token`** + поля статическое превью (`videoPreviewUrl`, …). **Без** аудио-ассетов — звук остаётся в **GPT Realtime**.

**Тело (обязателен только `sessionId`):**

```json
{
  "sessionId": "ui_sess_demo_1",
  "employeeId": "66",
  "nullxesSessionId": "optional_from_line_a"
}
```

**Ответ 200 (пример):**

```json
{
  "sessionId": "ui_sess_demo_1",
  "token": "opaque…",
  "websocketUrl": "wss://…/v1/ws?token=opaque…",
  "issuedAt": "2026-04-03T12:00:00Z",
  "expiresAt": "2026-04-03T12:15:00Z",
  "videoPreviewUrl": "https://…/v1/avatar/preview/asset.mp4",
  "avatarPreviewStatus": "ready",
  "pipelineMode": "static_preview",
  "arachneOutputProfile": "gpt-realtime-arachne-v1-mvp",
  "audioTransport": "gpt_realtime",
  "avatarPreviewCached": false
}
```

**AT2V / генерация:** сейчас `pipelineMode: "static_preview"` — **нет** вызова `infer.py` / DiT. Когда подключите реальный at2v, генерация будет жить за этим же контрактом; кулдаун (ниже) как раз чтобы **не гонять** её на каждый bootstrap.

**Кулдаун превью (анти-спам / «ваншот» на окно времени):**  
`NULLXES_AVATAR_BOOTSTRAP_PREVIEW_COOLDOWN_SEC` (например `300`) — для одной пары **`sessionId` + `employeeId`** в течение окна повторно подставляется **тот же** блок превью (`videoPreviewUrl`, …) из памяти процесса; **`token` и `websocketUrl` всё равно новые** на каждый вызов. В ответе **`avatarPreviewCached: true`**, если превью взято из кэша.

Сбросить кэш и пересчитать превью (позже — перезапустить генерацию):

```json
{ "sessionId": "ui_sess_demo_1", "regeneratePreview": true }
```

(алиас: `forceAvatarRefresh`.)

Те же env, что для §4 (превью). Если превью не сконфигурировано — **503** (токен не выдаётся).

**Интеграция `dai_saas`:** один Next route, например `POST /api/arachine-x/avatar-bootstrap` → прокси на `{ARACHNE_HTTP_BASE}/v1/avatar/bootstrap` с ключом; клиент получает и `websocketUrl`, и `videoPreviewUrl` одним ответом.

### curl

```bash
curl -sS -X POST "http://127.0.0.1:8080/v1/avatar/bootstrap" \
  -H "Content-Type: application/json" \
  -H "X-NULLXES-Realtime-Service-Key: $NULLXES_REALTIME_SERVICE_KEY" \
  -d "{\"sessionId\":\"ui_sess_1\",\"employeeId\":\"66\"}"
```

---

## Bruno

Готовые запросы: папка [bruno/](./bruno/) (импорт коллекции в Bruno через Open Folder).
