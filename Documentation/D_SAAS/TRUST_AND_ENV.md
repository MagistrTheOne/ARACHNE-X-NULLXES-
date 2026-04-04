# D_SAAS ↔ ARACHNE-X: доверие, секреты, переменные окружения

**В этом файле нет прод-значений секретов** — только имена переменных и роли.

## Кто подписывает / выдаёт браузерный токен

| Режим | Описание |
|-------|----------|
| **MVP (текущий код ARACHNE-X)** | Короткоживущий **opaque** токен создаёт **ARACHNE-X** при `POST /v1/realtime/token`. Браузер получает его только через **Next** (`POST /api/arachine-x/token` в `dai_saas`), который вызывает ARACHNE server-to-server. |
| **Альтернатива (будущее)** | Next подписывает JWT (HS256/RS256); ARACHNE-X валидирует по общему секрету или JWKS. Тогда поля ответа те же, `token` = JWT. |

Итог для интеграции: **источник истины для mint в MVP — ARACHNE-X** после авторизации пользователя на стороне Next.

## Server-to-server: вызов `POST /v1/realtime/token`, `POST /v1/chat`, `POST /v1/avatar/preview`, `POST /v1/avatar/bootstrap`

| Переменная | Обязательность | Назначение |
|------------|----------------|------------|
| `NULLXES_REALTIME_SERVICE_KEY` | Рекомендуется в stage/prod | Общий секрет: заголовок `X-NULLXES-Realtime-Service-Key: <value>` или `Authorization: Bearer <value>`. Если **не задан**, проверка отключена (**только dev**; в лог пишется предупреждение). |

## Webhook линии A (без смешения с дашбордом)

| Переменная | Назначение |
|------------|------------|
| `NULLXES_WEBHOOK_SECRET` | HMAC для `POST /v1/webhooks/session` (`X-NULLXES-Timestamp`, `X-NULLXES-Signature`). |

## Публичные URL в ответе токена

Браузеру возвращается `websocketUrl`. Его собирают из:

| Переменная | Назначение |
|------------|------------|
| `NULLXES_PUBLIC_WS_BASE` | Явная база, например `wss://arachne-api.example.com` (без пути `/v1/ws`). |
| `NULLXES_PUBLIC_HTTP_BASE` | Если `WS_BASE` не задан: `https://...` → подмена на `wss://...`, `http://...` → `ws://...`. |
| (fallback) | Схема и `Host` из входящего HTTP-запроса mint-токена — удобно локально, **не** для прод за прокси без корректного `Host`. |

## Прочее (realtime MVP)

| Переменная | По умолчанию | Назначение |
|------------|--------------|------------|
| `NULLXES_REALTIME_TOKEN_TTL_SEC` | `900` | TTL opaque токена (секунды, минимум 60). |
| `NULLXES_WS_AVATAR_STREAM_STUB` | `1` | WebSocket: после `chat.send` слать цепочку avatar-событий. `0` — отключить. |
| `NULLXES_WS_AVATAR_STREAM_MODE` | (авто) | `inference` — GPU-воркер (LongCat и т.п.); `video` — локальный mp4 из `NULLXES_AVATAR_PREVIEW_ASSET_PATH`; `stub` — только метаданные (dev); `off`. Авто: `inference` если задан `NULLXES_AVATAR_INFERENCE_URL`; иначе `video` если файл превью есть; иначе **off** (без цепочки). |
| `NULLXES_AVATAR_INFERENCE_URL` | пусто | База HTTP GPU-воркера, например `http://10.0.0.5:9090`. `POST` путь см. `NULLXES_AVATAR_INFERENCE_PATH`. Ответ: тело **video/mp4** или JSON `videoBase64`. Воркер: **`services/longcat-worker/`** — LongCat (`LONGCAT_*`) или ARACHNE-X ULTRA (`ARACHNE_VIDEO_REPO`, `ARACHNE_CHECKPOINT_DIR`). Моки воркера только при **`ALLOW_INFERENCE_DEV_MOCK=1`**. См. README воркера. |
| `NULLXES_AVATAR_INFERENCE_PATH` | `/v1/longcat/generate` | Путь относительно базы. Эталон: `services/longcat-worker`. |
| `NULLXES_AVATAR_INFERENCE_SERVICE_KEY` | пусто | Если задан — заголовок `X-NULLXES-Avatar-Inference-Key` на воркер. |
| `NULLXES_AVATAR_INFERENCE_TIMEOUT_SEC` | `600` | Таймаут HTTP к воркеру (сек). |
| `NULLXES_AVATAR_INFERENCE_TASK` | `text-to-video` | `task` по умолчанию для WS, если в `chat.send` нет `inference.task`. Поддерживаются также `audio-text-to-video`, `audio-image-to-video` (воркер ARACHNE). |
| `NULLXES_AVATAR_INFERENCE_IMAGE_BASE64` | пусто | Референс-картинка, если не передана в `chat.send.inference.imageBase64`. |
| `NULLXES_AVATAR_INFERENCE_AUDIO_BASE64` | пусто | Аудио (base64), если не в кадре `chat.send.inference.audioBase64`. |
| `NULLXES_AVATAR_INFERENCE_CONTINUATION_BASE64` | пусто | Conditioning mp4 base64 для continuation, если не в кадре. |
| `NULLXES_CHAT_ASSISTANT_FIXED_REPLY` | пусто | HTTP `POST /v1/chat`: если задано — текст ответа assistant; иначе эхо последнего user `content`. |
| `NULLXES_WS_CHAT_ASSISTANT_FIXED_REPLY` | пусто | WebSocket `chat.message.received`: если задано — текст assistant; иначе эхо `text` из `chat.send`. |
| `NULLXES_WS_AVATAR_STREAM_NUM_CHUNKS` | `5` | Режим **stub**: число `avatar.stream.chunk` (1…60). |
| `NULLXES_WS_AVATAR_STREAM_CHUNK_MS` | `40` | Режим **stub**: задержка между чанками (мс, 0…500). |
| `NULLXES_WS_AVATAR_VIDEO_MAX_FRAMES` | `120` | **video** / **inference**: макс. кадров за один ответ (после декода MP4). |
| `NULLXES_WS_AVATAR_VIDEO_MAX_WIDTH` | `480` | **video** / **inference**: ширина JPEG после даунскейла. |
| `NULLXES_WS_AVATAR_VIDEO_JPEG_QUALITY` | `80` | **video** / **inference**: JPEG quality 30…95. |
| `NULLXES_CORS_ORIGIN` | пусто | Если задано (один origin), к ответам `POST /v1/realtime/token`, `POST /v1/chat`, `POST /v1/avatar/preview` и `POST /v1/avatar/bootstrap` добавляются CORS-заголовки и обрабатывается `OPTIONS` (прямой вызов из браузера **не** рекомендуется; типичный путь — только Next server-side). |

## Avatar preview (static URL / файл, без infer)

| Переменная | Обязательность | Назначение |
|------------|----------------|------------|
| `NULLXES_PUBLIC_HTTP_BASE` | Для same-origin превью | Публичный origin браузера, например `https://1qs8mciim8zovo-8080.proxy.runpod.net`. Участвует в сборке `videoPreviewUrl` = `{base}/v1/avatar/preview/asset.mp4`. |
| `NULLXES_AVATAR_PREVIEW_ASSET_PATH` | Альтернатива внешнему URL | Локальный путь к **mp4** на машине ARACHNE; отдаётся через **`GET /v1/avatar/preview/asset.mp4`** (без ключа). |
| `NULLXES_AVATAR_PREVIEW_VIDEO_URL` | Опционально | Явный полный HTTPS URL mp4; если задан, **перекрывает** same-origin режим. |
| `NULLXES_ARACHNE_OUTPUT_PROFILE` | Нет | Строка для поля `arachneOutputProfile` в JSON-ответе (по умолчанию `gpt-realtime-arachne-v1-mvp`). |
| `NULLXES_AVATAR_BOOTSTRAP_PREVIEW_COOLDOWN_SEC` | `0` | Только `POST /v1/avatar/bootstrap`: секунды, в течение которых для одного ключа `sessionId` + `employeeId` повторно отдаётся **закэшированный** блок превью (без повторного «тяжёлого» шага; в static режиме это тот же URL). Токен WS **всегда** новый. Тело `regeneratePreview: true` сбрасывает кэш для ключа. |

Для **200** на `POST /v1/avatar/preview` нужно либо непустой `NULLXES_AVATAR_PREVIEW_VIDEO_URL`, либо валидный файл по `NULLXES_AVATAR_PREVIEW_ASSET_PATH` плюс корректный `NULLXES_PUBLIC_HTTP_BASE` (или сработает fallback `Host` / `X-Forwarded-Proto` из запроса).

## Рекомендуемые переменные на стороне `dai_saas` (справочно)

Имена задаёт репозиторий `dai_saas`; ориентир:

- `ARACHNE_HTTP_BASE` / `NEXT_PUBLIC_ARACHNE_HTTP_BASE` — база HTTP к ARACHNE (или к Next rewrite).
- Секрет для вызова mint: тот же `NULLXES_REALTIME_SERVICE_KEY` на стороне ARACHNE и **server-only** копия в Next (например `ARACHNE_REALTIME_SERVICE_KEY`), **не** `NEXT_PUBLIC_*`.
