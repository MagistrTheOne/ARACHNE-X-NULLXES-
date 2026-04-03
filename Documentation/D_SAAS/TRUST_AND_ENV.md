# D_SAAS ↔ ARACHNE-X: доверие, секреты, переменные окружения

**В этом файле нет прод-значений секретов** — только имена переменных и роли.

## Кто подписывает / выдаёт браузерный токен

| Режим | Описание |
|-------|----------|
| **MVP (текущий код ARACHNE-X)** | Короткоживущий **opaque** токен создаёт **ARACHNE-X** при `POST /v1/realtime/token`. Браузер получает его только через **Next** (`POST /api/arachine-x/token` в `dai_saas`), который вызывает ARACHNE server-to-server. |
| **Альтернатива (будущее)** | Next подписывает JWT (HS256/RS256); ARACHNE-X валидирует по общему секрету или JWKS. Тогда поля ответа те же, `token` = JWT. |

Итог для интеграции: **источник истины для mint в MVP — ARACHNE-X** после авторизации пользователя на стороне Next.

## Server-to-server: вызов `POST /v1/realtime/token`, `POST /v1/chat`, `POST /v1/avatar/preview`

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
| `NULLXES_CORS_ORIGIN` | пусто | Если задано (один origin), к ответам `POST /v1/realtime/token`, `POST /v1/chat` и `POST /v1/avatar/preview` добавляются CORS-заголовки и обрабатывается `OPTIONS` (прямой вызов из браузера **не** рекомендуется; типичный путь — только Next server-side). |

## Avatar preview (stub, без infer)

| Переменная | Обязательность | Назначение |
|------------|----------------|------------|
| `NULLXES_PUBLIC_HTTP_BASE` | Для same-origin превью | Публичный origin браузера, например `https://1qs8mciim8zovo-8080.proxy.runpod.net`. Участвует в сборке `videoPreviewUrl` = `{base}/v1/avatar/preview/asset.mp4`. |
| `NULLXES_AVATAR_PREVIEW_ASSET_PATH` | Альтернатива внешнему URL | Локальный путь к **mp4** на машине ARACHNE; отдаётся через **`GET /v1/avatar/preview/asset.mp4`** (без ключа). |
| `NULLXES_AVATAR_PREVIEW_VIDEO_URL` | Опционально | Явный полный HTTPS URL mp4; если задан, **перекрывает** same-origin режим. |
| `NULLXES_ARACHNE_OUTPUT_PROFILE` | Нет | Строка для поля `arachneOutputProfile` в JSON-ответе (по умолчанию `gpt-realtime-arachne-v1-mvp`). |

Для **200** на `POST /v1/avatar/preview` нужно либо непустой `NULLXES_AVATAR_PREVIEW_VIDEO_URL`, либо валидный файл по `NULLXES_AVATAR_PREVIEW_ASSET_PATH` плюс корректный `NULLXES_PUBLIC_HTTP_BASE` (или сработает fallback `Host` / `X-Forwarded-Proto` из запроса).

## Рекомендуемые переменные на стороне `dai_saas` (справочно)

Имена задаёт репозиторий `dai_saas`; ориентир:

- `ARACHNE_HTTP_BASE` / `NEXT_PUBLIC_ARACHNE_HTTP_BASE` — база HTTP к ARACHNE (или к Next rewrite).
- Секрет для вызова mint: тот же `NULLXES_REALTIME_SERVICE_KEY` на стороне ARACHNE и **server-only** копия в Next (например `ARACHNE_REALTIME_SERVICE_KEY`), **не** `NEXT_PUBLIC_*`.
