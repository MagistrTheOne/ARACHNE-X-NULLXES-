# ARACHNE-X ↔ D_SAAS (dashboard): контракт интеграции

**Сторона:** NULLXES  
**Назначение:** единая спека для стыковки **ARACHNE-X backend** (репозиторий ARACHNE-X) и **фронта D_SAAS** (`dai_saas`, Next.js dashboard цифровых сотрудников).  
**Источник типов на фронте:** `features/arachine-x/event-system/eventTypes.ts` (имя пакета с опечаткой `arachine-x`; переименование — отдельная задача).  
**Статус:** REST-оркестрация MVP описана и частично реализована в ARACHNE-X; **WebSocket realtime** и продуктовый **token flow** в этом документе **фиксируются как контракт** — реализация на бэке/фронте выполняется по этой спецификации.

---

## 1. Две линии (не смешивать)

| Линия | Назначение | Где на фронте (ориентир) | Где на бэке (ARACHNE-X) |
|--------|------------|---------------------------|-------------------------|
| **A. Оркестрация сессий** | Webhook JobAI, слоты, lifecycle без постоянного соединения с внешним HR-backend | Не обязательно в UI; может вызываться только platform-backend | Реализовано: `POST /v1/webhooks/session`, `/v1/sessions/...`, `/v1/media/slots` — см. [NULLXES_MVP_Media_Layer_API_03-04-2026.md](./NULLXES_MVP_Media_Layer_API_03-04-2026.md), OpenAPI `GET /v1/openapi.json` |
| **B. Dashboard realtime** | Токен, WebSocket, сигналы сессии/аватара/чата в браузере | `useAvatarRuntime`, `WebSocketTransport` (сейчас no-op), bootstrap сотрудника | **Не реализовано** в `src/server` на момент документа; контракт ниже |

**Текстовый чат** в UI может идти **отдельным HTTP/SSE** (`chatTurn.ts` — заглушка под `fetch`) **или** тем же WebSocket — продукту нужно выбрать **один** основной канал для чата и зафиксировать здесь (по умолчанию ниже: чат допускается и по WS как `chat.send` / `chat.message.received`, и отдельным REST — но не дублировать без согласования).

```mermaid
flowchart LR
  subgraph dashboard [D_SAAS Next app]
    UI[EmployeeInteractionPage]
    CHAT[chatTurn HTTP stub]
    WS[WebSocketTransport skeleton]
    TOK["route /api/.../token"]
  end
  subgraph arachne [ARACHNE-X backend]
    REST["REST /v1 webhooks sessions media"]
    WSrv["WS realtime TBD"]
  end
  UI --> CHAT
  UI --> WS
  UI --> TOK
  TOK -.->|contract| REST
  WS -.->|contract| WSrv
  CHAT -.->|optional| REST
```

---

## 2. Environments

| Окружение | Переменная / значение | Примечание |
|-----------|------------------------|------------|
| Dev | `NEXT_PUBLIC_ARACHNE_HTTP_BASE` (пример) | База HTTP API оркестратора (если фронт бьёт напрямую) |
| Dev | `NEXT_PUBLIC_ARACHNE_WS_URL` (пример) | Origin WebSocket **когда** сервер будет готов |
| Stage / Prod | Те же, с прод-оригинами | Рекомендуется **Next rewrite/proxy**: браузер бьёт в same-origin `/api/arachne/...` → upstream ARACHNE |

Плейсхолдеры имён env — согласовать с реальным `dai_saas`; важно зафиксировать **итоговые** имена в `.env.example` фронта и в деплое бэка.

---

## 3. HTTP: выдача сессии и токена (совместимость с фронтом)

Фронт ожидает форму ответа в духе (заглушка `issueAvatarTokenClaims` / route token):

```json
{
  "token": "<string>",
  "websocketUrl": "wss://example.com/v1/realtime",
  "issuedAt": "2026-04-03T12:00:00.000Z",
  "expiresAt": "2026-04-03T12:15:00.000Z"
}
```

**Рекомендации NULLXES для бэка:**

- **Вариант 1 — JWT:** подпись (например HS256/RS256), claims минимум: `sub` (user), `employeeId`, `sessionId` или `roomId`, `capabilities` (массив строк), `aud` (например `arachne-realtime`), `iat`, `exp`. Срок жизни **короткий** (5–15 мин) для WS; обновление — повторный `POST` token или refresh flow (описать отдельно).
- **Вариант 2 — opaque token:** случайный id, сервер хранит сессию в Redis; в ответе тот же JSON, `token` не parseable на клиенте.

**Эндпоинт:** может оставаться на Next (`POST /api/arachine-x/token`), который **проксирует** или **подписывает** ответ, либо напрямую `POST https://arachne.../v1/token` — главное, чтобы **поля** совпадали с ожиданием UI.

---

## 4. Связь с NULLXES webhook и `sessionId` в UI

- **`nullxes_session_id`** (ответ `POST /v1/webhooks/session`) — идентификатор сессии **оркестратора** на ноде NULLXES (интервью, media slot, worker).
- **`sessionId` в dashboard** может быть:
  - **тот же**, если platform создаёт интервью через webhook и передаёт id в ссылку на страницу сотрудника;
  - **отдельный** «комнатный» id, если UI-сессия создаётся при открытии страницы — тогда в токене нужны **оба** (например `nullxesSessionId` + `uiSessionId`) или маппинг на бэке platform.

Нужно явно выбрать модель продукта и отразить в claims токена; в таблице ниже поле `sessionId` на проводе = **то, что фронт кладёт в bootstrap** (`getEmployeeSessionBootstrap`).

---

## 5. WebSocket (целевой контракт)

**URL (пример):** `wss://api.example.com/v1/ws?token=<jwt_or_opaque>`  
**Альтернатива:** первый исходящий JSON-кадр от клиента после `onopen`:

```json
{ "type": "auth", "token": "<...>", "protocolVersion": 1 }
```

**Аутентификация:** предпочтительно **query `token`** или первый кадр `auth` (удобно в браузере). Заголовок `Authorization` на WS часто **не доступен** из JS — если прокси добавляет заголовок к upstream, это допустимо, но нужно документировать.

**Формат кадров (MVP сигналов):** **JSON text frames**, UTF-8, одно событие на кадр (можно расширить до NDJSON позже).

**Медиа (видео/аудио поток):** для production чаще **отдельный транспорт** (WebRTC, binary chunks с префиксом). В типах фронта уже есть `avatar.stream.chunk` с `seq` — на проводе для MVP допускается **только метаданные** без payload (`seq`, `kind`), а реальные кадры — в следующей версии протокола или параллельным каналом (**зафиксировать в `protocolVersion`**).

---

## 6. Таблица: провод ↔ `ArachineXEvent` / `ArachineXOutboundAction`

Имена полей на проводе **совпадают** с TypeScript-типами, если не оговорено иное.

### 6.1 Сервер → клиент (`ArachineXEvent`)

| `type` | Пример JSON |
|--------|-------------|
| `session.connecting` | `{"type":"session.connecting","at":1712140800123}` |
| `session.connected` | `{"type":"session.connected","at":1712140800456}` |
| `session.disconnected` | `{"type":"session.disconnected","at":1712140810000,"reason":"client_close"}` |
| `session.error` | `{"type":"session.error","at":1712140800999,"message":"auth_failed"}` |
| `avatar.state.changed` | `{"type":"avatar.state.changed","at":1712140801000,"state":"speaking"}` |
| `avatar.stream.chunk` | `{"type":"avatar.stream.chunk","at":1712140801100,"kind":"video","seq":42}` |
| `chat.message.received` | `{"type":"chat.message.received","at":1712140801200,"message":{"id":"msg_1","from":"assistant","text":"Здравствуйте."}}` |

`at` — Unix timestamp в миллисекундах (как `Date.now()` в JS).

### 6.2 Клиент → сервер (`ArachineXOutboundAction`)

| `type` | Пример JSON |
|--------|-------------|
| `chat.send` | `{"type":"chat.send","id":"client_uuid","text":"Привет"}` |
| `voice.mute` | `{"type":"voice.mute","muted":true}` |
| `session.disconnect` | `{"type":"session.disconnect"}` |

Сервер **игнорирует** неизвестные `type` с логом (MVP) или отвечает `session.error` — политику выбрать и не менять молча.

---

## 7. HTTP чат (опционально, если не в WS)

Если чат идёт только через REST:

- **Метод:** `POST /v1/chat` (префикс согласовать с OpenAPI).
- **Body (пример):**

```json
{
  "sessionId": "nx_...",
  "employeeId": "66",
  "messages": [{ "role": "user", "content": "..." }],
  "stream": false
}
```

- **Ответ:** либо полный JSON с текстом ассистента, либо **SSE** при `stream: true` (формат чанков зафиксировать: `data: {"delta":"..."}\n\n`).

Это заменяет заглушку в `features/arachne-x/chatTurn.ts` (подключение `fetch`).

---

## 8. Ошибки

| Контекст | Код / поведение |
|----------|------------------|
| HTTP token | `401` неверная сессия/пользователь; `403` нет прав на employee; `429` rate limit |
| HTTP chat | `400` валидация; `502` upstream LLM |
| WebSocket | Закрытие **код 4401** (или `1008` policy violation) при невалидном токене; тело причин до закрытия опционально дублировать событием `session.error` |
| Внутри WS | Всегда можно отправить `session.error` с `message` из машинно-читаемого набора (`auth_failed`, `session_expired`, `internal_error`) |

---

## 9. Версионирование протокола

- Рекомендуется **`protocolVersion: 1`** в первом клиентском сообщении (`auth` или отдельный кадр `hello`).
- Либо путь **`/v1/ws`** и несовместимые изменения → `/v2/ws`.

---

## 10. CORS, cookies, Better Auth

- При **разных доменах** фронта и ARACHNE: CORS для HTTP; для WS — либо **same-origin proxy** в Next, либо явный `wss://` и корректный `Origin` на сервере.
- **Better Auth:** практичный путь — Next route `/api/.../token` проверяет сессию пользователя и выдаёт **короткоживущий токен** для ARACHNE; сам ARACHNE **не обязан** доверять browser cookie, если нет общего секрета/proxy.

---

## 11. Артефакты для разработчика

- Этот файл: **единая страница контракта** для Cursor / партнёров.  
- Оркестрация и media MVP: [NULLXES_MVP_Media_Layer_API_03-04-2026.md](./NULLXES_MVP_Media_Layer_API_03-04-2026.md).  
- Машиночитаемые REST-маршруты MVP: в репозитории ARACHNE-X `GET /v1/openapi.json` (реализация [`src/server/openapi_spec.py`](../../src/server/openapi_spec.py)).  
- **Postman / Bruno / curl** — добавить по мере появления реальных стендов (URL тестового API без прод-секретов).

---

## 12. Вне текущего скоупа репозитория ARACHNE-X

- Реализация WebSocket-сервера (aiohttp / отдельный сервис) и прокидывание событий в типы выше.  
- Реализация `WebSocketTransport.connect` / `send` в `dai_saas`.  
- Переименование `arachine-x` → `arachne-x` на фронте.

---

*NULLXES — контракт dashboard ↔ ARACHNE-X.*
