# Передача во фронт `dai_saas` после поднятия стенда

Чеклист для команды Next после того, как DevOps выдал **финальные** URL (без секретов в чате).

## 1. URL и прокси

| Параметр | Что зафиксировать |
|----------|-------------------|
| HTTP base | Точный origin для server-side вызовов `POST /v1/realtime/token` (например `https://arachne-api.dev.nullxes.example`). |
| WebSocket | Либо прямой `wss://.../v1/ws`, либо **same-origin** прокси в Next (`/api/arachne/ws` → upstream), чтобы упростить CORS и cookies. |
| OpenAPI | `GET {HTTP_BASE}/v1/openapi.json` — версия и пути должны совпадать с [ARACHNE_X_FRONTEND_CONTRACT.md](./ARACHNE_X_FRONTEND_CONTRACT.md). |

**Рекомендация:** для prod предпочтительно **Next proxy** для WS и отсутствие прямого `wss` с фронта на чужой домен; иначе на ARACHNE нужен явный allowlist `Origin`.

## 2. Нормализация `issuedAt` / `expiresAt`

- **ARACHNE-X MVP** отдаёт **строки ISO-8601 с `Z`**.
- Если код `dai_saas` ожидает **числа (Unix ms)**, нормализовать **только** в route `POST /api/arachine-x/token` при проксировании ответа ARACHNE.

## 3. Секреты в Next (server-only)

- Хранить ключ для `X-NULLXES-Realtime-Service-Key` / `Authorization: Bearer` в **server** env (не префикс `NEXT_PUBLIC_`).
- Согласовать имя с [TRUST_AND_ENV.md](./TRUST_AND_ENV.md).

## 4. Поведение UI (MVP)

- **Чат:** основной канал — **WebSocket** (`chat.send` / `chat.message.received`); HTTP `chatTurn` не подключать к REST без отдельного решения продукта.
- **Bootstrap:** поле `sessionId` = UI-сессия; `nullxesSessionId` в mint-теле — только если platform-backend уже получил id из линии A.

## 5. Smoke-тест с фронта

1. Server action / route: mint через ARACHNE с тестовым `sessionId`.
2. Клиент: открыть `websocketUrl` из ответа (или прокси-эквивалент).
3. Убедиться в последовательности `session.connecting` → `session.connected` и ответе на `chat.send`.

Примеры curl: [WIRE_EXAMPLES.md](./WIRE_EXAMPLES.md).
