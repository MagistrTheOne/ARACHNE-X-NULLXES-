# D_SAAS (dashboard цифровых сотрудников)

Документы в этой папке относятся к стыковке **Next.js-приложения `dai_saas`** с **ARACHNE-X backend**. Они **не смешиваются** с материалами пилота JobAI — те лежат в [`Documentation/JOBAI PILOT/`](../JOBAI%20PILOT/).

| Файл | Назначение |
|------|------------|
| [ARACHNE_X_FRONTEND_CONTRACT.md](./ARACHNE_X_FRONTEND_CONTRACT.md) | Контракт: токен, WebSocket, HTTP-чат, avatar preview stub, CORS, типы `ArachineXEvent` |
| [WIRE_EXAMPLES.md](./WIRE_EXAMPLES.md) | Примеры JSON-кадров, коды WS, curl |
| [TRUST_AND_ENV.md](./TRUST_AND_ENV.md) | Секреты server-to-server, env (без значений) |
| [DAI_SAAS_HANDOFF.md](./DAI_SAAS_HANDOFF.md) | Чеклист передачи во фронт после стенда |
| [bruno/](./bruno/) | Коллекция Bruno (mint token, chat stub, avatar preview) |

Код фронта живёт в отдельном репозитории/дереве `dai_saas`; здесь только спецификации NULLXES для интеграции. Реализация линии B в ARACHNE-X: `src/server/realtime_api.py`, `src/server/realtime_store.py`, маршруты в `src/server/webrtc_server.py`.
