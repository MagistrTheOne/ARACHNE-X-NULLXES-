# Bruno — D_SAAS realtime

1. Установите [Bruno](https://www.usebruno.com/).
2. **Open Collection** → выберите эту папку `Documentation/D_SAAS/bruno`.
3. Выберите окружение **local** и при необходимости задайте `NULLXES_REALTIME_SERVICE_KEY` (если ключ задан на сервере).

WebSocket `GET /v1/ws` в Bruno проверяйте отдельно (возьмите `websocketUrl` из ответа mint-токена) или через `wscat` / клиент в `dai_saas`.
