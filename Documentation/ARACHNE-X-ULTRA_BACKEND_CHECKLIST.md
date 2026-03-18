# ARACHNE-X ULTRA Backend Checklist

## Done

- H200 pod поднят
- CUDA / PyTorch видят `NVIDIA H200`
- `ARACHNE-X` код на pod
- `src/` realtime backend skeleton добавлен
- `faster-whisper` runtime установлен
- `silero-vad` runtime установлен
- `aiohttp` установлен
- `aiortc` установлен
- `flash-attn` установлен
- `CosyVoice` runtime поднят
- `Qwen TTS` adapter добавлен в код
- Metered STUN/TURN credentials получены
- `scripts/infer.py` определён как основной inference entrypoint
- launch docs и быстрые launch-скрипты добавлены

## Partially Done

- `ARACHNE-X` base checkpoint path подтверждён
- `ARACHNE-X-Avatar` checkpoint path подтверждён
- `Qwen3-TTS` модель скачана на pod
- `Qwen3-TTS` реально прогнана в вашем runtime
- `single avatar ai2v` demo прогнан до конца
- `t2v` smoke test прогнан до конца
- asset layout на pod зафиксирован окончательно
- runtime env contract зафиксирован
- HTTP/WebRTC serving path поднят как сервис
- client-side ICE config подключён к frontend

## Missing

- единый runtime config файл
- `run_webrtc_server.py`
- backend endpoint для `/offer`
- backend endpoint для `/health`
- backend endpoint `/ice` или другой способ выдачи ICE config
- `Qwen TTS` end-to-end integration в orchestrator на pod
- полный E2E сценарий:
  - mic/audio in
  - VAD
  - ASR
  - OpenAI response
  - Qwen TTS
  - avatar render
  - video/audio out
- startup self-check script
- queue/backpressure tuning
- interrupt/barge-in runtime test
- metrics/logging contract
- production secret handling
- final TURN key rotation
- checkpoint/weights audit
- `Silero` local model file path зафиксирован
- `Whisper` model artifact path зафиксирован
- reference voice assets зафиксированы
- avatar identity assets зафиксированы

## Need From You

- финальный `OPENAI_API_KEY`
- финальный `METERED` key/credential после ротации
- решение по TTS:
  - `Qwen CustomVoice`
  - или `Qwen VoiceClone`
- точные checkpoint paths
- финальные avatar assets
- финальные reference voice assets

## Best Next Steps

- скачать и проверить `Qwen3-TTS`
- сделать runtime config
- добавить `run_webrtc_server.py`
- поднять `/health`
- поднять `/offer`
- сделать первый E2E backend-only прогон
