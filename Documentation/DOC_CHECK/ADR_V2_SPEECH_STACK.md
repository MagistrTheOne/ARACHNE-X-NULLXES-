# ADR: V2 speech stack (NIGHT FURY)

## Status

Accepted (documentation) — интеграция в код по фазам после HF approval.

## Context

V2 требует: ASR (Whisper Large V3 Turbo), emotion embeddings (emotion2vec), TTS (Qwen3-TTS), при этом **lip conditioning** остаётся **internal Wav2Vec2** из NULLXES AVATAR checkpoint (`audio/wav2vec2`).

## SpeechAdapters (вне CoreNULLXES ownership)

Публичные HF-модели в этом контуре: **Whisper Large V3 Turbo**, **`facebook/wav2vec2-base`** (только как опциональный фонемный/просодический адаптер, не как замена checkpoint lip-ветки без отдельного согласования), **`emotion2vec_base`**, **Qwen3-TTS**.

## Decision

1. **Lip:** только веса из чекпоинта; отдельный публичный HF Wav2Vec2 для фонем **не** подключается по умолчанию (избежать двойного конфликтующего conditioning).
2. **ASR:** `openai/whisper-large-v3-turbo` через **faster-whisper** / CTranslate2 или эквивалент; вход PCM mono 16 kHz.
3. **Emotion:** `emotion2vec/emotion2vec_base` → признаки в **ControlBus** ([`arachne_x/actor_v2`](../../arachne_x/actor_v2)).
4. **TTS:** Qwen3-TTS — primary; совместимость с [`arachne_x/runtime/inference_engine.py`](../../arachne_x/runtime/inference_engine.py) `resolve_avatar_wav_path` и `requirements-tts.txt`.

## Consequences

- Отдельный venv или образ слой для optional `requirements-audiodit` (тянет `transformers>=5.3`), конфликтующий с core `transformers==4.41.0`.
- Speech веса кэшируются отдельно от NULLXES AVATAR layout.

## References

- [`GTM_V2_HF_MODEL_APPROVAL.md`](GTM_V2_HF_MODEL_APPROVAL.md)
- [`GTM_NIGHT_FURY_V2_DIRECTIVE.md`](GTM_NIGHT_FURY_V2_DIRECTIVE.md)

---

**NULLXES** · ADR V2 speech
