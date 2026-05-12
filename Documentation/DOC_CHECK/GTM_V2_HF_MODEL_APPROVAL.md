# V2 — Hugging Face models approval table (SpeechAdapters)

Модели ниже **не** являются NULLXES proprietary VIDEO/AVATAR checkpoints; они задают **интеграционный** speech stack для NIGHT FURY V2.  
Строка считается **approved** только после явного sign-off NULLXES и фиксации ревизии в `requirements*.txt`.

| Role | Suggested `repo_id` | License (verify on card) | Revision / pin status | Approved (Y/N) |
|------|----------------------|----------------------------|------------------------|----------------|
| ASR | `openai/whisper-large-v3-turbo` | MIT (verify on HF) | pending pin with `faster-whisper` / CTranslate2 | |
| Emotion | `emotion2vec/emotion2vec_base` | verify on HF | pending integration + pin | |
| TTS | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (and collection) | verify on HF | `qwen-tts` in `requirements-tts.txt` | |

**Процедура**

1. Юридический/продуктовый просмотр карточки каждого `repo_id`.
2. Зафиксировать **commit hash** или **snapshot revision** в internal реестре (не только floating `main`).
3. После `Y` в колонке Approved — обновить пины в [`requirements.txt`](../../requirements.txt) / [`requirements_avatar.txt`](../../requirements_avatar.txt) / [`requirements-tts.txt`](../../requirements-tts.txt) согласно [`GTM_V2_RUNPOD_DEP_MATRIX.md`](GTM_V2_RUNPOD_DEP_MATRIX.md).

---

**NULLXES** · approval workflow
