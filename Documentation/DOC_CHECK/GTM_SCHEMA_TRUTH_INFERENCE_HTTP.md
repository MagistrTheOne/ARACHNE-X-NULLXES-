# Schema truth: HTTP worker bodies ↔ logical InferenceJob

Канонический программный контракт описан в [`GTM_PRODUCTION_CONTRACT.md`](GTM_PRODUCTION_CONTRACT.md) §5. Ниже — соответствие полей **Inference Worker** ([`services/longcat-worker/main.py`](../../services/longcat-worker/main.py)) логическим полям `InferenceJob` для V2 **NIGHT FURY**.

## `StreamFramesBody` ↔ streaming avatar job

| HTTP field (`StreamFramesBody`) | `InferenceJob` (logical) | Notes |
|----------------------------------|---------------------------|-------|
| `sessionId` | `session_id` | Корреляция логов; SessionMemory ключ |
| `prompt` | `prompt` | |
| `imageBase64` | `image_bytes` (decoded) | Не путь к файлу |
| `audioPcm16Base64` / `audioFloat32Base64` | `audio_pcm_f32` (decoded mono 16 kHz) | Ровно одно из двух обязательно |
| `negativePrompt` | `negative_prompt` | Принимается на HTTP границе; **streaming** сегодня не меняет `generate_ai2v` negative (см. `generate_streaming_ai2v` → фиксированный пустой negative) — зафиксировано в `GTM_PRE_RELEASE_AUDIT.md` как ограничение до отдельного согласования semantics |
| `numInferenceSteps` | `num_inference_steps` | |
| `textGuidanceScale` | `text_guidance_scale` | |
| `audioGuidanceScale` | `audio_guidance_scale` | |
| `resolution` | `resolution` | `480p` / `720p` |
| `numFrames` | `num_frames` | |
| `engine` | `engine` | Фильтр воркера; core: `arachne` / `nullxes` / пусто / legacy `longcat`; не часть DiT forward |

## `GenerateBody` ↔ legacy MP4 job (audio-* tasks)

| HTTP field | `InferenceJob` | Notes |
|------------|----------------|-------|
| `task` | `mode` | `audio-image-to-video` → логический `ai2v`-class job с image+audio |
| `prompt` | `prompt` | |
| `imageBase64` | temp file path после decode | Реализация в `run_generate_sync` |
| `audioBase64` | temp wav path | MODE A: приоритет над `speakText`, если оба заданы |
| `speakText` | `speak_text` | MODE B: TTS → WAV (`ttsProvider`, детали в `inputJson`) |
| `ttsProvider` | `tts_provider` | По умолчанию `qwen` |
| `embedAudio` | `embed_audio` | По умолчанию `true`: финальный MP4 с AAC (ffmpeg mux через [`mp4_export.py`](../../arachne_x/runtime/mp4_export.py)) |
| `outputMode` | `output_mode` | Только `mp4` |
| `numInferenceSteps` / `textGuidanceScale` / `audioGuidanceScale` / `resolution` / `numFrames` | generation knobs | Прокидываются в `generate_frames_numpy` |
| `negative_prompt` | `negative_prompt` | |
| `inputJson` | `input_json` | TTS overrides (`ttsModel`, `ttsDeviceMap`, audiodit-поля и т.д.) |

**MODE C** (голос → текст промпт): отдельный оркестраторный слой (Whisper), не поля `GenerateBody` в этой версии.

## Исполняемый путь (V2 target)

Все GPU пути аватара из воркера должны вызывать реализацию в [`arachne_x/runtime/avatar_serving.py`](../../arachne_x/runtime/avatar_serving.py) (единый модуль с [`arachne_x/runtime/inference_engine.py`](../../arachne_x/runtime/inference_engine.py) для CLI).

---

**NULLXES** · schema truth · NIGHT FURY V2
