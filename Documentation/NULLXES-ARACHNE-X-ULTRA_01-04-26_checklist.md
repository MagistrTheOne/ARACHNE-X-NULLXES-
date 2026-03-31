# NULLXES-ARACHNE-X-ULTRA — checklist

**Дата:** 01.04.2026  
**Автор:** MagistrTheOne  

Документ для поэтапной приёмки и эксплуатации стека ULTRA (базовое видео + аватар + TTS + стриминг). Отмечайте пункты по мере готовности.

---

## 1. Окружение и веса

- [ ] Python-окружение и `PYTHONPATH` на корень репозитория
- [ ] `requirements.txt` установлен; при TTS — дополнительно `requirements-tts.txt`
- [ ] GPU / CUDA пригодны для выбранного режима (VRAM под чекпоинт)
- [ ] Локальный `checkpoint_dir` в формате `WeightsLayout` (`tokenizer/`, `text_encoder/`, `vae/`, `scheduler/`, `dit/` или `avatar_single/` / `avatar_multi/`, `audio/wav2vec2/`)
- [ ] При необходимости Hub: `HF_TOKEN`, `--allow_hub_download`, кэш `--weights_cache_dir`

---

## 2. Базовый пайплайн (без аватара)

- [ ] `t2v`: короткий прогон, файл без аудиодорожки (ожидаемо для base)
- [ ] `i2v`: изображение + промпт
- [ ] `vc`: продолжение с контекстом видео

Справка: [`scripts/infer.py`](../scripts/infer.py), [INFERENCE_MODES_AND_LAUNCH.md](INFERENCE_MODES_AND_LAUNCH.md).

---

## 3. Аватар (аудио как conditioning)

- [ ] `ai2v`: референс-изображение + `--audio` + mp4 с mux через `save_video_ffmpeg`
- [ ] `at2v`: только аудио + промпт
- [ ] `avc`: видео-контекст + аудио; при multi — веса `avatar_multi/`
- [ ] `streaming_ai2v`: стрим чанков аудио; согласован `--audio_chunk_sec` с микро-ходами оркестратора
- [ ] `enroll_identity` при использовании identity bank
- [ ] Опционально: LoRA (`--lora_path`, метаданные / CLI rank-alpha)

---

## 4. Text → TTS → аватар → mp4

- [ ] Установлен `qwen-tts` и `soundfile` (или выбранный в будущем провайдер)
- [ ] `ai2v` / `at2v` / `avc` / `streaming_ai2v` с `--speak_text` без `--audio` (или приоритет `--audio`, если оба заданы)
- [ ] Проверены `--tts_language`, `--tts_speaker`, при необходимости `--tts_instruct`, `--tts_device_map`
- [ ] Итоговый mp4 содержит звук, длительность согласована с обрезкой в `save_video_ffmpeg`

Справка: [INFERENCE_MODES_AND_LAUNCH.md — раздел TTS](INFERENCE_MODES_AND_LAUNCH.md), [`arachne_x/tts/`](../arachne_x/tts/).

---

## 5. Обучение и данные (по необходимости)

- [ ] `scripts/train.py`: smoke `base` / `avatar` на предагментированных латентах
- [ ] `scripts/train_lora_avatar.py`: короткий прогон LoRA → загрузка в `infer.py`
- [ ] `scripts/export_latent_training_sample.py`: один сэмпл под формат датасета

---

## 6. Реалтайм и продакшн

- [ ] Согласован контракт: **аудио — master clock**; один чанк TTS/PCM = один микро-ход аватара (`arachne_x/tts/realtime.py`, `chunking.py`)
- [ ] WebRTC / BFF: при появлении `src/server` — конфиг `pipeline_config.json` (vad, asr, llm, tts, avatar) и `scripts/run_webrtc_server.py`
- [ ] Логирование, лимиты параллелизма, отмена генерации (barge-in) по архитектурным докам при необходимости

---

## 7. Известные ограничения (не блокеры чеклиста, но в уме)

- [ ] FP8 в стриминге: заглушка / предупреждение в `config_realtime` — не считать готовым prod-квантованием DiT без доработки
- [ ] Импорт `import arachne_x` тянет тяжёлый граф (loader); для утилит по возможности узкие импорты

---

 
