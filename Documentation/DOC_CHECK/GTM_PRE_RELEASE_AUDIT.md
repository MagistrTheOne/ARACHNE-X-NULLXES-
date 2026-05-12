# GTM Pre-Release Audit — ARACHNE-X-ULTRA V2 NIGHT FURY

Чеклист перед GTM. Политика среды: **только RunPod Linux GPU** (H200 primary, H100 secondary, B200 experimental). Windows — не production path.

**Идентичность продукта:** **CoreNULLXES** — проприетарный DiT, VAE, runtime, inference contracts, identity/temporal systems. **SpeechAdapters** (не core ownership): Whisper Large V3 Turbo, `facebook/wav2vec2-base` (опциональный адаптер, см. ADR), `emotion2vec_base`, Qwen3-TTS (см. `ADR_V2_SPEECH_STACK.md`).

---

## Runtime

- [ ] Единый runtime facade: `arachne_x.runtime.avatar_serving` — канон кэша пайплайна, NDJSON, MP4 job; воркер проксирует через `services/longcat-worker/gpu_avatar_runtime.py` (lazy), без второй реализации forward.
- [ ] Нет дублированного inference glue вне worker → `avatar_serving` → `loader.load_avatar_pipeline`.
- [ ] Worker → runtime path only: HTTP слой не оркестрирует torchrun subprocess для текущего in-process avatar path.
- [ ] Нет параллельной оркестрации пайплайна в `src/server` для DiT/VAE (оркестратор вызывает HTTP к воркеру).

## GPU

- [ ] CUDA compatibility: образ/хост согласованы с пином `torch` в `requirements.txt` и CUDA-драйвером RunPod.
- [ ] Triton: окружение Triton-compatible (Linux, torch CUDA build); импортный граф задокументирован в `GTM_V2_RUNTIME_AUDIT.md`.
- [ ] xformers policy: зафиксирована в `GTM_V2_RUNPOD_DEP_MATRIX.md` / контракте.
- [ ] flash-attn: пин `requirements.txt` (Linux marker); сборка на целевом GPU классе проверена.
- [ ] No CPU fallback in prod: воркер падает с ошибкой при отсутствии CUDA (`avatar_serving.get_avatar_pipeline`).

## Dependencies

- [ ] Единые версии core: `requirements.txt` + `requirements_avatar.txt` + `docker/Dockerfile.gpu` согласованы.
- [ ] Нет конфликтующих дублей librosa/onnxruntime между основным стеком и optional audiodit venv (политика в `GTM_DEPENDENCY_PURGE.md`).
- [ ] Нет дублирующих деревьев transformers без явной причины.
- [ ] Audiodit isolation: отдельный optional install / venv по `requirements-audiodit.txt`.

## Naming

- [ ] Нет публичного бренда LongCat в пользовательских путях/доках по умолчанию: канон MP4 `POST /v1/arachne/generate`, legacy `/v1/longcat/generate` помечен deprecated.
- [ ] Нет смешанного брендинга в README воркера и OpenAPI описаниях (NULLXES / NIGHT FURY).
- [ ] Каноническая терминология NULLXES в env: `NULLXES_INFERENCE_SERVICE_KEY` (и совместимые алиасы), `NULLXES_CHECKPOINT_DIR`.

## Docs

- [ ] Нет противоречивых описаний архитектуры: топология worker ↔ `arachne_x.runtime` согласована с `GTM_PRODUCTION_CONTRACT.md` и `ARACHNE-X_ARCHITECTURE_SPEC_NULLXES.md`.
- [ ] Все ссылки на inference HTTP согласованы с `GTM_SCHEMA_TRUTH_INFERENCE_HTTP.md`.
- [ ] Checkpoint ownership везде описан как NULLXES bundle + HF ids в манифестах апрува (`GTM_V2_HF_MODEL_APPROVAL.md`).

## Deployment

- [ ] RunPod-only policy задокументирована (`GTM_ONE_SHOT_DEPLOY.md`, матрица деплоя).
- [ ] H200 startup path проверен (smoke + health).
- [ ] B200 явно помечен experimental в матрице.
- [ ] One-shot deployment flow описан (`GTM_ONE_SHOT_DEPLOY.md`).

## Eval

- [ ] E-LIPS / E-ID / E-TEMP / E-MOS процедуры зафиксированы (`GTM_DATA_EVAL.md`).
- [ ] Realtime NDJSON stress в манифесте merge/eval.

## Security / Stability

- [ ] Нет случайного mock mode в prod: моки воркера только при явном dev-флаге (см. README воркера / код).
- [ ] Скрытые dev-флаги не включены в production env по умолчанию.
- [ ] Детерминированный startup path: health до первого тяжёлого GPU load; ленивая загрузка пайплайна при первом inference.
- [ ] Lazy GPU loading проверен: импорт воркера не тянет `arachne_x` до первого GPU-вызова.

## Известные ограничения (не блокируют GTM, но требуют тикета)

- [ ] Поле `negativePrompt` в NDJSON-стриме принимается на границе HTTP, но **streaming path** в `generate_streaming_ai2v` сегодня передаёт в `generate_ai2v` фиксированный `negative_prompt=""` — изменение потребует согласования inference semantics.
