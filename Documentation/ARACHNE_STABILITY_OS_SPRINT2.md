# ARACHNE-X Stability OS — Sprint 2 Plan

**Цель:** LiveKit-ready avatar — стабильная identity, контролируемый рот, предсказуемые chunk’и.  
**Пререквизит:** [Sampling OS Sprint 1](ARACHNE_ENGINE_ITERATION_HANDOFF_2026.md#91-sprint-1--sampling-os-done-criteria) (merged).  
**Политика:** без mock MP4, без fake streaming, без seed-only KV без consume, без `update_identity_bank` в realtime.

**Трекер:** [ARACHNE_ITERATION_ROADMAP.md](ARACHNE_ITERATION_ROADMAP.md)

---

## Live checklist (обновлять в PR)

- [ ] **S2-0** H200 eval + KV debt
- [ ] **S2-1** KV consume в `generate_ai2v`
- [ ] **S2-2** Per-chunk identity refresh
- [ ] **S2-3** Silence / audio motion gate
- [ ] **S2-4** Drift monitor + corrective policy
- [ ] **S2-5** Mouth mask + hybrid worker default
- [ ] **S2-6** LiveKit worker schema
- [ ] **S2-7** `eval_stability_bench.py` green

---

## Этапы (overview)

| Этап | ID | Содержание | Оценка |
|------|-----|------------|--------|
| **0** | `S2-0` | Закрыть долги Sprint 1 (KV seed-only, H200 eval gate) | 0.5–1 д |
| **1** | `S2-1` | Cross-chunk KV consume в `generate_ai2v` | 2–3 д |
| **2** | `S2-2` | Per-chunk identity reinjection | 1 д |
| **3** | `S2-3` | Silence / audio motion gate | 1–2 д |
| **4** | `S2-4` | Identity drift monitor + corrective policy | 2 д |
| **5** | `S2-5` | Mouth motion budget (hybrid + worker contract) | 1 д |
| **6** | `S2-6` | Worker / LiveKit schema + session bank | 1–2 д |
| **7** | `S2-7` | GPU eval bench + merge gate | 1 д |

**Итого:** ~7–10 инженерных дней + H200 eval.

---

## Этап 0 — Долги Sprint 1 (`S2-0`)

| Task | ID | DoD |
|------|-----|-----|
| H200 smoke operational vs cinematic | `S2-0-EVAL` | `scripts/gpu/smoke_operational_profile.sh` green; lipsync/identity ≥ cinematic −ε |
| KV seed-only: disable or complete | `S2-0-KV` | Либо удалить `ARACHNE_CHUNK_KV` из prod docs до `S2-1`, либо merge `S2-1` в том же PR |
| Worker default `operational` | `S2-0-WORKER` | Включить только после `S2-0-EVAL` |

---

## Этап 1 — Cross-chunk continuity (`S2-1`)

**Проблема:** каждый chunk = независимый `generate_ai2v`; overlap только pixel stitch.

| Task | ID | Файлы | DoD |
|------|-----|-------|-----|
| `use_kv_cache` path в `generate_ai2v` | `S2-1-AI2V-KV` | `pipeline_arachne_x_video_avatar.py` | Контракт как `generate_avc`: `_cache_clean_latents`, `kv_cache_dict` в `_predict_avatar_noise` |
| Chunk tail → seed KV | `S2-1-CHUNK-SEED` | `chunk_kv.py`, `generate_chunked_ai2v` | После chunk i: encode overlap tail → KV |
| Chunk i+1 consume KV | `S2-1-CHUNK-CONSUME` | `generate_chunked_ai2v` | `use_kv_cache=True`, denoise только новые latent frames |
| Trim temporal KV | `S2-1-TRIM` | `_compress_kv_cache_dict_temporal` | `kv_keep_last=24` config |
| Metrics | `S2-1-METRICS` | `sampling_metrics.py` | `kv_cache_hits`, `cross_chunk_kv_frames` в `.run.json` |

**Не делать:** full latent volume между chunk’ами.

---

## Этап 2 — Per-chunk identity (`S2-2`)

| Task | ID | Файлы | DoD |
|------|-----|-------|-----|
| `_refresh_identity_tokens` каждый chunk | `S2-2-REFRESH` | `generate_chunked_ai2v` | Без `update_identity_bank` |
| Тот же `image` + `identity_id` | `S2-2-ANCHOR` | `inference_engine`, worker | Уже в Sprint 1; проверить streaming path |
| Запрет bank update в live | `S2-2-GUARD` | `avatar_serving`, worker | Assert / ignore `update_identity_bank` в realtime jobs |

---

## Этап 3 — Silence / audio motion gate (`S2-3`)

| Task | ID | Файлы | DoD |
|------|-----|-------|-----|
| `audio_motion_gate` модуль | `S2-3-GATE` | `runtime/audio_motion_gate.py` | `silence_ratio`, `effective_audio_guidance_scale` |
| Wire в denoise | `S2-3-WIRE` | `generate_ai2v`, chunked | Масштаб только audio CFG branch |
| Operational audio scale | `S2-3-SCALE` | `sampling_profiles`, worker | `audio_guidance_scale` 5.0–5.5 после eval |

---

## Этап 4 — Identity drift monitor (`S2-4`)

| Task | ID | Файлы | DoD |
|------|-----|-------|-----|
| Face ROI cosine vs anchor | `S2-4-MONITOR` | `runtime/identity_drift_monitor.py` | Per-chunk similarity, no new foundation weights |
| Metrics export | `S2-4-METRICS` | `sampling_metrics.py`, `write_run_metadata` | `identity_cosine_per_chunk`, `identity_drift_max` |
| Corrective policy | `S2-4-POLICY` | `generate_chunked_ai2v` | if cosine < τ → refresh tokens + reduce audio CFG 15% |

---

## Этап 5 — Mouth motion budget (`S2-5`)

| Task | ID | Файлы | DoD |
|------|-----|-------|-----|
| Worker mouth_mask contract | `S2-5-MASK` | `main.py`, `avatar_serving` | Optional `mouthMaskBase64` |
| Hybrid default when mask | `S2-5-HYBRID` | `avatar_serving` | `hybrid_renderer_enabled` if mask present |
| Bench A/B with mask | `S2-5-EVAL` | `scripts/gpu/eval_stability_bench.py` | Cheeks/eyes stable vs no mask |

---

## Этап 6 — LiveKit worker contract (`S2-6`)

| Task | ID | Файлы | DoD |
|------|-----|-------|-----|
| Session `identity_id` + bank preload | `S2-6-SESSION` | worker, `avatar_serving` | One bank load per checkpoint |
| NDJSON debug meta (optional) | `S2-6-NDJSON` | worker | Sampling metrics snapshot every N frames |
| Gateway schema doc | `S2-6-SCHEMA` | `GTM_SCHEMA_TRUTH_*` | `identityId`, `runtimeProfile`, `silenceGate` |

---

## Этап 7 — Eval gate (`S2-7`)

| Task | ID | DoD |
|------|-----|-----|
| `scripts/gpu/eval_stability_bench.py` | `S2-7-BENCH` | Real infer; JSON report |
| Thresholds | `S2-7-GATE` | `ttff_sec` < 4s; `identity_cosine_min` > τ; lipsync ≥ cinematic −ε |
| Merge criterion | `S2-7-MERGE` | Worker operational default only if bench green |

---

## Sprint 3+ (out of scope Sprint 2)

| Sprint | Содержание |
|--------|------------|
| **3** | Motion Adapter (train + infer `motion_scale`) |
| **4** | CFG text cache, distill LoRA, FP8 ckpt |

---

## Explicit DO-NOT

- Mock / fake frames в prod path  
- `update_identity_bank` в realtime  
- LoRA на `ffn` / `audio_proj` для «стабильности»  
- Full latent carry между chunks  
- Оставлять `ARACHNE_CHUNK_KV=1` как рабочий без `S2-1-CHUNK-CONSUME`  
- Motion через identity LoRA  

---

## Tracking (для PR / issues)

Копировать в issue title: `[S2-1-AI2V-KV] …`

```
S2-0-EVAL  S2-0-KV  S2-0-WORKER
S2-1-AI2V-KV  S2-1-CHUNK-SEED  S2-1-CHUNK-CONSUME  S2-1-TRIM  S2-1-METRICS
S2-2-REFRESH  S2-2-ANCHOR  S2-2-GUARD
S2-3-GATE  S2-3-WIRE  S2-3-SCALE
S2-4-MONITOR  S2-4-METRICS  S2-4-POLICY
S2-5-MASK  S2-5-HYBRID  S2-5-EVAL
S2-6-SESSION  S2-6-NDJSON  S2-6-SCHEMA
S2-7-BENCH  S2-7-GATE  S2-7-MERGE
```

---

**NULLXES · Stability OS Sprint 2 · 2026**
