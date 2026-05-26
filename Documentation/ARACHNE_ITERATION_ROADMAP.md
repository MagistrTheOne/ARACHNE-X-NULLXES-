# ARACHNE-X — Iteration Roadmap (NULLXES)

Живой трекер итераций. Обновлять при merge каждого этапа.  
**Политика:** без mock MP4, без fake streaming, без seed-only KV без consume, без `update_identity_bank` в realtime.

| Документ | Роль |
|----------|------|
| [ARACHNE_ENGINE_ITERATION_HANDOFF_2026.md](ARACHNE_ENGINE_ITERATION_HANDOFF_2026.md) | GPT handoff, symptom → fix |
| [ARACHNE_STABILITY_OS_SPRINT2.md](ARACHNE_STABILITY_OS_SPRINT2.md) | Sprint 2 детальный план |
| [ARCHITECTURE.md](../ARCHITECTURE.md) | Binding policy |

**Ветка итераций:** `arachne-last-patch` → `origin` ([ARACHNE-X-NULLXES](https://github.com/MagistrTheOne/ARACHNE-X-NULLXES-.git))

---

## Sprint 1 — Sampling OS ✅ (shipped)

| ID | Task | Status |
|----|------|--------|
| S1-1 | `sampling_profiles.py` + `--runtime_profile` | done |
| S1-2 | `use_distill` + frame cap в `inference_engine` | done |
| S1-3 | `chunk_stitch.py` + unit tests | done |
| S1-4 | `generate_chunked_ai2v` | done |
| S1-5 | `generate_streaming_ai2v` → chunked; `ARACHNE_LEGACY_STREAMING` | done |
| S1-6 | `RuntimeSamplingMetrics` → `.run.json` | done |
| S1-7 | Worker NDJSON fields + `avatar_serving` operational default | done |
| S1-8 | `scripts/gpu/smoke_operational_profile.sh` | done (manual H200) |
| S1-9 | KV seed-only (`chunk_kv.py`) | **debt → S2-0 / S2-1** |

---

## Sprint 2 — Stability OS 🔄 (code landed; H200 gate pending)

**Цель:** LiveKit-ready — identity не плывёт, рот под контролем, chunk chain стабилен.

| Этап | ID | Статус | PR / commit |
|------|-----|--------|-------------|
| 0 | [S2-0](ARACHNE_STABILITY_OS_SPRINT2.md#этап-0--долги-sprint-1-s2-0) | skipped (user H200 run) | |
| 1 | [S2-1](ARACHNE_STABILITY_OS_SPRINT2.md#этап-1--cross-chunk-continuity-s2-1) | code done | |
| 2 | [S2-2](ARACHNE_STABILITY_OS_SPRINT2.md#этап-2--per-chunk-identity-s2-2) | code done | |
| 3 | [S2-3](ARACHNE_STABILITY_OS_SPRINT2.md#этап-3--silence--audio-motion-gate-s2-3) | code done | |
| 4 | [S2-4](ARACHNE_STABILITY_OS_SPRINT2.md#этап-4--identity-drift-monitor-s2-4) | code done | |
| 5 | [S2-5](ARACHNE_STABILITY_OS_SPRINT2.md#этап-5--mouth-motion-budget-s2-5) | code done | |
| 6 | [S2-6](ARACHNE_STABILITY_OS_SPRINT2.md#этап-6--livekit-worker-contract-s2-6) | code done | |
| 7 | [S2-7](ARACHNE_STABILITY_OS_SPRINT2.md#этап-7--eval-gate-s2-7) | script done; GPU gate pending | |

### Следующий шаг (с тобой на H200)

1. `scripts/gpu/eval_stability_bench.py` — operational vs cinematic + JSON gate  
2. Визуальный lipsync/identity на internal bench-клипе  
3. При green — worker operational default в prod

---

## Sprint 3 — Motion Adapter (planned)

| ID | Task |
|----|------|
| S3-1 | Train audio motion residual adapter (frozen DiT) |
| S3-2 | Infer `motion_scale`; no identity bank writes |
| S3-3 | Bench: motion ↑, identity cosine не падает |

---

## Sprint 4 — Sampling perf (planned)

| ID | Task |
|----|------|
| S4-1 | CFG text-branch cache |
| S4-2 | NULLXES distill LoRA weights |
| S4-3 | FP8 inference checkpoint |

---

## Порядок merge в main

```
Sprint 1 (done) → S2-0 eval → S2-1..S2-4 (core stability) → S2-5..S2-6 (worker) → S2-7 gate → LiveKit integration
```

---

**NULLXES · iteration roadmap · 2026**
