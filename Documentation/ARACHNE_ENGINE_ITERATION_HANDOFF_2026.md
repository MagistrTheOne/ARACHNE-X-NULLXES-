# ARACHNE-X Engine — Iteration Handoff (GPT ↔ NULLXES)

**Scope:** только репозиторий `ARACHNE-X/` (движок: `arachne_x/`, `scripts/infer.py`, worker, train/export).  
**Вне scope:** монорепо HR AI, LiveKit, gateway, landing, kaira-agent — behavior layer снаружи.

**Цель этапа:** довести движок до **operational realtime avatar OS**, не «ещё один Comfy lipsync workflow».

**Связанные доки:** [`ARCHITECTURE.md`](../ARCHITECTURE.md) · [`REQUIREMENTS.md`](REQUIREMENTS.md) · [`ARACHNE_X_CLASSIFICATION_2026-05-21.md`](ARACHNE_X_CLASSIFICATION_2026-05-21.md)

---

## 1. Что это за система (одним блоком)

```text
ACV-DiT Runtime =
  Frozen NULLXES ULTRA-AVATAR weights (~13.6B, 48×4096)
  + audio cross-attn inside every DiT block (not external ControlNet)
  + Wan VAE latents + flow-match Euler scheduler
  + UMT5 text + Wav2Vec2 → AudioProj (32 tokens / frame)
  + identity bank + attention-only character LoRA
  + (planned) motion adapter · chunked sampling OS
```

**Production weights (не LongCat Hub runtime):**

- [MagistrTheOne/ARACHNE-X-ULTRA-AVATAR](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-AVATAR)
- [MagistrTheOne/ARACHNE-X-ULTRA-VIDEO](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-VIDEO)

**Lineage:** архитектурный класс 3D ACV-DiT (совместимость tensor ABI с public LongCat-Video reports). Веса — **независимо обученные NULLXES**.

---

## 2. Главный bottleneck (подтверждено кодом)

### ❌ Monolithic full-volume denoise

Каждый `ai2v` / batch path:

```text
FOR step in 1..S:
    FOR cfg_branch in {uncond, text-only, full-audio [, emotion]}:
        full DiT forward (48 blocks, all latent frames T×H×W)
    scheduler.step on latents[:, :, 1:]
VAE.decode(entire volume)
```

**Сложность (порядок):** `O(S × CFG × T_frames × R² × 48 × 13.6B-attn)`

| Множитель | Типичное prod | Где в коде |
|-----------|---------------|------------|
| **S** steps | 25–35 cinematic; worker default **8** | `num_inference_steps` |
| **CFG** passes | **3–4** DiT forwards / step | `generate_ai2v` denoise loop, `_predict_avatar_noise` |
| **T** frames | `num_frames` (правило **4n+1**) | `inference_frames.py` |
| **R** resolution | 480p / 720p | `get_condition_shape` |
| **Distill σ** | **OFF** в batch `ai2v` из CLI | `use_distill` не проброшен в `inference_engine` |
| **KV reuse** | **OFF** в `ai2v` | `forward_with_kv_cache` в основном `avc` |

### ❌ Fake streaming

`generate_streaming_ai2v`:

1. Склеивает весь audio stream  
2. Вызывает **полный** `generate_ai2v(..., output_type="latent")`  
3. Только **VAE** отдаёт кадры по одному (`StreamingVAEDecoder`)

→ Stream decode ≠ stream denoise. InfiniteTalk-стиль chunk denoise **ещё не prod path**.

---

## 3. Что уже есть в коде (не изобретать заново)

| Capability | Status | Files |
|------------|--------|-------|
| Triple / quad CFG (text + audio) | ✅ prod | `pipeline_arachne_x_video_avatar.py` ~2550–2630 |
| `use_distill` + 50 distill indices | ✅ pipeline, частично wired | `get_timesteps_sigmas`, auto if `steps≤16` in streaming latent path |
| KV cache attention | ✅ partial | `modules/avatar/attention.py` `forward_with_kv_cache` |
| Identity bank 1024 slots | ✅ | pipeline + `enroll_identity` |
| Attention-only LoRA policy | ✅ locked | `modules/lora_utils.py`, `train_lora_avatar.py` |
| BSA infer (train = dense) | ✅ policy | `infer_attention.py`, `disable_bsa()` on train |
| Windowed audio emb | ✅ train≡infer | `inference_audio.py` |
| Prompt compiler (pre-UMT5) | ✅ optional | `prompt_compiler/` |
| Hybrid mouth renderer | ✅ optional | pipeline flags |
| Motion Adapter | ⬜ reserved | `ARCHITECTURE.md` §3 |
| Chunked denoise + stitch | ⬜ **нет** | — |
| `runtime_profile` operational/cinematic | ⬜ **нет** | — |
| CFG cache / reduced forwards | ⬜ **нет** | — |

---

## 4. Два runtime (целевое разделение — ещё не formalized)

| Profile | Steps | Frames | Denoise | Use |
|---------|-------|--------|---------|-----|
| **Operational** | 8–12 + `use_distill=true` | chunk 17–49, overlap | **chunked** (target) | worker, micro-turn, LiveKit path |
| **Cinematic** | 25–35, full σ | full clip sync/duration | **monolithic** (сейчас) | export, gold eval |

**Baseline wall-clock (H200, monolithic, из classification card):**

| ~frames | steps | ~denoise wall |
|---------|-------|----------------|
| 97 | 25 | ~31 min |
| 113 | 25 | ~41 min |

→ Bottleneck = **topology**, не «слабый GPU».

---

## 5. ROI roadmap (не premature)

**Делать первым (infer / topology, без full foundation retrain):**

1. **Chunked denoise** + overlap + anchor latents  
2. **Distill runtime** — проброс `use_distill`, профили steps  
3. **KV cross-chunk reuse** — подключить existing cache к chunk loop  
4. **CFG compute reduction** — cache uncond/text внутри chunk  
5. **Motion residual adapter** — train, frozen DiT  

**Не делать сейчас:**

- Full 13.6B retrain  
- Sparse attention surgery (до стабильного chunk path)  
- Новый backbone / MoE  
- Смешение `requirements-audiodit` (transformers≥5.3) с core venv  

**Ожидаемый выигрыш:** X2–X6 от **flow of compute**, не от «нового magical Wan».

---

## 6. Формат итераций (как кидаем друг другу кейсы)

### Ввод от NULLXES (каждый кейс)

```text
1. Симптом          — что визуально/метрически ломается
2. Конфиг           — mode, steps, frames, resolution, cfg scales, distill, lora, identity bank
3. Ожидание         — что должно было быть
4. Факт             — что вышло (видео / кадр / latency)
5. Артефакт         — лог, tqdm, metrics, кусок кода / line range
6. Железо           — H200/H100, VRAM peak, CUDA, torch, flash-attn
7. Приоритет        — speed | lipsync | identity | realism | realtime (один главный)
```

### Ответ GPT (каждый кейс)

```text
— корневая механика (topology | scheduler | audio | identity | attention | dataset | CUDA)
— infer-only fix vs retrain vs architecture debt
— файлы / функции для правки (только ARACHNE-X)
— что НЕ трогать (LoRA policy, BSA train, ABI class names)
— cheap fix vs Phase 2+ train
```

---

## 7. Карта симптом → гипотеза → куда смотреть

| Симптом | Вероятная механика | Куда в коде | Fix class |
|---------|-------------------|-------------|-----------|
| Долго на любой длине | Monolithic S×CFG×T | `generate_ai2v` loop | topology: chunk + distill |
| Плывёт identity на 8+ s | Нет anchor между chunks | identity bank, `ref_img_index` | chunk anchors + bank reinject |
| Рот дёргается на silence | Audio CFG на нулевой speech | `audio_guidance_scale`, zero audio emb | silence gate / motion adapter |
| Lip mismatch | Window / embedding_fps | `inference_audio.py`, `_build_windowed` | embedding_fps, phoneme path |
| Flicker / temporal noise | BSA train≠infer mismatch | `infer_attention`, train `disable_bsa` | не менять train; eval dense |
| Stream «не realtime» | Full denoise before first frame | `generate_streaming_ai2v` | real chunk denoise |
| Grain / snow на LoRA | High-σ steps | `training_lora_loss` Min-SNR | train policy, не infer |
| OOM 720p long T | Full volume attention | reduce T, chunk, 480p op profile | operational profile |

---

## 8. Touchpoints по файлам (engine only)

| Задача | Primary files |
|--------|----------------|
| Denoise loop / CFG | `arachne_x/pipeline_arachne_x_video_avatar.py` |
| DiT forward | `arachne_x/modules/avatar/arachne_avatar_dit.py`, `attention.py` |
| CLI / prod entry | `scripts/infer.py`, `arachne_x/runtime/inference_engine.py` |
| Worker serve | `arachne_x/runtime/avatar_serving.py`, `services/arachnex-worker/` |
| Frame budget | `arachne_x/inference_frames.py` |
| Audio emb | `arachne_x/inference_audio.py`, `pipeline` `get_audio_embedding` |
| Distill schedule | `get_timesteps_sigmas`, `use_distill` kwargs |
| KV cache | `attention.py` `forward_with_kv_cache` |
| Stream decode only | `arachne_x/streaming_inference.py` `StreamingVAEDecoder` |
| LoRA train | `scripts/train_lora_avatar.py`, `training_lora_loss.py` |
| Weights load | `arachne_x/loader.py`, `weights_resolve.py` |
| Semiauto presets (use_distill) | `arachne_x/orchestration/presets.py` |

---

## 9. Первая вводная (baseline для итерации #0)

Скопируй в GPT как стартовый кейс:

```text
Repo: ARACHNE-X only
GPU: H200
Checkpoint: NULLXES_CHECKPOINT_DIR → merged ARACHNE-X-ULTRA-AVATAR
Mode: ai2v (monolithic)
Config:
  resolution: 720p
  num_frames_mode: sync (или duration — указать)
  num_frames: ~97–165
  num_inference_steps: 25–35
  use_distill: false (inference_engine не передаёт)
  text_guidance_scale: 4.0
  audio_guidance_scale: 5.0–5.5
  identity_bank: optional .pt

Симптом:
  wall-clock denoise ~30–40 min на ~3–6 s video
  streaming_ai2v не снижает time-to-first-frame (full denoise first)

Ожидание:
  operational path < few min или chunk TTFF < 1–2 s

Факт:
  O(S × 3 × full DiT × T) per clip

Приоритет: speed + realtime (без потери lipsync/identity на eval gate)

Вопрос GPT:
  1) минимальный wedge (infer-only) за 1 sprint
  2) что включить в chunk MVP (overlap, KV, distill, anchors)
  3) что не трогать
```

---

## 9.1 Sprint 1 — Sampling OS (done criteria)

| Item | Status |
|------|--------|
| `sampling_profiles.py` + `--runtime_profile` | shipped |
| `use_distill` wired in `inference_engine` + streaming | shipped |
| `generate_chunked_ai2v` + `chunk_stitch` | shipped |
| `generate_streaming_ai2v` → chunked TTFF path | shipped |
| `RuntimeSamplingMetrics` in `.run.json` | shipped |
| Worker `/v1/realtime/avatar_frames` default `operational` | shipped (eval gate on H200 before prod flag day) |
| Cross-chunk KV consume in `generate_ai2v` | deferred (seed only via `ARACHNE_CHUNK_KV`) |

GPU smoke (manual): `scripts/gpu/smoke_operational_profile.sh` on H200 — lipsync/identity vs cinematic baseline.

**Next:** [ARACHNE_STABILITY_OS_SPRINT2.md](ARACHNE_STABILITY_OS_SPRINT2.md) — LiveKit identity/lip stability (этапы `S2-0` … `S2-7`, без стабов).

---

## 10. Session title для GPT

**`ARACHNE-X Sampling OS v1 — Chunked ACV Runtime Iteration`**

Slug: `ARACHNE-SAMPLING-OS-V1`

---

## 11. Policy reminders (binding)

- Foundation DiT **frozen** at train; character = attention LoRA + identity bank only.  
- Train: **BSA off**, dense FlashAttn; infer: BSA env-gated.  
- Do not rename `LongCatVideo*Transformer3DModel` (checkpoint ABI).  
- Prod weights = **MagistrTheOne/ARACHNE-X-ULTRA-***, not meituan-longcat snapshots.  
- `requirements-audiodit.txt` = **separate venv**.

---

**NULLXES · engine handoff · 2026 · paste to GPT and iterate with §6 format**
