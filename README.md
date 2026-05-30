<div align="center">

# ARACHNE-X

**Hyper-realistic realtime digital-human infrastructure**

### BY NULLXES LLC TEAM

[![Architecture](https://img.shields.io/badge/Architecture-13.6B%20ACV--DiT-brightgreen)](ARCHITECTURE.md)
[![Deploy](https://img.shields.io/badge/Deploy-RunPod%20H200%2FH100-blue)](Documentation/NULLXES_ARACHNE_RUNPOD_27-05-2026.md)
[![License](https://img.shields.io/badge/License-NULLXES%20Proprietary%202.0-red)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/MagistrTheOne/ARACHNE-X-NULLXES-)

**Contact:** [ceo@nullxes.com](mailto:ceo@nullxes.com) · Telegram [@MagistrTheOne](https://t.me/MagistrTheOne)

</div>

---

## Overview

**ARACHNE-X ULTRA** is NULLXES operational infrastructure for realtime avatar generation: audio-conditioned 13.6B ACV-DiT, chunked streaming inference, explicit GPU worker admission control, and a single canonical WebSocket orchestration path.

Production weights are **NULLXES proprietary** — not public LongCat checkpoints.

| Resource | Link |
|----------|------|
| Architecture (policy) | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| RunPod deployment | [`Documentation/NULLXES_ARACHNE_RUNPOD_27-05-2026.md`](Documentation/NULLXES_ARACHNE_RUNPOD_27-05-2026.md) |
| Dependencies | [`Documentation/REQUIREMENTS.md`](Documentation/REQUIREMENTS.md) |
| ULTRA-AVATAR weights | [huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-AVATAR](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-AVATAR) |
| ULTRA-VIDEO weights | [huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-VIDEO](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-VIDEO) |

---

## What ships today (2026-05-27)

| Layer | Role |
|-------|------|
| **Foundation DiT** | 13.6B ACV-DiT · Wan VAE · UMT5 · Wav2Vec2 audio conditioning |
| **Streaming inference** | `generate_streaming_ai2v()` · incremental wav2vec · TTFF-first chunked denoise |
| **Stability OS** | Cross-chunk KV · identity drift monitor · silence gate |
| **GPU worker** | `services/arachnex-worker` · explicit queue · `/health` · `/v1/runtime/metrics` |
| **Orchestrator** | `src/server/*` · STT → LLM → TTS → NDJSON avatar stream · WS `protocolVersion: v1` |

**Process isolation:** TTS and LLM live in the orchestrator. The GPU worker runs DiT inference only.

---

## Production realtime path

```text
WebSocket (chat.send / voice.pcm16)
  → src/server/realtime_api.py
  → src/server/session_worker.py
  → src/server/realtime_avatar_loop.py
  → src/server/avatar_stream_client.py
  → services/arachnex-worker  POST /v1/realtime/avatar_frames
  → arachne_x/runtime/avatar_serving.py  →  generate_streaming_ai2v()
  → WS avatar.stream.chunk
```

Details, env vars, and endpoints: [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## System requirements

| Component | Requirement |
|-----------|-------------|
| GPU (prod) | NVIDIA H200 / H100 / A100 |
| CUDA (wheel) | 12.4 (`torch 2.6.0+cu124`) |
| Python | 3.10 or 3.11 (3.10 on RunPod) |
| OS (infer) | Linux (RunPod). Windows = SSH client only |
| VRAM | ~110–120 GB full avatar runtime on H200 class |

---

## Quick start (RunPod)

Full step-by-step: [`Documentation/NULLXES_ARACHNE_RUNPOD_27-05-2026.md`](Documentation/NULLXES_ARACHNE_RUNPOD_27-05-2026.md).

```bash
git clone https://github.com/MagistrTheOne/ARACHNE-X-NULLXES-.git
cd ARACHNE-X
git checkout arachne-last-patch

python3.10 -m venv .venv && source .venv/bin/activate

# 1. torch + flash-attn (Linux only — see RunPod doc §4)
pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
MAX_JOBS=8 pip install flash-attn==2.7.4.post1 --no-build-isolation

# 2. Core stack + worker HTTP
pip install -r requirements_avatar.txt
pip install -r services/arachnex-worker/requirements.txt

export PYTHONPATH="$PWD"
export NULLXES_CHECKPOINT_DIR=/path/to/merged/checkpoint_dir

# 3. Smoke (offline ai2v)
python scripts/infer.py --mode ai2v --audio path/to.wav --image path/to.jpg \
  --output /tmp/smoke.mp4 --profile operational

# 4. Worker (realtime NDJSON)
cd services/arachnex-worker && uvicorn main:app --host 0.0.0.0 --port 9090
```

Orchestrator (CPU gateway): `pip install -r requirements_orchestrator.txt` — see RunPod doc §8.

In-tree TTS (`arachne_x.tts` / `arachne_x.speech`) was removed. Provide speech audio externally: pass a WAV via `--audio` (CLI) or pre-rendered PCM (orchestrator); wire an external TTS in `src/server/tts_runner.py`.

---

## Dependencies

| File | Use |
|------|-----|
| `requirements.txt` | Core GPU stack (after torch + flash-attn) |
| `requirements_avatar.txt` | Worker / `infer.py` |
| `requirements_orchestrator.txt` | `src/server` gateway (aiohttp, whisper) |
| `requirements-training.txt` | Latent export / WebDataset (not prod infer) |

Install order is strict. See [`Documentation/REQUIREMENTS.md`](Documentation/REQUIREMENTS.md).

---

## Repository layout

```text
arachne_x/              # DiT pipeline, loader, avatar_serving, inference
src/server/             # Realtime orchestrator (WebSocket gateway)
services/arachnex-worker/   # GPU HTTP worker (dumb inference)
scripts/infer.py        # CLI entry (ai2v, streaming, VIDEO modes)
Documentation/          # RunPod guide, requirements, schemas
```

---

## Frontend ↔ backend contract

Provider secrets (RunPod API keys, internal worker URLs) stay on the backend. The frontend talks only to your gateway.

**Recommended stack:** STT → LLM → TTS → ARACHNE avatar stream.

| Endpoint | Purpose |
|----------|---------|
| `POST /api/avatar/session` | Create session or generation request |
| `GET /api/avatar/jobs/:jobId` | Poll async job status |
| `POST /api/avatar/stop` | Interrupt / cleanup |

Schema and WS events: [`Documentation/ARACHNE_AVATAR_STT_LLM_TTS_SCHEMA.md`](Documentation/ARACHNE_AVATAR_STT_LLM_TTS_SCHEMA.md).

**Never expose from frontend:** `RUNPOD_API_KEY`, inference keys, raw worker URLs.

---

## License

NULLXES Proprietary License 2.0 — see [`LICENSE`](LICENSE).  
Unauthorized modification, redistribution, or derivative works are prohibited.

---

<div align="center">

**ARACHNE-X BY NULLXES LLC TEAM** · © 2026 NULLXES

</div>
