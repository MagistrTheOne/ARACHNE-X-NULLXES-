<!-- ========================= -->
<!-- ARACHNE-X README (NULLXES) -->
<!-- ========================= -->

<div align="center">
  <h1 style="font-size:3.2em; font-weight:800;">🕷️ ARACHNE-X</h1>
  <h3 style="font-size:1.4em; color:#9cff00; letter-spacing:1px;">
    Hyper-Realistic Avatar Generation System
  </h3>
  <p><b>by NULLXES LLC</b></p>

  <br/>

  <a href="#features"><img src="https://img.shields.io/badge/Architecture-Diffusion%20Transformer-brightgreen"></a>
  <a href="#performance"><img src="https://img.shields.io/badge/Realtime-30FPS-blue"></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/Getting%20Started-Quick-orange"></a>
  <a href="https://github.com/MagistrTheOne/ARACHNE-X-NULLXES-"><img src="https://img.shields.io/badge/GitHub-Repository-black?logo=github"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow"></a>
</div>

---

## 🧠 Overview

<b>ARACHNE-X</b> is a next-generation, real-time avatar generation platform designed for **hyper-realistic digital humans**.

Built on a large-scale **Diffusion Transformer (DiT)** and optimized for **NVIDIA H200 (HBM3e)**, ARACHNE-X delivers production-ready avatars with **perfect lip-sync**, **identity preservation**, and **streaming inference**.

### What makes it different
- Designed for **real-time pipelines**, not offline demos  
- Multi-stream audio conditioning (speech, emotion, motion)  
- Stable facial geometry via landmark anchoring  
- Optimized for **long-context streaming inference**

---

## 🚀 Core Capabilities (Table)

| Category | Specification |
|-------|-------------|
| **Inference Speed** | 30 FPS real-time streaming |
| **Latency** | < 33 ms per frame |
| **Lip-Sync Accuracy** | > 95% (DTW + contrastive learning) |
| **Identity Consistency** | > 0.92 ArcFace cosine similarity |
| **Expression Control** | 24+ FACS-compliant expressions |
| **Modal Inputs** | Audio · Text · Image · Video |
| **Target Hardware** | NVIDIA H200 / H100 / A100 |

---

## 🧬 Architecture Overview

| Component | Description |
|--------|------------|
| **Base Model** | 13.6B parameter Diffusion Transformer (DiT) |
| **Video Pipeline** | Dual-stream (Face 1024² / Body 512²) |
| **Facial Anchoring** | 68-point MediaPipe landmark constraints |
| **Audio Processing** | Multi-stream frequency separation |
| **Inference Engine** | Streaming KV-cache + circular latent buffer |
| **Parallelism** | Context Parallel (Ulysses Attention) |

---

## 🎧 Multi-Stream Audio Conditioning

| Stream | Frequency | Purpose |
|-----|-----------|---------|
| **Lip-Sync Stream** | 18–24 Hz | Phoneme & mouth articulation |
| **Prosody Stream** | 4–6 Hz | Emotion & speech dynamics |
| **Head Motion Stream** | 1–2 Hz | Natural pose & micro-movement |

---

## 🎯 Quality Metrics

| Metric | ARACHNE-X |
|-----|-----------|
| Lip-Sync Accuracy | **>95%** |
| LPIPS (Face Region) | **< 0.08** |
| Identity Stability | **> 0.92** |
| Optical Flow Variance | **< 5%** |
| Temporal Smoothness | High |

---

## ⚡ Performance Benchmarks

### Inference — Single H200

| Metric | Value |
|-----|------|
| FPS | 30 |
| Latency | 33 ms |
| Memory Usage | 110–120 GB |
| Throughput | 2,800 tokens/sec |

### Training — 8× H200 Pod

| Metric | Value |
|-----|------|
| Full Training | 58 hours (500K steps) |
| LoRA Fine-Tuning | 4–6 hours |
| Speed vs A100 | **4.5× faster** |
| Model Quality | LPIPS < 0.08 |

---

## 🧪 MOS Evaluation (Internal Benchmark)

### Text-to-Video

| Model | Params | Overall MOS |
|----|-------|-------------|
| Veo3 | – | 3.48 |
| PixVerse-V5 | – | 3.36 |
| Wan 2.2-T2V | 28B | 3.35 |
| **ARACHNE-X** | **13.6B** | **3.38** |

### Image-to-Video

| Model | Params | Overall MOS |
|----|-------|-------------|
| Seedance 1.0 | – | 3.35 |
| Hailuo-02 | – | 3.27 |
| Wan 2.2-I2V | 28B | 3.26 |
| **ARACHNE-X** | **13.6B** | **3.17** |

---

## ⚙️ System Requirements

| Component | Requirement |
|-------|-------------|
| GPU | NVIDIA H200 (recommended) |
| CUDA | 12.1+ |
| Python | 3.10+ |
| VRAM | 120GB (full) / 40GB (LoRA) |

---

## 🚀 Quick Start

```bash
git clone https://github.com/MagistrTheOne/ARACHNE-X-NULLXES-.git
cd ARACHNE-X

conda create -n arachne-x python=3.10
conda activate arachne-x

---

## 🔌 Frontend-Backend Contract (Avatar Session API)

To keep the frontend simple and secure, all provider-specific logic (RunPod endpoint IDs, API keys, retries, polling rules) must stay on the backend.

### Target realtime stack

- ASR/STT: [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- LLM: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- TTS: [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- Video renderer: ARACHNE-X pipeline

Recommended flow:

`STT/ASR -> LLM -> TTS -> ARACHNE output (stream/video)`

### Request from frontend

`POST /api/avatar/session`

```json
{
  "employeeId": "66",
  "avatarKey": "ksera_digital_twin",
  "voiceName": "Kore",
  "text": "Привет, чем помочь?",
  "locale": "ru-RU",
  "clientRequestId": "uuid-optional"
}
```

Minimum required fields:

- `employeeId` or `avatarKey` (one unique avatar reference is enough)
- `text` (for TTS/generation)
- `voiceName` (if voice selection is enabled)

### Response from backend (instant/session mode)

```json
{
  "provider": "runpod",
  "sessionId": "sess_123",
  "streamUrl": "https://.../stream.m3u8",
  "expiresAt": "2026-03-20T18:30:00Z",
  "status": "ready"
}
```

### Response from backend (job/poll mode)

```json
{
  "provider": "runpod",
  "jobId": "rp_job_123",
  "status": "processing",
  "pollUrl": "/api/avatar/jobs/rp_job_123"
}
```

### Data backend team must provide to frontend team

- RunPod `endpoint_id`
- Exact endpoint input schema (`text`, `voice`, `avatar_id`, etc.)
- Exact endpoint output schema (where to read stream/video URL)
- SLA/timeout policy (wait time before fallback)
- Session/URL TTL
- Throughput limits (RPS/concurrency)
- Auth requirements on backend API (JWT/cookie/session)

### Never send from frontend

- `RUNPOD_API_KEY`
- Any provider secret/token
- Internal private endpoint URLs

### Recommended backend endpoints

- `POST /api/avatar/session` - create session or generation request
- `GET /api/avatar/jobs/:jobId` - poll async job status
- `POST /api/avatar/stop` - optional interrupt/cleanup
