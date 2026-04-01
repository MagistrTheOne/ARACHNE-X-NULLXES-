# NULLXES x JobAI integration (H200 & RTX 6000+)

## 1) Goal

Build a production-ready "live AI interviewer/avatar" platform for JobAI based on ARACHNE-X:

- Real-time conversational interview flow
- Natural speech, lip-sync, and identity-consistent avatar output
- Secure backend orchestration (no provider secrets on frontend)
- Pilot first on RTX PRO 6000, production scale on H200

Target model stack:

- ASR/STT: `Qwen3-ASR-1.7B`
- LLM: `Qwen2.5-7B-Instruct`
- TTS: `Qwen3-TTS-12Hz-1.7B-CustomVoice`
- Video renderer: `ARACHNE-X` (LongCat-Video + Avatar pipeline)

---

## 2) Product vision for JobAI use case

### Core user journey

1. Candidate opens interview session.
2. Speaks to AI HR avatar in natural dialog.
3. System transcribes speech, generates HR response, synthesizes voice, renders talking avatar.
4. Candidate receives real-time response video/audio stream.
5. Backend stores interview transcript, events, metrics, and scoring signals.

### Business value

- Faster screening throughput
- Consistent interviewer behavior and standards
- Lower recruiter workload
- 24/7 interview availability

---

## 3) High-level architecture

Pipeline:

`Client WebRTC/WebSocket -> Orchestrator -> ASR -> LLM -> TTS -> ARACHNE Renderer -> Stream Out`

Service boundaries:

- `session-gateway` (auth, rate limit, session lifecycle)
- `asr-service` (streaming ASR + VAD)
- `llm-service` (dialog policy + response generation)
- `tts-service` (voice synthesis, chunk streaming)
- `avatar-service` (ARACHNE session render)
- `media-service` (WebRTC/HLS delivery)
- `analytics-service` (logs, KPIs, interview scoring)

State/data:

- Redis: ephemeral session state and pub/sub
- Postgres: interview metadata and transcripts
- Object storage (S3 compatible): video artifacts

---

## 4) Backend API contract (frontend-safe)

Frontend must never know RunPod/internal keys.

### Create session

`POST /api/avatar/session`

Request:

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

Response (ready):

```json
{
  "provider": "runpod",
  "sessionId": "sess_123",
  "streamUrl": "https://.../stream.m3u8",
  "expiresAt": "2026-03-20T18:30:00Z",
  "status": "ready"
}
```

Response (job/poll):

```json
{
  "provider": "runpod",
  "jobId": "rp_job_123",
  "status": "processing",
  "pollUrl": "/api/avatar/jobs/rp_job_123"
}
```

Recommended endpoints:

- `POST /api/avatar/session`
- `GET /api/avatar/jobs/:jobId`
- `POST /api/avatar/stop`

---

## 5) Deployment strategy: RTX pilot -> H200 production

## Phase A: Pilot on RTX PRO 6000 (1 GPU)

Goal: validate UX and business flow with 1 concurrent session.

Settings:

- Lower render settings (fewer frames/steps)
- `flash-attn` optional; fallback to SDPA/xformers if needed
- Conservative latency budget, optimize stability first

Pilot acceptance criteria:

- End-to-end dialog works reliably
- Interrupt/barging-in works
- Session creation/stop is stable
- Interview transcript saved

## Phase B: Pre-production hardening

- Containerize all services
- Add tracing/metrics dashboards
- Backpressure and retry policies
- Failover behavior and timeout rules

## Phase C: Production on H200

Goal: improve quality, lower latency, increase concurrency.

H200 profile:

- Higher resolution and quality presets
- More inference steps for realism
- Better multi-session throughput

---

## 6) Realtime performance targets (initial)

- ASR first partial: <= 500 ms
- LLM first token: <= 700 ms
- TTS first chunk: <= 800 ms
- Avatar first frame after TTS chunk: <= 1000 ms
- End-to-end perceived response start: <= 1.5-2.5 s

Track p50/p95 per stage:

- `asr_latency_ms`
- `llm_first_token_ms`
- `tts_first_chunk_ms`
- `avatar_render_window_latency_ms`
- `session_e2e_first_response_ms`

---

## 7) Interview-specific logic (JobAI)

- Role-based prompt templates (Sales, Support, Engineering, etc.)
- Structured interview script with adaptive follow-up
- Candidate scoring rubric (communication, relevance, confidence, consistency)
- Safety filters and policy guardrails
- Human override/escalation path

---

## 8) Security and compliance baseline

- No provider secrets in frontend
- Backend JWT auth and RBAC for recruiter/admin
- Session TTL and signed stream URLs
- PII handling policy and encryption at rest
- Full audit logs for interview actions

---

## 9) Pilot implementation plan (6 weeks)

Week 1:

- Infra bootstrap, repositories, CI, base services

Week 2:

- ASR + LLM integration, prompt templates, transcript storage

Week 3:

- TTS streaming + interrupt handling

Week 4:

- ARACHNE avatar rendering and media output

Week 5:

- Frontend integration, end-to-end QA, latency tuning

Week 6:

- UAT with real interview scenarios, KPI review, go/no-go for H200 scale

---

## 10) Risks and mitigation

- GPU compatibility drift (`flash-attn`, CUDA, driver mismatch)
  - Mitigation: lock images, keep SDPA fallback, smoke tests on startup

- Latency spikes under load
  - Mitigation: queueing, backpressure, per-stage timeouts, autoscaling rules

- Model quality variance in domain dialogs
  - Mitigation: structured prompts, evaluation set, periodic prompt tuning

- Operational complexity
  - Mitigation: clear service boundaries, centralized observability, runbooks

---

## 11) Recommended next steps

1. Freeze API contract with frontend team.
2. Build pilot MVP on RTX PRO 6000 with one-session SLA.
3. Run 20-50 internal interview sessions and collect metrics.
4. Approve production budget/profile for H200 rollout.
5. Migrate same containers/config to H200 and scale concurrency.

