# Digital employee: real-time voice + avatar — system architecture

**Audience:** backend/platform engineers wiring production full-duplex “digital employees” (not a demo chatbot).  
**Scope:** structure, service boundaries, GPU placement, data flows, and enterprise behaviors (tools, policy, audit).  
**Reference stack:** Qwen streaming ASR → Qwen Next (MoE) LLM → Qwen streaming TTS → ARACHNE-X `generate_streaming_ai2v`; WebRTC to client.

---

## 1. Product definition vs “HeyGen-style talking head”

| Dimension | Consumer avatar SaaS | Digital employee (this doc) |
|-----------|----------------------|-----------------------------|
| Core output | Lip-sync video + script | **Same**, plus **grounded actions** |
| LLM role | Deliver dialogue | Dialogue + **tools**, workflows, KB, escalation |
| Compliance | Light | **Audit trail**, PII boundaries, retention |
| Latency | Interactive | **\<500 ms** first audio where possible |
| Interrupt | Optional | **Full-duplex barge-in** required |

Consumer avatar SaaS optimizes **low-latency A/V**; a digital employee optimizes **LLM that executes** (tickets, CRM, calendars, internal APIs) under guardrails.

---

## 2. Hardware tiers (your infra)

Assume two physical **roles** (exact SKU mapping is flexible):

| Tier | Role | Typical placement | Responsibility |
|------|------|-------------------|----------------|
| **Pool A** | Large VRAM compute (e.g. aggregate **~3 TB** RTX PRO class) | Many heavy models, parallel tenants | **LLM serving**, optional **secondary** inference, batch/RAG embedder, replicas |
| **Pool B** | **ARACHNE-X-ULTRA-X-2** node: **1× H200 SXM**, ~24 vCPU, ~251 GB RAM | Latency-critical path | **Avatar diffusion + VAE stream**, **ASR**, **TTS**, optional **ForcedAligner**, **WebRTC media** (or edge encoder) |

**Principle:** keep **time-critical multimodal path** on the H200 node (or tightly coupled rack) so **TTS → Avatar → encode** does not cross a slow network hop. Pool A holds **what does not need to be colocated** with every frame: primary LLM weights and scale-out.

---

## 3. Logical architecture (services)

```
                    ┌─────────────────────────────────────────────┐
                    │              Gateway (FastAPI)               │
                    │  sessions · auth · rate limits · signaling   │
                    └───────┬──────────────────────▲──────────────┘
                            │ WebRTC / WSS            │
                     ┌──────▼──────┐           ┌─────┴─────┐
                     │  Media IO   │           │  Client   │
                     │ (SDP, SRTP) │           │  (browser) │
                     └──────┬──────┘           └───────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   ┌────▼────┐        ┌─────▼─────┐      ┌──────▼──────┐
   │ Session │        │Orchestrator│     │ Policy /    │
   │ State   │◀──────▶│ (turns,  │◀─▶  │ RAG /       │
   │ Store   │        │ canceltok)│      │ Tools RPC   │
   └────┬────┘        └─────┬─────┘      └──────┬──────┘
        │                   │                    │
        │     Pool B (H200) │                    │ Pool A (~3TB)
        │                   │                    │
   ┌────▼────────────────────▼────┐      ┌──────▼──────┐
   │ Realtime inference lane      │      │ LLM service │
   │ · ASR (stream)               │      │ (Qwen Next) │
   │ · TTS (stream)               │      │ stream tok  │
   │ · Aligner (optional, batch)  │      │ + tool loop  │
   │ · ARACHNE-X Avatar           │      └─────────────┘
   └──────────────────────────────┘
```

**Mandatory separation:**

- **Gateway** never runs CUDA; only auth, routing, signaling, **generation_id** propagation.
- **Orchestrator** owns: VAD commit, SpanRouter, micro-turn queue, **barge-in** invalidation, **one in-flight Avatar per session**.
- **LLM service** (Pool A): streaming tokens **+ tool calling**; returns **speakable spans** and **structured tool results** to orchestrator.
- **Realtime lane** (Pool B): ASR, TTS, Avatar, optional aligner; **no business DB** in the hot path.

---

## 4. GPU scheduling (“who runs where”)

### 4.1 Recommended default

| GPU | Node | Processes |
|-----|------|-----------|
| **PRO pool** | Pool A | **Qwen Next 80B MoE** (Tensor Parallel / EP as your stack requires), replicas, **embedding models** for RAG |
| **H200 SXM** | Pool B | **Avatar resident** + **TTS** + **ASR** + **ForcedAligner** (optional), serialized or MPS-managed |

**Serialize on H200:** one **owner thread** (or ordered CUDA events) for `Avatar ≫ TTS ≫ ASR bursts` so two heavy kernels do not implicitly sync-fight.

### 4.2 Why not put LLM on the same H200 as Avatar

Avatar + TTS + ASR already consume VRAM and **latency budget**. MoE LLM spikes **activation memory** and **contends for SMs**; TTFA and lip-sync jitter will rise under load.

### 4.3 When to use PRO for Avatar instead

If you **burst** many concurrent Avatar sessions beyond one H200’s throughput, **shard Avatar replicas** across PRO GPUs **after** you have a **horizontal** pattern (multiple `Avatar workers`, sticky session or stateless checkpoint mount). First milestone: **single H200 realtime lane** + scale **LLM** horizontally on PRO pool.

---

## 5. Digital employee layer (LLM ≠ “болталка”)

### 5.1 Capabilities to implement explicitly

| Capability | Purpose | Typical integration |
|------------|---------|---------------------|
| **Tool / function calling** | Actions: create ticket, check order, book slot | Internal REST/gRPC; idempotent APIs |
| **RAG** | Grounding in employer KB | Embedders on Pool A; vector store; **inject citations** to logs, not always to TTS |
| **Policies** | What the employee may say/do | Rules engine **before** TTS (PII, refunds caps) |
| **Human handoff** | Legal/sensitive | Event to contact center + session freeze |
| **Audit** | Compliance | Immutable log: user audio hash, ASR final, LLM trace ID, tool calls, TTS text, **no raw secrets** |
| **Persona + task state** | “Employee”, not generic chat | System prompt + **FSM** (greeting → diagnose → act → close) |

### 5.2 Two-channel LLM output

1. **User channel (speakable):** only text that passes policy and is flushed by SpanRouter to TTS.  
2. **Ops channel (silent):** tool calls, state updates, CRM writes — **never** spoken verbatim unless approved templates.

Orchestrator **merges** streams: LLM may **stream tokens** for UX preview on UI while **holding** dangerous content from TTS until policy allows.

---

## 6. End-to-end data flow (full duplex)

### 6.1 User → system

1. **WebRTC** receives Opus → decode **48 kHz** → resample branch to **16 kHz** for ASR if required.  
2. **VAD + endpointing** (client AEC strongly recommended).  
3. **Streaming ASR** → partial text to UI + Orchestrator buffer.  
4. On commit → LLM **+** RAG context **+** tool results from previous turns.

### 6.2 System → user

1. LLM streams → **SpanRouter** emits micro-turns (200–800 ms audio target).  
2. Each sealed span → **TTS stream** → **PCM segment complete** → **Avatar** `generate_streaming_ai2v` → **frame stream** → encoder → **WebRTC video track**; **audio track** from same segment timeline (**audio master clock**).

### 6.3 Micro-turn invariant

**One Avatar call = one full PCM segment.** Real-time feel = **many short segments**, not one long render.

---

## 7. Barge-in (production)

- **generation_id** per “utterance cycle”; every queued message carries it.  
- **Hard cancel:** kill TTS decode, cooperative-exit diffusion loop, drop pending frames; **control queue** priority over data FIFO (NATS/Redis Stream with high-priority consumer or side channel).  
- **Soft cancel:** client fades outgoing audio; server stops **enqueuing** new spans; hard cancel if user continues speaking.  
- **Load:** max concurrent Avatar jobs globally; **reject new sessions** before starving cancel handling.

---

## 8. WebRTC and sync

- **Single PeerConnection** per session.  
- **Audio:** Opus 48 kHz; **video:** H.264 low-latency or VP9 with short GOP.  
- **Timeline:** per micro-turn, anchor **PTS** to audio sample index; pace video sends (avoid burst-after-decode).  
- **Jitter:** client plays audio with small jitter buffer; video **follows** audio clock (drop late video frames rather than desync).

---

## 9. Observability (SLO-driven)

| Metric | Use |
|--------|-----|
| TTFA (server) | Mic commit → first TTS PCM |
| TTFF (video) | First frame RTP sent |
| Tool latency | p95 per integration |
| Cancel latency | VAD start → GPU work stopped |
| Queue depth | Per-session pending segments (should stay ≤1) |

Alert on **Pool B SM occupancy** and **LLM span flood** (Orchestrator over-producing chunks).

---

## 10. Deployment checklist (minimal “magic path”)

1. Gateway + WebRTC **without** model load (health only).  
2. Pool A: LLM + RAG + tools **proven** under streaming + tool loop.  
3. Pool B: ASR + TTS + Avatar **warm path** measured (TTFA, first frame).  
4. Orchestrator: SpanRouter + **single-flight Avatar** + cancel path **load-tested**.  
5. Enterprise: audit log, PII redaction, handoff, retention.  
6. Scale: add **Avatar replicas** only after step 5 is green.

---

## 11. Summary

- **PRO ~3 TB pool** = scale **brain** (LLM, RAG, embeddings, replicas).  
- **H200 SXM node** = **face and voice path** (ASR, TTS, Avatar, optional aligner) with **strict GPU discipline**.  
- **Digital employee** = same realtime stack **plus** **tooling, policy, audit, and silent ops channel** — that is what differentiates product from generic avatar SaaS.

*This document describes integration architecture; ARACHNE-X repo paths for inference remain `arachne_x.loader.load_avatar_pipeline` and `generate_streaming_ai2v` as in project docs.*

---

## 12. Spec gaps: what a *real* digital employee still needs (after LLM ∥ Avatar split)

Splitting **LLM pool** and **H200 realtime lane** solves **throughput and contention**. It does **not** complete the product spec. Below is what enterprise buyers and production SREs will still ask for — capture these as **explicit requirements**, not implied.

**See also:** [ENTERPRISE_RAG_AND_DIGITAL_EMPLOYEE_2026.md](ENTERPRISE_RAG_AND_DIGITAL_EMPLOYEE_2026.md) — enterprise RAG stack choice, **Retrieval Policy Layer**, **Tool–RAG coupling**, PII pipeline by stage, and decision matrix (Atlas vs pgvector vs specialized vector DB vs managed RAG).

### 12.1 Tenant & identity (multi-employee, not one demo)

| Missing in many specs | Why it matters |
|----------------------|----------------|
| **Tenant model** (org → workspace → “digital employee” profiles) | Isolation of prompts, voices, avatars, quotas, API keys |
| **RBAC** who can edit behavior vs. only view analytics | Prevents prompt injection via admin UI |
| **Per-employee asset versioning** (face ref, voice id, style pack) | Reproducible sessions; rollback when a bad deploy breaks lip-sync |
| **Secret / connector vault** per tenant | Tools use OAuth — no secrets in LLM context |

### 12.2 “Brain” contract: tools, not vibes

| Missing | Specify in project |
|---------|-------------------|
| **Tool catalog** (ID, schema, side-effects class: read / write / money / PII) | Same as internal API catalog; versioned |
| **Idempotency & retries** | Tool calls MUST be safe under LLM + network retries |
| **Timeouts & circuit breakers** per integration | Or LLM will hallucinate “done” while CRM is down |
| **Human approval gates** for irreversible actions | Refund, delete, legal commit |
| **Dry-run / confirm-to-speak** pattern | Silent tool → policy → then templated spoken confirmation |

### 12.3 Knowledge (RAG) as operations, not a checkbox

| Missing | Specify |
|---------|---------|
| **Ingestion pipeline** (formats, OCR, access control inherited from source) | Who may see which doc slice |
| **Refresh / TTL / “I don’t know”** policy | Stale KB is a liability; define escalation |
| **Citation & attribution in audit log** | Not always read aloud — but must exist for disputes |
| **Evaluation set** per tenant (golden questions + expected tool + expected doc IDs) | Not MOS-only |

### 12.4 Safety, compliance, consent

| Missing | Specify |
|---------|---------|
| **Jurisdiction**: recording consent, retention, **biometric** (voice/face) where applicable | Product + legal sign-off |
| **PII redaction** on ingest, in logs, and **before** TTS where needed | Classifiers + blocklists |
| **Abuse & jailbreak** handling for **voice** (not just text) | Different attack surface |
| **Disclosure** (“you are speaking with an AI employee”) | Region-dependent |
| **Data residency** path (EU RuStore, etc.) | If claimed in sales |

### 12.5 Session semantics (what “one conversation” means)

| Missing | Specify |
|---------|---------|
| **Long tasks** spanning VAD gaps (“call me back in this flow”) | **Task state machine** persisted |
| **Handoff** to human (warm transfer, context package: summary + tool log + no raw PAN) | SLA |
| **Resume** across devices / next day (contact id, not only `session_id`) | CRM link |
| **Degraded modes**: LLM slow, Avatar saturated, tools down — **graceful voice copy** | Static image + TTS-only v.s. queue estimate |

### 12.6 Client & UX (not in backend doc alone)

| Missing | Specify |
|---------|---------|
| **AEC + noise** requirements; minimum browser / native shell | Full-duplex fails without this |
| **Accessibility** (captions, keyboard, screen reader for UI chrome) | Enterprise procurement |
| **Interrupt UX** (visual “I’m listening”, partial ASR latency) | Reduces user rage & repeat barges |

### 12.7 SRE / scale (beyond “we have GPUs”)

| Missing | Specify |
|---------|---------|
| **Global limits**: concurrent Avatar jobs, LLM tokens/s, cost caps per tenant | Fair scheduling |
| **Chaos / failover**: single AZ loss, model weight mount failure | Read-only fallback |
| **Runbooks**: cancel storm, WebRTC glare, GPU OOM on Pool B | On-call reality |
| **Synthetic probes**: TTFA, TTFF, tool happy-path every N minutes | Not manual demo |

### 12.8 What to do next after the split (ordering)

1. **Freeze Orchestrator contract** (messages, `generation_id`, span envelope, cancel priority channel).  
2. **Implement tool plane** with one **read-only** + one **write** integration and full audit.  
3. **Ship RAG** with tenant ACL + eval set before marketing “grounded”.  
4. **Load-test** barge-in + tool latency + Avatar queue (not average FPS).  
5. **Legal / security review** on recording, retention, handoff payloads.

Until **12.1–12.4** exist as written requirements, the system remains a **real-time avatar with an LLM attached**, not a **digital employee** product.
