# Enterprise RAG and Digital Employee Platform (2026)

**Purpose:** Specify how retrieval-augmented generation fits a **multi-tenant, enterprise-grade digital employee** — not as a standalone vector demo, but as **control plane + data plane + evidence**.  
**Companion:** [DIGITAL_EMPLOYEE_SYSTEM_ARCHITECTURE.md](DIGITAL_EMPLOYEE_SYSTEM_ARCHITECTURE.md) (runtime, Pool A/B, Orchestrator, WebRTC).

Marketing landing pages (e.g. [MongoDB Atlas — AI / vector](https://www.mongodb.com/lp/products/platform/atlas-vector-search-artificial-intelligence)) are **not** the technical source of truth; this document references **product documentation** where applicable.

---

## 1. Thesis for 2026

Enterprise RAG is **not** “a vector index next to an LLM”. It is an end-to-end **system**:

1. **Ingestion** with ACL inheritance from sources  
2. **Normalization** (chunking, enrichment, sensitivity tags)  
3. **Embed** → **index** (possibly multiple: public vs sensitive)  
4. **Hybrid retrieval** (keyword + vector; rank fusion / re-rank)  
5. **Context pack** assembly with **citations**  
6. **LLM** generation under **policy**  
7. **Evidence pack** for audit (chunk IDs, hashes, scores — not secrets)

For a **digital employee**, two properties are non-negotiable:

- **Provability:** for disputes and compliance, answers must tie to **retrieved evidence** (expected chunk IDs in eval; content hash in audit).  
- **No cross-tenant / cross-role leakage:** every query must enforce **subject** (user / role / group) against chunk metadata.

---

## 2. Storage and retrieval: decision criteria

Use this table when choosing **where** vectors live and **how** retrieval is run — not as a one-size-fits-all.

| Criterion | What to decide |
|-----------|----------------|
| **Multi-tenant model** | **Silo:** index (or cluster) per tenant — strongest isolation, higher cost. **Pool:** shared index with strict `tenant_id` + ACL filters — cheaper, needs discipline. **Bridge:** silo for enterprise customers, pool for SMB. |
| **ACL** | On **every** retrieval request: `subject` must match `chunk.metadata.acl` (and tenant / employee scope). |
| **Hybrid** | Combine BM25 / full-text with vector; use **RRF** or vendor **rank fusion** (e.g. MongoDB Atlas hybrid patterns). Pure vector often underperforms on SKU/IDs and exact policy text. |
| **Operational data** | If KB must stay **consistent** with live objects (tickets, orders), prefer **one logical store** for docs + metadata + vectors or a **strict sync contract** between OLTP and vector store. |
| **Latency / SLO** | p99 retrieval budget; dedicated search capacity where the vendor allows (e.g. Atlas Search nodes). |
| **Compliance** | Residency, encryption at rest, audit logs from the provider, key management (CMK) for enterprise deals. |

---

## 3. MongoDB Atlas Vector Search — when “yes”, when “no”

**Fit (“yes”):**

- You want **operational documents** (orgs, employees, tool runs, session summaries) **next to** embeddings in **one** data platform — fewer sync bugs than “Postgres for app + separate vector DB” without strong ETL.  
- **Hybrid search** and framework integrations (e.g. [LangChain hybrid search with Atlas](https://www.mongodb.com/docs/atlas/atlas-vector-search/ai-integrations/langchain/hybrid-search/)) reduce time-to-production.  
- Enterprise features (HA, security, ops) are part of the **Atlas** story; see [Vector Search overview](https://www.mongodb.org/docs/atlas/atlas-vector-search/vector-search-overview/).

**Caution (“no” or “not alone”):**

- **Vendor / region lock-in** and cost at scale — must be accepted explicitly.  
- **ACL is still your schema** — Atlas does not replace modeling `principal`, `visibility`, or **per-tenant indexes** for regulated tenants.  
- **Extreme vector QPS** or **very large** dedicated vector fleets may favor a **specialized** engine (Qdrant, Milvus, etc.) with a clear sync path from OLTP.

---

## 4. Alternatives (short)

| Option | When |
|--------|------|
| **PostgreSQL + pgvector** | Product already on Postgres; use RLS / schemas for tenants; combine full-text + `pgvector` for hybrid. |
| **Qdrant / Milvus / Weaviate** | Vector-first SLA, heavy filtered ANN, large-scale shards; pair with a separate OLTP system of record. |
| **Managed cloud RAG** (e.g. Bedrock Knowledge Bases, Azure AI Search patterns) | Fast MVP; less control over ACL nuance and custom eval harness — acceptable for early stage if limitations are documented. |

The product should support **branching** (MVP vs enterprise silo), not a single global choice forever.

---

## 5. Mapping to the Digital Employee stack

- **Ingestion:** ACL from source systems (SharePoint, S3, Confluence, etc.) → `chunk.metadata.acl` (+ `tenant_id`, `employee_id` where scoped).  
- **Retrieval query:** always filter `tenant_id` + `employee_id` (if applicable) + `subject ∈ allowed_principals`.  
- **Evaluation (per tenant or golden set):**  
  `question → optional expected_tool → expected_chunk_ids`  
  Run in CI or nightly; fail builds or alert on regressions — **not** MOS-only.  
- **Audit evidence pack** (for disputes): `trace_id`, retrieved chunk IDs, content hash at retrieval time, scores; **never** log raw secrets, PAN, health data.

---

## 6. Retrieval Policy Layer (explicit rules)

Retrieval must not be “whatever `top_k` returns”. Add a **policy layer** that **gates** the next step (LLM, tool call, or spoken reply).

| Situation | Policy (examples — tune per tenant) |
|-----------|--------------------------------------|
| **High-risk action** (money, destructive write, PII-class tool) | Require **top_k ≥ N**, **minimum similarity score** (or max distance), optional **second-stage re-rank** or second retrieval pass. If thresholds fail → **do not call tool**; use escalation template or “cannot confirm”. |
| **Low confidence** | Empty hit set, score below floor, or **conflicting** chunks → fallback: “I don’t know”, clarifying question, or **human escalation** per tenant policy. |
| **Tool requires entity grounding** | Before a write/money tool, retrieval (or a **lookup** read tool) must **confirm** entity existence and visibility (ID, status). If not confirmed → **lookup tool**, user clarification, or refuse. |

**Intent:** RAG **decides** whether it is safe to proceed — not only to “stuff context” into the LLM.

---

## 7. Tool–RAG coupling (“brain” and “hands”)

RAG and tools must not be two disconnected boxes. Specify **coupling**:

1. **Retrieval → tool selection**  
   Signals from chunks (entity types, intents, policy excerpts) **narrow** the allowed tool set or select a **schema** — Orchestrator + PolicyEngine enforce this before any call.

2. **Tool results → short-term memory**  
   Successful **read** results (sanitized, no secrets) may be written to **session-scoped memory** with TTL and optionally to an **ephemeral retrieval index** for the rest of the conversation.  
   **Writes** to long-term KB from tool output only with **explicit rules** (never blindly index raw PII).

3. **RAG → validation of tool inputs**  
   For **write / money** tools: validate IDs, emails, contract numbers against **retrieved** evidence and **last consistent tool read**. Mismatch → **reject** the call, ask for clarification, or **re-retrieve**.

4. **Tracing**  
   Same `trace_id` ties **retrieval decisions**, **tool calls**, and **spoken** output in audit logs.

---

## 8. PII and sensitive data flow (by pipeline stage)

| Stage | Action |
|-------|--------|
| **Ingestion** | Classify sensitivity: `public` / `internal` / `pii`. Tag chunks; optionally route **sensitive** chunks to a **separate collection or index**. |
| **Embedding** | Per tenant policy: embed **after** redaction, or **do not embed** body text for certain fields (metadata + pointer only). |
| **At rest** | Encryption (provider default + **CMK** for enterprise). Vectors subject to **same** policy class as source documents. |
| **Logging / audit** | **Redact before write**; store chunk id, hash, scores — not raw PAN, health identifiers, full account numbers. |
| **LLM context** | Optional strip or mask by tag before model call. |
| **TTS** | Optional **redaction or substitution** (e.g. never speak full card numbers) per policy. |

---

## 9. Decision matrix: when to choose what

| If you need… | Lean toward… |
|--------------|----------------|
| **Single source of truth** and **operational docs** co-located with search and hybrid retrieval | **MongoDB Atlas** (Vector Search + operational model in one platform; see [Vector Search overview](https://www.mongodb.org/docs/atlas/atlas-vector-search/vector-search-overview/)). |
| **Heavy vector-only QPS**, huge indices, vector-first SRE | **Qdrant / Milvus** (+ separate OLTP for operational data and strict sync). |
| **Existing Postgres**-centric product and team | **pgvector** + RLS / schemas for multi-tenant isolation + hybrid SQL. |
| **Fastest MVP** with minimal platform headcount | **Managed cloud RAG** (trade control of ACL and eval for speed; document lock-in). |

---

## 10. References

- MongoDB Atlas Vector Search — [Vector Search overview](https://www.mongodb.org/docs/atlas/atlas-vector-search/vector-search-overview/)  
- Hybrid / LangChain — [Hybrid search with MongoDB and LangChain](https://www.mongodb.com/docs/atlas/atlas-vector-search/ai-integrations/langchain/hybrid-search/)  
- MongoDB Atlas marketing context — [Atlas AI / vector](https://www.mongodb.com/lp/products/platform/atlas-vector-search-artificial-intelligence)  
- Multi-tenant secure RAG checklist (pattern reference) — [Azure Architecture — secure multitenant RAG](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/secure-multitenant-rag)

---

## See also

- [DIGITAL_EMPLOYEE_SYSTEM_ARCHITECTURE.md](DIGITAL_EMPLOYEE_SYSTEM_ARCHITECTURE.md) — gateway, Orchestrator, Pool A/B, WebRTC, barge-in, enterprise gaps (§12).
