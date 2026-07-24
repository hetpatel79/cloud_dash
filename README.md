# CloudDash — Autonomous Multi-Agent AI Customer Support System

CloudDash is a multi-agent customer support pipeline that routes every incoming message through a fixed, auditable sequence of specialized agents — input guardrail, triage, hybrid retrieval, a specialist responder, and output guardrail — instead of relying on a single general-purpose language model to handle every kind of query.

Built with **LangGraph**, **FastAPI**, **Groq**, **NVIDIA NIM**, and **Qdrant**.

📄 **Read Medium Blog :** [Beyond the Single Chatbot: Building an Autonomous Multi-Agent AI System for Customer Support](https://medium.com/@ihetpatel79/beyond-the-single-chatbot-building-an-autonomous-multi-agent-ai-system-for-customer-support-8b8317730561)

---


---

## Why CloudDash

A support chatbot built around a single general-purpose model tends to run into the same problems in practice: it has no natural way to specialize between a billing dispute and a server outage, no built-in defense against prompt injection or off-topic misuse, and no mechanism to verify its own answer is actually grounded in real documentation before it reaches the customer.

CloudDash splits these responsibilities across a small team of purpose-built agents, coordinated through a deterministic graph rather than a single long prompt, to produce a more reliable and auditable support pipeline.

## Architecture

Every customer message travels through the same fixed sequence of nodes:

```
Customer Message
      │
      ▼
Input Guardrail  (4-layer check)
      │
      ▼
Triage Agent  (intent + language detection)
      │
      ▼
Hybrid Retriever  (BM25 + NIM embeddings, RRF fusion, rerank)
      │
      ▼
Specialist Agent  (Technical / Billing / Escalation)
      │
      ▼
Output Guardrail  (PII redaction + grounding check)
      │
      ▼
Reply → FastAPI /chat endpoint
```

See **ARCHITECTURE.md** for a deeper component walk-through.

| Stage | Responsibility |
|---|---|
| **Input Guardrail** | Rejects unsafe, off-topic, or prompt-injection messages before any language model is invoked. |
| **Triage Agent** | Classifies intent (technical, billing, escalation) and detects language. |
| **Hybrid Retriever** | Rewrites the query, then combines dense vector search (Qdrant + NVIDIA NIM embeddings) with BM25 keyword search via Reciprocal Rank Fusion and a NIM reranker. |
| **Technical / Billing Agent** | Drafts a grounded reply using only the passages retrieval supplied. |
| **Escalation Agent** | Packages the conversation for human hand-off when a specialist agent cannot resolve it, or when the customer signals they need a human. |
| **Handover Protocol** | Governs and logs every agent-to-agent transfer against an allow-listed set of transitions. |
| **Output Guardrail** | Redacts sensitive data (PII) and verifies the reply is grounded in retrieved documentation before release. |

## Features

- Deterministic, sequential multi-agent pipeline orchestrated with LangGraph
- Four-layer input guardrail (regex jailbreak detection, fast on-topic check, hard-block rules, LLM classifier fallback)
- Hybrid retrieval: dense (NVIDIA NIM embeddings via Qdrant) + sparse (BM25) search merged with Reciprocal Rank Fusion and reranked
- Governed, audited handover protocol between agents with a JSONL audit trail
- Output guardrail combining PII redaction with an LLM-based grounding/hallucination check
- REST API (FastAPI) with a lightweight browser-based chat interface for manual testing
- Structured, trace-correlated logging across the entire pipeline

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Orchestration | LangGraph |
| Backend / API | FastAPI, uvicorn, Pydantic v2 |
| LLM Inference | Groq (LLaMA 3.3 70B, LLaMA 3.1 8B) |
| Embeddings / Reranking | NVIDIA NIM (via OpenAI-compatible client) |
| Vector Store | Qdrant Cloud |
| Keyword Search | rank-bm25 |
| Logging | structlog |
| Tooling | Git/GitHub, Postman, VS Code |

## Project Structure

```
clouddash-support/
├── agents/           Triage, Technical, Billing, Escalation agents + orchestrator
├── api/              FastAPI entry point, routes, response models
├── config/           YAML configuration (agents.yaml, settings.yaml)
├── guardrails/        Input guard (4-layer) and output guard (PII + hallucination)
├── handover/          Handover protocol, packaging, and JSONL audit log
├── knowledge_base/    20 curated KB articles (JSON) + ingestion pipeline
├── models/            Pydantic schemas shared across all modules
├── retrieval/         Embedder, Qdrant store, hybrid retriever, reranker
├── tests/             Scenario tests + full QA suite (18 tests)
├── ui/                Browser-based chat interface
├── utils/             Structured logging, trace context, OpenAI compatibility layer
└── logs/              Runtime handover audit logs
```

## Getting Started

### Prerequisites

- Python 3.11+
- API keys for [Groq](https://console.groq.com/), [NVIDIA NIM](https://build.nvidia.com/), and a [Qdrant Cloud](https://qdrant.tech/) instance

### Installation

```bash
git clone https://github.com/hetpatel79/cloud_dash.git
cd cloud_dash
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and add your credentials:

```bash
cp .env.example .env
```

Populate `.env` with your `GROQ_API_KEY`, NVIDIA NIM credentials, and Qdrant connection details, then adjust `config/settings.yaml` and `config/agents.yaml` as needed.

### Ingest the knowledge base

```bash
python -m knowledge_base.ingest
```

### Run the server

```bash
uvicorn api.main:app --reload --port 8001
```

The chat interface is available at `http://127.0.0.1:8001/ui/`.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/chat` | POST | Accepts a customer message and returns the agent-generated reply along with routing metadata. |
| `/health` | GET | Reports system status, knowledge base article count, and Qdrant connection state. |
| `/handovers` | GET | Returns the handover audit log for inspection. |

## Tests

```bash
pytest                              # all scenario tests
pytest -v tests/test_scenarios.py
```

| # | Scenario | Covers |
|---|---|---|
| 1 | Single technical agent | Alerts, AWS credentials, KB citations |
| 2 | Cross-agent handover | Multi-intent SSO → Enterprise upgrade |
| 3 | Escalation trigger | Double-charge + frustrated sentiment |
| 4 | KB miss graceful | Unknown integration, no hallucination |

The full QA suite (`tests/qa_full.py`) covers 18 tests spanning routing, guardrails, retrieval, handover, observability, and edge cases.

## Observability

- Every request is assigned a `trace_id` (UUID), propagated via the `X-Trace-ID` response header and the JSON body.
- All agent invocations, LLM calls, guardrail triggers, and handover events are logged as structured JSON via `structlog`, correlated by `trace_id` and `conversation_id`.
- Handover events are persisted to `logs/handovers.jsonl` for audit and compliance review.

## Known Limitations

- The knowledge base currently holds 20 articles, written in English only.
- The system depends on three external services (Groq, NVIDIA NIM, Qdrant Cloud) and has no offline fallback.
- No persistent memory across sessions.
- Runs without authentication, rate limiting, or load balancing — not yet production-ready as-is.
- A mixed-intent conversation is handed to a single agent (typically Escalation) rather than split across specialists.


## Acknowledgements

Built during a Summer Internship II at **Techmicra IT Solutions**, Ahmedabad, under the guidance of Mr. Pallav Mamtora (CEO, Techmicra IT Solutions) and Dr. Hardik Jayswal (CSPIT, CHARUSAT).

---

**Author:** [Het Patel](https://github.com/hetpatel79) 
