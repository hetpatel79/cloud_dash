# CloudDash Support -- Architecture

## High-Level Modules

- **`models/`** -- Canonical Pydantic models shared by API and agents (`ConversationState`, `KBChunk`, `ExtractedEntities` with `detected_language`, etc.).
- **`config/`** -- YAML for retrieval/Qdrant and per-agent prompts/models (`agents.yaml`, `settings.yaml`).
- **`knowledge_base/`** -- Twenty curated JSON articles plus `ingest.py` (chunk, embed, Qdrant upsert, BM25 pickle).
- **`retrieval/`** -- `Embedder`, `QdrantStore` (singleton, lazy client), `HybridRetriever` (query rewrite with few-shot prompt, RRF fusion, three-stage fallback, optional category filter), `Reranker` (LLM scoring with safe fallback).
- **`guardrails/`** -- Four-layer input guardrail (injection regex, allow-keywords, off-topic regex, LLM few-shot classification). Output guardrail with PII redaction and hallucination heuristics.
- **`handover/`** -- `package_handover`, `HandoverProtocol` (allowed transitions), JSONL audit log under `logs/handovers.jsonl`, queryable via both per-conversation and global `/handovers` endpoints.
- **`agents/`** -- `BaseAgent` utilities (language-aware `build_messages`), specialist agents (`TriageAgent` with language detection, `TechnicalSupportAgent`, `BillingAgent`, `EscalationAgent`), and `SupportOrchestrator` LangGraph.
- **`api/`** -- FastAPI entry (`main.py`) and routes; maps HTTP to graph runs. Endpoints for conversations, handovers, health, and KB reload.
- **`utils/`** -- `structlog` setup, trace helpers, and OpenAI-compatible LLM client abstraction.

## LangGraph Topology

1. `START -> input_guard -> triage`
2. Triage routes to `technical`, `billing`, `escalation`, or `output_guard` (clarifications and short paths).
3. Specialists may route to each other (`technical <-> billing`, validated by `HandoverProtocol`) or to `escalation` when `requires_human` is set.
4. `escalation -> output_guard -> END`
5. Other branches end via `output_guard`.

`GraphState` (`TypedDict`) mirrors `ConversationState` JSON while remaining LangGraph-friendly. `run_conversation` binds `trace_id` and `conversation_id` into `structlog` context vars.

## Input Guardrail Pipeline

The `InputGuardrail` runs four layers in sequence on every message. There are no bypass paths.

1. **Regex injection check** -- 19 compiled patterns catch instruction override, jailbreak, system prompt extraction, and DAN attacks.
2. **Allow-keyword fast pass** -- If the message contains known CloudDash terms (e.g. "alert", "billing", "SSO", "dashboard"), the message is allowed through without further off-topic checks.
3. **Regex off-topic check** -- Hard blocks for clearly non-support content (recipes, sports scores, homework, creative writing).
4. **LLM off-topic classification** -- Groq-hosted Llama 3.1 8B with few-shot examples classifies ambiguous messages. On LLM failure, the message is allowed through (fail-open for borderline cases only).

The orchestrator passes conversation history to the guardrail so context-aware decisions can be made.

## Retrieval Pipeline

1. **Query rewriting** -- Groq-hosted Llama 3.1 8B converts user messages into keyword search queries using few-shot examples that stay close to the original intent.
2. **Vector + BM25** -- Fused via **RRF** (`k=60`).
3. **Three-stage fallback** -- If the rewritten query returns zero results, retry with the original query. If still empty and a category filter was applied, retry without the filter.
4. **Optional rerank** -- LLM scores chunks 0-10; failures keep the fused order.
5. **Thresholding** -- If the best dense score is below `0.5`, technical guidance forces an explicit "no documentation" admission.

## Multi-Language Support

- `TriageAgent` detects the language of incoming messages during intent classification.
- The detected language is stored in `entities.detected_language`.
- `BaseAgent.build_messages()` checks this field and appends a language instruction to the system prompt, directing downstream agents to respond in the customer's language.
- KB article content is translated inline by the responding agent's LLM.

## Security and Compliance Hooks

- **Input**: Four-layer guardrail (see above). Every message is checked; no debug bypasses exist.
- **Output**: Credit card, SSN, and email redaction; hallucination prompt compares the answer to retrieved passages.
- **Audit**: All handover events are logged to `logs/handovers.jsonl` with conversation correlation.

## Files Worth Reading First

1. `agents/orchestrator.py` -- Routing truth table and LangGraph node definitions.
2. `guardrails/input_guard.py` -- Four-layer guardrail implementation.
3. `retrieval/retriever.py` -- Hybrid search, query rewriting, and fallback logic.
4. `handover/protocol.py` -- Allowed agent transitions.
5. `api/routes.py` -- All HTTP endpoints.
