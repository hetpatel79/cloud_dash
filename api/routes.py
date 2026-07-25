"""FastAPI routes for CloudDash support API."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Response

from agents.orchestrator import SupportOrchestrator, build_default_orchestrator
from api.models import ConversationDetailResponse, HandoverListResponse, HealthResponse, KBReloadResponse
from handover.audit_log import AUDIT
from models.schemas import (
    ConversationResponse,
    ConversationStartRequest,
    ConversationState,
    ExtractedEntities,
    Message,
    MessageRequest,
    MessageRole,
)
from utils.logger import get_logger
from utils.trace import TraceContext, generate_conversation_id, generate_trace_id

logger = get_logger("api.routes")

router = APIRouter()

_orchestrator: SupportOrchestrator | None = None
_conversations: dict[str, ConversationState] = {}


def get_orch() -> SupportOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = build_default_orchestrator()
    return _orchestrator


def _latest_assistant(state: ConversationState) -> str:
    for m in reversed(state.messages):
        if m.role == MessageRole.ASSISTANT:
            return m.content
    return ""


def _citations(state: ConversationState) -> list:
    return list(state.retrieved_chunks)


@router.post("/chat", response_model=ConversationResponse)
async def chat(body: MessageRequest, response: Response) -> ConversationResponse:
    """Single-endpoint convenience wrapper: accepts a customer message and returns
    the agent-generated reply along with routing metadata.

    Starts a new conversation if `conversation_id` is empty/unknown, otherwise
    appends to the existing one. Internally delegates to the same orchestrator
    used by /conversations and /conversations/{id}/messages.
    """
    if not body.conversation_id or body.conversation_id not in _conversations:
        return await start_conversation(
            ConversationStartRequest(initial_message=body.message), response
        )
    return await append_message(body.conversation_id, body, response)


@router.post("/conversations", response_model=ConversationResponse)
async def start_conversation(body: ConversationStartRequest, response: Response) -> ConversationResponse:
    trace_id = generate_trace_id()
    TraceContext.set(trace_id)
    conv_id = generate_conversation_id()
    now = datetime.now(timezone.utc)
    user_msg = Message(role=MessageRole.USER, content=body.initial_message, timestamp=now)
    state = ConversationState(
        conversation_id=conv_id,
        trace_id=trace_id,
        messages=[user_msg],
        entities=ExtractedEntities(customer_id=body.customer_id, plan_type=body.plan_type),
    )
    try:
        final = await asyncio.to_thread(get_orch().run_conversation, state)
    except Exception as exc:  # noqa: BLE001
        logger.error("conversation_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Conversation engine failed") from exc
    _conversations[conv_id] = final
    response.headers["X-Trace-ID"] = trace_id
    logger.info("conversation_started", conversation_id=conv_id, trace_id=trace_id)
    return ConversationResponse(
        conversation_id=conv_id,
        trace_id=trace_id,
        response=_latest_assistant(final),
        current_agent=final.current_agent,
        citations=_citations(final),
        is_resolved=final.is_resolved,
        requires_human=final.requires_human,
        escalation_package=final.escalation_package,
        input_guard_failed=final.input_guard_failed,
        detected_language=final.entities.detected_language,
    )


@router.post("/conversations/{conversation_id}/messages", response_model=ConversationResponse)
async def append_message(conversation_id: str, body: MessageRequest, response: Response) -> ConversationResponse:
    if conversation_id not in _conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    trace_id = _conversations[conversation_id].trace_id
    TraceContext.set(trace_id)
    now = datetime.now(timezone.utc)
    state = _conversations[conversation_id]
    if state.messages and state.messages[-1].role == MessageRole.USER and state.messages[-1].content.strip() == body.message.strip():
        raise HTTPException(status_code=409, detail="Duplicate user message already pending")
    state.messages.append(Message(role=MessageRole.USER, content=body.message, timestamp=now))
    try:
        final = await asyncio.to_thread(get_orch().run_conversation, state)
    except Exception as exc:  # noqa: BLE001
        logger.error("conversation_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Conversation engine failed") from exc
    _conversations[conversation_id] = final
    response.headers["X-Trace-ID"] = trace_id
    return ConversationResponse(
        conversation_id=conversation_id,
        trace_id=trace_id,
        response=_latest_assistant(final),
        current_agent=final.current_agent,
        citations=_citations(final),
        is_resolved=final.is_resolved,
        requires_human=final.requires_human,
        escalation_package=final.escalation_package,
        input_guard_failed=final.input_guard_failed,
        detected_language=final.entities.detected_language,
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(conversation_id: str) -> ConversationDetailResponse:
    if conversation_id not in _conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    st = _conversations[conversation_id]
    return ConversationDetailResponse(
        conversation_id=st.conversation_id, trace_id=st.trace_id, state=st, messages=st.messages
    )


@router.get("/conversations/{conversation_id}/handovers", response_model=HandoverListResponse)
async def get_handovers(conversation_id: str) -> HandoverListResponse:
    # Return empty array (not 404) if no handovers exist — some convos never trigger one
    events = AUDIT.get_handover_history(conversation_id)
    return HandoverListResponse(conversation_id=conversation_id, events=events)


@router.get("/handovers")
async def list_all_handovers():
    """Return all handover events from disk log."""
    import json as _json
    from pathlib import Path

    log_path = Path(__file__).resolve().parents[1] / "logs" / "handovers.jsonl"
    entries: list[dict] = []
    if log_path.exists():
        with log_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    entries.append(_json.loads(line.strip()))
                except _json.JSONDecodeError:
                    continue
    return entries


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    kb_loaded = False
    try:
        from retrieval.qdrant_store import QdrantStore

        info = QdrantStore.instance().get_collection_info()
        count = info.get("points_count")
        kb_loaded = bool(count)
    except Exception:
        kb_loaded = False

    kb_article_count = 0
    try:
        from pathlib import Path

        articles_dir = Path(__file__).resolve().parents[1] / "knowledge_base" / "articles"
        kb_article_count = len(list(articles_dir.glob("KB-*.json")))
    except Exception:
        kb_article_count = 0

    return HealthResponse(
        status="ok",
        kb_loaded=kb_loaded,
        kb_article_count=kb_article_count,
        agents_ready=True,
    )


@router.post("/kb/reload", response_model=KBReloadResponse)
async def reload_kb() -> KBReloadResponse:
    from knowledge_base import ingest as ingest_mod

    async def _run() -> None:
        await asyncio.to_thread(ingest_mod.main)

    asyncio.create_task(_run())
    return KBReloadResponse(status="accepted")
