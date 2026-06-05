"""Build structured handover payloads."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from utils.openai_compat import chat_openai

from models.schemas import AgentType, ExtractedEntities, HandoverPayload, Message
from utils.logger import get_logger
from utils.trace import TraceContext

logger = get_logger("handover.context")


def compress_history(messages: Sequence[Message]) -> str:
    trace_id = TraceContext.get() or "unknown"
    transcript = "\n".join(f"{m.role.value}: {m.content}" for m in messages[-30:])
    sys = SystemMessage(
        content="Summarize this support conversation in 200 words or fewer. Preserve IDs, plans, errors."
    )
    try:
        llm = chat_openai(model="meta/llama-3.3-70b-instruct", temperature=0.2)
        resp = llm.invoke([sys, HumanMessage(content=transcript)])
        return (getattr(resp, "content", "") or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("compress_history_failed", trace_id=trace_id, error=str(exc))
        return transcript[:1200]


def extract_handover_context(state: dict) -> dict:
    entities = state.get("entities") or {}
    if hasattr(entities, "model_dump"):
        entities = entities.model_dump()
    return {
        "conversation_id": state.get("conversation_id"),
        "customer_id": entities.get("customer_id"),
        "plan_type": entities.get("plan_type"),
        "issue_type": entities.get("issue_type"),
        "resolution_attempts": list(state.get("routing_history", [])),
        "urgency": entities.get("urgency", "medium"),
    }


def package_handover(state: dict, target_agent: AgentType, reason: str) -> HandoverPayload:
    entities_raw = state.get("entities") or {}
    if isinstance(entities_raw, dict):
        entities = ExtractedEntities.model_validate(entities_raw)
    else:
        entities = entities_raw  # type: ignore[assignment]
    msgs_raw = state.get("messages", [])
    messages = [Message.model_validate(m) if isinstance(m, dict) else m for m in msgs_raw]
    summary = compress_history(messages)
    source_raw = state.get("current_agent", AgentType.TRIAGE.value)
    if isinstance(source_raw, AgentType):
        source = source_raw
    else:
        source = AgentType(str(source_raw))
    if entities.urgency == "high" and entities.sentiment == "frustrated":
        priority = "critical"
    elif entities.urgency == "high":
        priority = "high"
    else:
        priority = "medium"
    return HandoverPayload(
        handover_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        source_agent=source,
        target_agent=target_agent,
        reason=reason,
        conversation_summary=summary,
        entities=entities,
        message_history=messages,
        context_snapshot=extract_handover_context(state),
        priority=priority,
    )
