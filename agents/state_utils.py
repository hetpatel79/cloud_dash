"""Serialize ConversationState for LangGraph dict state."""

from __future__ import annotations

import json
from typing import Any

from models.schemas import AgentType, ConversationState, ExtractedEntities, IntentType, KBChunk, Message


def conversation_to_dict(cs: ConversationState) -> dict[str, Any]:
    return json.loads(cs.model_dump_json())


def dict_to_conversation(data: dict[str, Any]) -> ConversationState:
    payload = dict(data)
    payload["messages"] = [Message.model_validate(m) for m in payload.get("messages", [])]
    payload["entities"] = ExtractedEntities.model_validate(payload.get("entities", {}))
    payload["retrieved_chunks"] = [KBChunk.model_validate(c) for c in payload.get("retrieved_chunks", [])]
    if payload.get("intent") is not None:
        payload["intent"] = IntentType(payload["intent"])
    if payload.get("current_agent") is not None:
        payload["current_agent"] = AgentType(payload["current_agent"])
    if payload.get("previous_agent") is not None:
        payload["previous_agent"] = AgentType(payload["previous_agent"])
    if payload.get("handover_target_agent") is not None:
        payload["handover_target_agent"] = AgentType(payload["handover_target_agent"])
    return ConversationState.model_validate(payload)
