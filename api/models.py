"""API-specific Pydantic models."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from models.schemas import AgentType, ConversationState, KBChunk, Message


class HealthResponse(BaseModel):
    status: str
    kb_loaded: bool
    agents_ready: bool


class ConversationDetailResponse(BaseModel):
    conversation_id: str
    trace_id: str
    state: ConversationState
    messages: list[Message]


class HandoverListResponse(BaseModel):
    conversation_id: str
    events: list[dict[str, Any]]


class ErrorResponse(BaseModel):
    detail: str


class KBReloadResponse(BaseModel):
    status: str
    message: str = Field(default="Re-ingestion scheduled")
