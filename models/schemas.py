"""Shared Pydantic v2 models for CloudDash multi-agent support."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class AgentType(str, Enum):
    TRIAGE = "triage"
    TECHNICAL = "technical"
    BILLING = "billing"
    ESCALATION = "escalation"


class IntentType(str, Enum):
    TECHNICAL = "technical"
    BILLING = "billing"
    ACCOUNT = "account"
    GENERAL = "general"
    ESCALATION = "escalation"
    UNKNOWN = "unknown"


class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime
    agent: Optional[AgentType] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class KBChunk(BaseModel):
    article_id: str
    title: str
    category: str
    content: str
    score: float
    chunk_index: int


class ExtractedEntities(BaseModel):
    customer_id: Optional[str] = None
    plan_type: Optional[str] = None
    issue_type: Optional[str] = None
    product_references: List[str] = Field(default_factory=list)
    urgency: Literal["low", "medium", "high"] = "medium"
    sentiment: Literal["neutral", "frustrated", "satisfied"] = "neutral"
    detected_language: str = "english"


class ConversationState(BaseModel):
    """Shared conversational state across agents (API / persistence view)."""

    conversation_id: str
    trace_id: str
    messages: List[Message] = Field(default_factory=list)
    current_agent: AgentType = AgentType.TRIAGE
    previous_agent: Optional[AgentType] = None
    intent: Optional[IntentType] = None
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
    retrieved_chunks: List[KBChunk] = Field(default_factory=list)
    handover_reason: Optional[str] = None
    is_resolved: bool = False
    requires_human: bool = False
    escalation_package: Optional[dict[str, Any]] = None
    agent_responses: List[dict[str, Any]] = Field(default_factory=list)
    routing_history: List[str] = Field(default_factory=list)
    handover_target_agent: Optional[AgentType] = None
    triage_confidence: float = 1.0
    input_guard_failed: bool = False
    iteration_count: int = 0
    secondary_intents: List[IntentType] = Field(default_factory=list)


class HandoverPayload(BaseModel):
    handover_id: str
    timestamp: datetime
    source_agent: AgentType
    target_agent: AgentType
    reason: str
    conversation_summary: str
    entities: ExtractedEntities
    message_history: List[Message]
    context_snapshot: dict[str, Any]
    priority: Literal["low", "medium", "high", "critical"]


class AgentResponse(BaseModel):
    agent: AgentType
    content: str
    citations: List[KBChunk] = Field(default_factory=list)
    suggested_next_agent: Optional[AgentType] = None
    handover_required: bool = False
    handover_reason: Optional[str] = None
    is_resolved: bool = False
    requires_human: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationStartRequest(BaseModel):
    customer_id: Optional[str] = None
    initial_message: str
    plan_type: Optional[str] = None


class MessageRequest(BaseModel):
    conversation_id: Optional[str] = ""
    message: str


class ConversationResponse(BaseModel):
    conversation_id: str
    trace_id: str
    response: str
    current_agent: AgentType
    citations: List[KBChunk] = Field(default_factory=list)
    is_resolved: bool = False
    requires_human: bool = False
    escalation_package: Optional[dict[str, Any]] = None
    input_guard_failed: bool = False
    detected_language: str = "english"

