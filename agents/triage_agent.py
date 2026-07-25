"""Triage agent for intent classification."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from config import AgentConfig
from models.schemas import AgentType, ConversationState, IntentType, Message, MessageRole
from retrieval.retriever import HybridRetriever
from utils.trace import TraceContext


class TriageDecision(BaseModel):
    intent: IntentType
    secondary_intents: List[IntentType] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    routing_reason: str
    detected_language: str = Field(default="english")


class TriageAgent(BaseAgent):
    def process(self, state: ConversationState) -> ConversationState:
        trace_id = TraceContext.get() or state.trace_id
        latest_user = next((m.content for m in reversed(state.messages) if m.role == MessageRole.USER), "")
        entities = self.extract_entities(latest_user)
        if state.entities.plan_type and not entities.plan_type:
            entities.plan_type = state.entities.plan_type
        if state.entities.customer_id and not entities.customer_id:
            entities.customer_id = state.entities.customer_id
        state.entities = entities

        sys_content = (
            self.cfg.system_prompt
            + "\n\nIMPORTANT: The user message may be in any language. "
            "Identify the language AND classify the intent regardless of language. "
            "If the message is not in English, set detected_language to the language name (e.g. 'spanish'). "
            "Otherwise set detected_language to 'english'."
        )
        sys = SystemMessage(content=sys_content)
        try:
            decision = self.get_structured_output(
                TriageDecision,
                [sys, HumanMessage(content=f"Latest customer message:\n{latest_user}")]
            )
        except Exception:
            decision = TriageDecision(
                intent=IntentType.UNKNOWN,
                secondary_intents=[],
                confidence=0.0,
                routing_reason="parse_error",
                detected_language="english",
            )

        state.triage_confidence = float(decision.confidence)
        state.entities.detected_language = getattr(decision, "detected_language", "english")
        if decision.confidence < 0.7:
            clarify = (
                "Thanks for reaching out to CloudDash support. Could you share a bit more detail about whether "
                "this is mainly a technical issue, a billing question, or an account/access topic?"
            )
            state.messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=clarify,
                    timestamp=datetime.now(timezone.utc),
                    agent=AgentType.TRIAGE,
                )
            )
            state.intent = IntentType.UNKNOWN
            state.secondary_intents = []
            state.routing_history.append("triage:clarification_needed")
            state.current_agent = AgentType.TRIAGE
            return state

        state.intent = decision.intent
        # Deduplicate: remove primary from secondary list if accidentally included
        state.secondary_intents = [i for i in decision.secondary_intents if i != decision.intent]

        # Set current_agent based on detected intent for proper routing
        if decision.intent == IntentType.TECHNICAL:
            state.current_agent = AgentType.TECHNICAL
        elif decision.intent == IntentType.BILLING:
            state.current_agent = AgentType.BILLING
        elif decision.intent == IntentType.ESCALATION:
            state.current_agent = AgentType.ESCALATION
        else:
            # ACCOUNT, GENERAL, or other intents route to TECHNICAL
            state.current_agent = AgentType.TECHNICAL

        state.routing_history.append(
            f"triage:{decision.intent.value}:secondary={[i.value for i in state.secondary_intents]}:{decision.routing_reason}"
        )
        state.agent_responses.append(
            {
                "agent": "triage",
                "intent": decision.intent.value,
                "secondary_intents": [i.value for i in state.secondary_intents],
                "confidence": decision.confidence,
                "routing_reason": decision.routing_reason,
            }
        )
        self.log.info(
            "agent_invoked",
            trace_id=trace_id,
            agent="triage",
            primary_intent=decision.intent.value,
            secondary_intents=[i.value for i in state.secondary_intents],
            confidence=decision.confidence,
            routing_reason=decision.routing_reason,
            input_length=len(latest_user),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        return state
