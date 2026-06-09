"""Escalation packaging agent."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from utils.openai_compat import chat_openai

from agents.base_agent import BaseAgent
from handover.audit_log import AUDIT
from models.schemas import AgentType, ConversationState, Message, MessageRole
from utils.logger import get_logger
from utils.trace import TraceContext

logger = get_logger("agents.escalation")


def classify_priority(state: ConversationState) -> str:
    if state.entities.urgency == "high" or state.entities.sentiment == "frustrated":
        return "high"
    if state.entities.urgency == "medium":
        return "medium"
    return "low"


class EscalationAgent(BaseAgent):
    def process(self, state: ConversationState) -> ConversationState:
        trace_id = TraceContext.get() or state.trace_id
        transcript = "\n".join(f"{m.role.value}: {m.content}" for m in state.messages[-40:])
        model = chat_openai(model=self.cfg.model, temperature=self.cfg.temperature)
        summary = ""
        try:
            resp = model.invoke(
                [
                    SystemMessage(content=self.cfg.system_prompt),
                    HumanMessage(content=f"Summarize for human agents:\n{transcript}"),
                ]
            )
            summary = (getattr(resp, "content", "") or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("escalation_summary_failed", trace_id=trace_id, error=str(exc))
            summary = transcript[:1200]

        priority = classify_priority(state)
        recommended = "billing-manager" if state.current_agent == AgentType.BILLING else "technical-tier2"
        if "legal" in transcript.lower():
            recommended = "account-manager"

        escalation_id = str(uuid.uuid4())
        compact = escalation_id.replace("-", "")
        ticket = f"CD-{compact[:8].upper()}"
        package = {
            "escalation_id": escalation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "customer_id": state.entities.customer_id,
            "issue_summary": summary,
            "priority": priority,
            "sentiment": state.entities.sentiment,
            "conversation_history": [m.model_dump(mode="json") for m in state.messages],
            "previous_agents": list(state.routing_history),
            "recommended_team": recommended,
            "context_snapshot": {
                "intent": state.intent.value if state.intent else None,
                "entities": state.entities.model_dump(),
                "ticket": ticket,
            },
        }
        state.escalation_package = package
        state.requires_human = True
        AUDIT.log_escalation(package, conversation_id=state.conversation_id)

        eta = {"high": "2 hours", "medium": "4 hours", "low": "24 hours"}.get(priority, "4 hours")
        customer_msg = (
            f"We're sorry you've had trouble with CloudDash. I've opened escalation {ticket} for our team. "
            f"Priority is {priority}; typical first response is within {eta}. "
            f"A specialist will review the full thread and follow up directly."
        )
        state.current_agent = AgentType.ESCALATION
        state.messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=customer_msg,
                timestamp=datetime.now(timezone.utc),
                agent=AgentType.ESCALATION,
            )
        )
        state.agent_responses.append({"agent": "escalation", "content": customer_msg, "package": package})
        logger.info(
            "escalation_triggered",
            trace_id=trace_id,
            priority=priority,
            recommended_team=recommended,
        )
        return state
