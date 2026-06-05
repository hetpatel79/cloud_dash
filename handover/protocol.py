"""Handover execution and validation."""

from __future__ import annotations

from typing import Any

from handover.audit_log import AUDIT
from handover.context_packager import package_handover
from models.schemas import AgentType, ConversationState
from utils.logger import get_logger

logger = get_logger("handover.protocol")


class HandoverProtocol:
    def validate_handover(self, source: AgentType, target: AgentType) -> bool:
        if source == AgentType.ESCALATION:
            return False
        if source == AgentType.TRIAGE:
            return True
        if {source, target} == {AgentType.TECHNICAL, AgentType.BILLING}:
            return True
        if target == AgentType.ESCALATION:
            return True
        return False

    def execute_handover(self, state: dict[str, Any], target_agent: AgentType, reason: str) -> dict[str, Any]:
        payload = package_handover(state, target_agent, reason)
        try:
            AUDIT.log_handover(payload)
        except Exception as exc:  # noqa: BLE001
            logger.error("handover_audit_failed", error=str(exc))
        state = dict(state)
        state["handover_reason"] = reason
        state["previous_agent"] = state.get("current_agent")
        state["current_agent"] = target_agent.value
        state["handover_target_agent"] = None
        state["routing_history"] = list(state.get("routing_history", [])) + [
            f"handover:{payload.source_agent.value}->{payload.target_agent.value}"
        ]
        return state

    def handle_failed_handover(self, state: dict[str, Any], error: str) -> dict[str, Any]:
        logger.error("handover_failed", error=error, conversation_id=state.get("conversation_id"))
        state = dict(state)
        state["current_agent"] = AgentType.TRIAGE.value
        state["routing_history"] = list(state.get("routing_history", [])) + ["handover_failed:triage_fallback"]
        state["handover_reason"] = None
        state["handover_target_agent"] = None
        return state
