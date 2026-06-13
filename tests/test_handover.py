from __future__ import annotations

from datetime import datetime, timezone

import pytest

from handover.audit_log import AUDIT
from handover.protocol import HandoverProtocol
from models.schemas import AgentType, ExtractedEntities, HandoverPayload, Message, MessageRole


def _payload(conv_id: str = "conv-1") -> HandoverPayload:
    now = datetime.now(timezone.utc)
    return HandoverPayload(
        handover_id="h1",
        timestamp=now,
        source_agent=AgentType.TECHNICAL,
        target_agent=AgentType.BILLING,
        reason="test",
        conversation_summary="summary",
        entities=ExtractedEntities(customer_id="c1"),
        message_history=[
            Message(role=MessageRole.USER, content="hi", timestamp=now),
        ],
        context_snapshot={"conversation_id": conv_id},
        priority="medium",
    )


def test_handover_preserves_entities() -> None:
    p = _payload()
    assert p.entities.customer_id == "c1"


def test_failed_handover_falls_back_to_triage() -> None:
    hp = HandoverProtocol()
    state = {"conversation_id": "conv-1", "routing_history": [], "current_agent": AgentType.TECHNICAL.value}
    out = hp.handle_failed_handover(state, "boom")
    assert out["current_agent"] == AgentType.TRIAGE.value


def test_audit_log_records_handover() -> None:
    AUDIT._memory.clear()  # type: ignore[attr-defined]
    AUDIT.log_handover(_payload("conv-audit-1"))
    hist = AUDIT.get_handover_history("conv-audit-1")
    assert any(e.get("type") == "handover" for e in hist)


def test_escalation_handover_is_terminal() -> None:
    hp = HandoverProtocol()
    assert hp.validate_handover(AgentType.ESCALATION, AgentType.TRIAGE) is False
