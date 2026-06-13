from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agents.base_agent import EntityExtraction
from agents.triage_agent import TriageAgent, TriageDecision
from config import load_agents_config
from models.schemas import ConversationState, IntentType, Message, MessageRole


class FakeStructured:
    def __init__(self, schema):  # noqa: ANN001
        self.schema = schema

    def invoke(self, messages):  # noqa: ANN001
        if self.schema.__name__ == "EntityExtraction":
            return EntityExtraction(plan_type="Pro", customer_id="cust-123")
        if self.schema.__name__ == "TriageDecision":
            return self._triage()
        raise AssertionError("unexpected schema")

    def _triage(self) -> TriageDecision:  # pragma: no cover - overridden
        return TriageDecision(intent=IntentType.TECHNICAL, confidence=0.9, routing_reason="mock")


class FakeChat:
    def __init__(self, *args, **kwargs):
        pass

    def with_structured_output(self, schema):  # noqa: ANN001
        return FakeStructured(schema)


class FakeStructuredAmbiguous(FakeStructured):
    def _triage(self) -> TriageDecision:
        return TriageDecision(intent=IntentType.UNKNOWN, confidence=0.2, routing_reason="unclear")


class FakeChatAmbiguous(FakeChat):
    def with_structured_output(self, schema):  # noqa: ANN001
        if schema.__name__ == "EntityExtraction":
            return FakeStructured(schema)
        return FakeStructuredAmbiguous(schema)


def test_intent_classification_technical(monkeypatch: pytest.MonkeyPatch, sample_state: ConversationState) -> None:
    monkeypatch.setattr("agents.base_agent.chat_openai", lambda *a, **k: FakeChat())

    cfg = load_agents_config()
    agent = TriageAgent("triage", cfg, MagicMock())
    out = agent.process(sample_state)
    assert out.intent == IntentType.TECHNICAL


class FakeStructuredBilling(FakeStructured):
    def _triage(self) -> TriageDecision:
        return TriageDecision(intent=IntentType.BILLING, confidence=0.95, routing_reason="invoice")


class FakeChatBilling(FakeChat):
    def with_structured_output(self, schema):  # noqa: ANN001
        if schema.__name__ == "EntityExtraction":
            return FakeStructured(schema)
        return FakeStructuredBilling(schema)


def test_intent_classification_billing(monkeypatch: pytest.MonkeyPatch, sample_state: ConversationState) -> None:
    monkeypatch.setattr("agents.base_agent.chat_openai", lambda *a, **k: FakeChatBilling())

    cfg = load_agents_config()
    agent = TriageAgent("triage", cfg, MagicMock())
    out = agent.process(sample_state)
    assert out.intent == IntentType.BILLING


def test_entity_extraction(monkeypatch: pytest.MonkeyPatch, sample_state: ConversationState) -> None:
    monkeypatch.setattr("agents.base_agent.chat_openai", lambda *a, **k: FakeChat())

    cfg = load_agents_config()
    agent = TriageAgent("triage", cfg, MagicMock())
    out = agent.process(sample_state)
    assert out.entities.plan_type == "Pro"
    assert out.entities.customer_id == "cust-123"


def test_ambiguous_intent_asks_clarification(monkeypatch: pytest.MonkeyPatch, sample_state: ConversationState) -> None:
    monkeypatch.setattr("agents.base_agent.chat_openai", lambda *a, **k: FakeChatAmbiguous())

    cfg = load_agents_config()
    agent = TriageAgent("triage", cfg, MagicMock())
    out = agent.process(sample_state)
    assert out.triage_confidence < 0.7
    assert any(m.role == MessageRole.ASSISTANT for m in out.messages)
