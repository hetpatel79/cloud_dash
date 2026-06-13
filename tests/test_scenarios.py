from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from agents.billing_agent import BillingAgent
from agents.technical_agent import TechnicalSupportAgent
from config import load_agents_config
from handover.protocol import HandoverProtocol
from models.schemas import AgentType, ConversationState, ExtractedEntities, IntentType, Message, MessageRole
from retrieval.reranker import Reranker
from retrieval.retriever import HybridRetriever


def test_scenario_1_single_agent_technical(monkeypatch: pytest.MonkeyPatch, sample_kb_chunks) -> None:
    monkeypatch.setattr(
        TechnicalSupportAgent,
        "call_llm",
        lambda self, messages: (
            "Based on [KB-005: Alerts not firing after credential update], "
            "1) Open Integrations → AWS. 2) Re-test connection. 3) Validate IAM permissions."
        ),
    )
    monkeypatch.setattr(Reranker, "rerank", lambda self, q, chunks: chunks)

    retriever = HybridRetriever(embedder=MagicMock(), store=MagicMock())
    monkeypatch.setattr(retriever, "retrieve_with_vector_peak", lambda *a, **k: (sample_kb_chunks, 0.9))

    cfg = load_agents_config()
    agent = TechnicalSupportAgent("technical", cfg, retriever)
    now = datetime.now(timezone.utc)
    state = ConversationState(
        conversation_id="conv-s1",
        trace_id="trace-s1",
        messages=[
            Message(
                role=MessageRole.USER,
                content=(
                    "My CloudDash alerts stopped firing after I updated my AWS integration credentials "
                    "yesterday. I'm on the Pro plan."
                ),
                timestamp=now,
            )
        ],
        intent=IntentType.TECHNICAL,
        entities=ExtractedEntities(plan_type="Pro"),
    )
    out = agent.process(state)
    answer = out.messages[-1].content
    assert "KB-005" in answer
    assert "Integrations" in answer or "IAM" in answer


def test_scenario_2_cross_agent_handover(monkeypatch: pytest.MonkeyPatch, sample_kb_chunks) -> None:
    monkeypatch.setattr(TechnicalSupportAgent, "call_llm", lambda self, messages: "SSO guidance with KB-008.")
    monkeypatch.setattr(BillingAgent, "call_llm", lambda self, messages: "Billing upgrade guidance citing KB-011.")
    monkeypatch.setattr(Reranker, "rerank", lambda self, q, chunks: chunks)

    retriever = HybridRetriever(embedder=MagicMock(), store=MagicMock())
    monkeypatch.setattr(retriever, "retrieve_with_vector_peak", lambda *a, **k: (sample_kb_chunks, 0.9))
    monkeypatch.setattr(retriever, "retrieve", lambda *a, **k: sample_kb_chunks)

    cfg = load_agents_config()
    tech = TechnicalSupportAgent("technical", cfg, retriever)
    now = datetime.now(timezone.utc)
    msg = (
        "I want to upgrade from Pro to Enterprise, but first can you check if the SSO integration issue "
        "I reported last week has been resolved?"
    )
    state = ConversationState(
        conversation_id="conv-s2",
        trace_id="trace-s2",
        messages=[Message(role=MessageRole.USER, content=msg, timestamp=now)],
        intent=IntentType.TECHNICAL,
        entities=ExtractedEntities(plan_type="Pro"),
    )
    out_t = tech.process(state)
    assert out_t.handover_target_agent == AgentType.BILLING

    hp_state = {
        "conversation_id": "conv-s2",
        "trace_id": "trace-s2",
        "routing_history": [],
        "current_agent": AgentType.TECHNICAL.value,
        "messages": [m.model_dump(mode="json") for m in out_t.messages],
        "entities": out_t.entities.model_dump(),
        "handover_reason": out_t.handover_reason,
    }
    hp = HandoverProtocol()
    hp_state2 = hp.execute_handover(hp_state, AgentType.BILLING, out_t.handover_reason or "handover")
    assert any("handover:technical->billing" in x for x in hp_state2.get("routing_history", []))

    bill = BillingAgent("billing", cfg, retriever)
    conv = out_t.model_copy(deep=True)
    conv.previous_agent = AgentType.TECHNICAL
    conv.handover_reason = out_t.handover_reason
    out_b = bill.process(conv)
    assert "SSO" in out_b.messages[-1].content or "billing" in out_b.messages[-1].content.lower()


def test_scenario_3_escalation(monkeypatch: pytest.MonkeyPatch, sample_kb_chunks) -> None:
    monkeypatch.setattr(BillingAgent, "call_llm", lambda self, messages: "should not run")
    monkeypatch.setattr(Reranker, "rerank", lambda self, q, chunks: chunks)
    retriever = HybridRetriever(embedder=MagicMock(), store=MagicMock())
    monkeypatch.setattr(retriever, "retrieve", lambda *a, **k: sample_kb_chunks)

    cfg = load_agents_config()
    bill = BillingAgent("billing", cfg, retriever)
    now = datetime.now(timezone.utc)
    state = ConversationState(
        conversation_id="conv-s3",
        trace_id="trace-s3",
        messages=[
            Message(
                role=MessageRole.USER,
                content="I've been charged twice for April. I need an immediate refund and I want to speak to a manager.",
                timestamp=now,
            )
        ],
        entities=ExtractedEntities(sentiment="frustrated"),
        intent=IntentType.BILLING,
    )
    out = bill.process(state)
    assert out.requires_human is True
    assert out.handover_target_agent == AgentType.ESCALATION


def test_scenario_4_kb_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        TechnicalSupportAgent,
        "call_llm",
        lambda self, messages: (
            "I don't have specific documentation on this. I can escalate you to a human specialist."
        ),
    )
    monkeypatch.setattr(Reranker, "rerank", lambda self, q, chunks: chunks)

    retriever = HybridRetriever(embedder=MagicMock(), store=MagicMock())
    monkeypatch.setattr(retriever, "retrieve_with_vector_peak", lambda *a, **k: ([], 0.1))

    cfg = load_agents_config()
    agent = TechnicalSupportAgent("technical", cfg, retriever)
    now = datetime.now(timezone.utc)
    state = ConversationState(
        conversation_id="conv-s4",
        trace_id="trace-s4",
        messages=[
            Message(
                role=MessageRole.USER,
                content="Does CloudDash support integration with Datadog for cross-platform alerting?",
                timestamp=now,
            )
        ],
        intent=IntentType.TECHNICAL,
    )
    out = agent.process(state)
    answer = out.messages[-1].content.lower()
    assert "don't have specific documentation" in answer
