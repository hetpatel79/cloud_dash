"""Pytest fixtures and shared helpers."""

from __future__ import annotations

import os

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from models.schemas import AgentType, ConversationState, ExtractedEntities, IntentType, KBChunk, Message, MessageRole


@pytest.fixture
def sample_kb_chunks() -> list[KBChunk]:
    return [
        KBChunk(
            article_id="KB-005",
            title="Alerts not firing after credential update",
            category="troubleshooting",
            content="Open Integrations → AWS and re-test IAM permissions.",
            score=0.9,
            chunk_index=0,
        ),
        KBChunk(
            article_id="KB-007",
            title="AWS CloudWatch integration failing",
            category="troubleshooting",
            content="Validate IAM policy includes cloudwatch:DescribeAlarms.",
            score=0.4,
            chunk_index=1,
        ),
    ]


@pytest.fixture
def sample_state(sample_kb_chunks: list[KBChunk]) -> ConversationState:
    now = datetime.now(timezone.utc)
    return ConversationState(
        conversation_id="conv-test",
        trace_id="trace-test",
        messages=[
            Message(role=MessageRole.USER, content="Alerts stopped after AWS credential update.", timestamp=now)
        ],
        entities=ExtractedEntities(plan_type="Pro"),
        retrieved_chunks=sample_kb_chunks,
        intent=IntentType.TECHNICAL,
        current_agent=AgentType.TRIAGE,
    )


@pytest.fixture
def mock_qdrant_store(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    from retrieval import qdrant_store as qs

    mock = MagicMock()
    mock.search.return_value = [
        {
            "article_id": "KB-005",
            "title": "Alerts not firing after credential update",
            "category": "troubleshooting",
            "content": "Re-auth integration and validate IAM.",
            "chunk_index": 0,
            "_score": 0.91,
        }
    ]
    mock.get_collection_info.return_value = {"points_count": 10, "name": "clouddash_kb", "status": "green"}

    monkeypatch.setattr(qs.QdrantStore, "instance", staticmethod(lambda: mock))
    return mock
