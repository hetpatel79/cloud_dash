from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agents.base_agent import BaseAgent
from config import load_agents_config
from models.schemas import ConversationState, Message, MessageRole
from retrieval.embedder import Embedder
from retrieval.retriever import HybridRetriever


class DummyAgent(BaseAgent):
    def process(self, state: ConversationState) -> ConversationState:  # pragma: no cover - not used
        return state


def test_embedding_returns_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "retrieval.embedder._embed_query_cached",
        lambda *args, **kwargs: tuple([0.01] * 1536),
    )
    emb = Embedder()
    vec = emb.embed_query("hello")
    assert len(vec) == 1536


def test_hybrid_rrf_fuses_rankings(sample_kb_chunks: list) -> None:
    vec = [sample_kb_chunks[0]]
    bm = [sample_kb_chunks[1]]
    fused = HybridRetriever._rrf_fuse(vec, bm, top_k=2)
    assert len(fused) == 2


def test_query_rewriting_uses_history(monkeypatch: pytest.MonkeyPatch) -> None:
    from langchain_core.messages import AIMessage

    class FakeChat:
        def invoke(self, messages):  # noqa: ANN001
            return AIMessage(content="aws alerts credential update")

    monkeypatch.setattr("retrieval.retriever.chat_openai", lambda *a, **k: FakeChat())
    r = HybridRetriever(embedder=MagicMock(), store=MagicMock())
    hist = [
        Message(
            role=MessageRole.USER,
            content="I updated credentials yesterday",
            timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        )
    ]
    rewritten = r.query_rewriting("alerts stopped", hist)
    assert "aws" in rewritten.lower()


def test_citation_formatting(sample_kb_chunks) -> None:
    cfg = load_agents_config()
    agent = DummyAgent("technical", cfg, MagicMock())
    text = agent.format_citations(sample_kb_chunks)
    assert "KB-005" in text
    assert "Alerts not firing" in text
