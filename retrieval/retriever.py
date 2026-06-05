"""Hybrid BM25 + dense retrieval with query rewriting and RRF fusion."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from utils.openai_compat import chat_openai
from rank_bm25 import BM25Okapi

from models.schemas import KBChunk, Message
from retrieval.embedder import Embedder
from retrieval.qdrant_store import QdrantStore
from utils.logger import get_logger
from utils.trace import TraceContext

logger = get_logger("retrieval.retriever")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BM25_PATH = ROOT / "knowledge_base" / "bm25_index.pkl"
RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\W+", text.lower()) if t]


def _messages_to_lc(messages: Sequence[Message]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for m in messages[-12:]:
        if m.role.value == "user":
            out.append(HumanMessage(content=m.content))
        elif m.role.value == "assistant":
            out.append(AIMessage(content=m.content))
        else:
            out.append(SystemMessage(content=m.content))
    return out


class HybridRetriever:
    """Vector + BM25 hybrid retrieval with optional query rewriting."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: QdrantStore | None = None,
        bm25_path: Path | None = None,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.store = store or QdrantStore.instance()
        self.bm25_path = Path(bm25_path or DEFAULT_BM25_PATH)
        self._bm25: BM25Okapi | None = None
        self._bm25_meta: list[dict[str, Any]] = []
        self._bm25_docs: list[str] = []
        self._load_bm25()

    def _load_bm25(self) -> None:
        if not self.bm25_path.exists():
            logger.warning("bm25_index_missing", path=str(self.bm25_path))
            return
        with self.bm25_path.open("rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._bm25_meta = list(data["metadatas"])
        self._bm25_docs = list(data["documents"])

    def query_rewriting(self, user_query: str, conversation_history: Sequence[Message]) -> str:
        from config import load_settings

        if not load_settings().get("retrieval", {}).get("query_rewrite_llm", True):
            return user_query

        trace_id = TraceContext.get() or "unknown"
        # Use Groq 8B for query rewriting — simple keyword extraction, no need for 70B
        model = chat_openai(model="llama-3.1-8b-instant", temperature=0.0, max_tokens=150, provider="groq")
        
        sys = SystemMessage(
            content=(
                "You are a search query optimizer for a CloudDash cloud monitoring knowledge base.\n"
                "Convert the user message into a SHORT keyword search query.\n"
                "Stay CLOSE to the original intent. Do not rephrase as a question.\n"
                "Use technical nouns only. Max 8 words.\n\n"
                "EXAMPLES:\n"
                'Input: "My alerts stopped firing after I updated AWS credentials"\n'
                'Output: alerts not firing AWS credentials update\n\n'
                'Input: "CloudWatch integration fails with permission error"\n'
                'Output: CloudWatch integration permission error IAM\n\n'
                'Input: "How do I invite someone to my team?"\n'
                'Output: invite team member CloudDash\n\n'
                'Input: "I was charged twice for April refund"\n'
                'Output: duplicate charge refund billing April\n\n'
                'Input: "high CPU usage on my RDS instances"\n'
                'Output: high CPU RDS troubleshoot CloudDash metrics\n\n'
                "Use the conversation history to resolve pronouns like 'it' or 'that problem'.\n"
                "Output the search keywords only, no explanation, no punctuation."
            )
        )
        
        msgs: list[BaseMessage] = [sys, *_messages_to_lc(conversation_history), HumanMessage(content=user_query)]
        try:
            resp = model.invoke(msgs)
            rewritten = (getattr(resp, "content", None) or "").strip()
            
            # Post-process to remove conversational filler if model ignored system prompt
            rewritten = re.sub(r"^(is|how|can|what|where|why|does|if|please|i|want|to|know|about)\s", "", rewritten, flags=re.IGNORECASE)
            rewritten = rewritten.strip(".")
            
            if not rewritten:
                return user_query
                
            logger.info(
                "kb_retrieved",
                trace_id=trace_id,
                query=user_query,
                rewritten_query=rewritten,
                kb_event="query_rewrite_ok",
            )
            return rewritten
        except Exception as exc:  # noqa: BLE001
            logger.warning("query_rewrite_failed", trace_id=trace_id, error=str(exc))
            return user_query

    def _payload_to_chunk(self, payload: dict[str, Any], score: float, chunk_index: int) -> KBChunk:
        return KBChunk(
            article_id=str(payload.get("article_id", "")),
            title=str(payload.get("title", "")),
            category=str(payload.get("category", "")),
            content=str(payload.get("content", "")),
            score=float(score),
            chunk_index=int(payload.get("chunk_index", chunk_index)),
        )

    def vector_search(self, query: str, top_k: int = 5, category: str | list[str] | None = None) -> list[KBChunk]:
        trace_id = TraceContext.get() or "unknown"
        vec = self.embedder.embed_query(query)
        hits = self.store.search(vec, top_k=top_k, filter_by_category=category)
        chunks: list[KBChunk] = []
        top_score = 0.0
        for h in hits:
            score = float(h.get("_score", 0.0))
            top_score = max(top_score, score)
            chunks.append(
                self._payload_to_chunk(
                    {k: v for k, v in h.items() if k != "_score"},
                    score,
                    int(h.get("chunk_index", 0)),
                )
            )
        logger.info(
            "kb_retrieved",
            trace_id=trace_id,
            query=query,
            rewritten_query=query,
            chunks_found=len(chunks),
            top_score=top_score,
            kb_event="vector_search",
        )
        return chunks

    def bm25_search(self, query: str, top_k: int = 5) -> list[KBChunk]:
        trace_id = TraceContext.get() or "unknown"
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        chunks: list[KBChunk] = []
        for i in ranked:
            meta = self._bm25_meta[i]
            chunks.append(
                KBChunk(
                    article_id=str(meta.get("article_id", "")),
                    title=str(meta.get("title", "")),
                    category=str(meta.get("category", "")),
                    content=self._bm25_docs[i],
                    score=float(scores[i]),
                    chunk_index=int(meta.get("chunk_index", 0)),
                )
            )
        logger.info(
            "kb_retrieved",
            trace_id=trace_id,
            query=query,
            rewritten_query=query,
            chunks_found=len(chunks),
            top_score=float(max(scores)) if len(scores) else 0.0,
            kb_event="bm25_search",
        )
        return chunks

    @staticmethod
    def _rrf_fuse(vector_chunks: list[KBChunk], bm25_chunks: list[KBChunk], top_k: int) -> list[KBChunk]:
        def key(c: KBChunk) -> tuple[str, int]:
            return (c.article_id, c.chunk_index)

        scores: dict[tuple[str, int], float] = {}
        store: dict[tuple[str, int], KBChunk] = {}
        for rank, ch in enumerate(vector_chunks, start=1):
            k = key(ch)
            scores[k] = scores.get(k, 0.0) + 1.0 / (RRF_K + rank)
            store[k] = ch
        for rank, ch in enumerate(bm25_chunks, start=1):
            k = key(ch)
            scores[k] = scores.get(k, 0.0) + 1.0 / (RRF_K + rank)
            store[k] = ch
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        fused: list[KBChunk] = []
        for k, sc in ordered:
            ch = store[k].model_copy(update={"score": float(sc)})
            fused.append(ch)
        return fused

    def _do_hybrid(self, query: str, conversation_history: Sequence[Message], top_k: int, category: str | list[str] | None) -> tuple[list[KBChunk], float]:
        """Core hybrid search — vector + BM25 + RRF."""
        vec_hits = self.vector_search(query, top_k=top_k, category=category)
        peak_vector = max((c.score for c in vec_hits), default=0.0)
        bm_hits = self.bm25_search(query, top_k=top_k)
        fused = self._rrf_fuse(vec_hits, bm_hits, top_k)
        return fused, peak_vector

    def hybrid_search(
        self,
        query: str,
        conversation_history: Sequence[Message],
        top_k: int = 5,
        category: str | list[str] | None = None,
    ) -> tuple[list[KBChunk], float]:
        """Returns fused chunks and the max raw cosine score from the dense (vector) branch.
        
        Includes fallback: if rewritten query returns 0 chunks, retry with original query.
        If still empty and a category filter was applied, retry without the filter.
        """
        rewritten = self.query_rewriting(query, conversation_history)
        fused, peak = self._do_hybrid(rewritten, conversation_history, top_k, category)

        # Fallback 1: rewritten query found nothing → try original
        if not fused and rewritten != query:
            logger.warning("rewritten_query_no_results", original=query, rewritten=rewritten)
            fused, peak = self._do_hybrid(query, conversation_history, top_k, category)

        # Fallback 2: still nothing and category filter was active → drop filter
        if not fused and category:
            logger.warning("category_filter_no_results", query=query, filter=str(category))
            fused, peak = self._do_hybrid(query, conversation_history, top_k, None)

        return fused, peak

    def retrieve(
        self,
        query: str,
        conversation_history: Sequence[Message],
        category_filter: str | list[str] | None = None,
        top_k: int | None = None,
    ) -> list[KBChunk]:
        from config import load_settings

        settings = load_settings()
        k = top_k or int(settings.get("retrieval", {}).get("top_k", 5))
        chunks, _ = self.hybrid_search(query, conversation_history, top_k=k, category=category_filter)
        return chunks

    def retrieve_with_vector_peak(
        self,
        query: str,
        conversation_history: Sequence[Message],
        category_filter: str | list[str] | None = None,
        top_k: int | None = None,
    ) -> tuple[list[KBChunk], float]:
        """Single hybrid pass: KB chunks plus max dense-retrieval score (for low-confidence KB gating)."""
        from config import load_settings

        settings = load_settings()
        k = top_k or int(settings.get("retrieval", {}).get("top_k", 5))
        return self.hybrid_search(query, conversation_history, top_k=k, category=category_filter)

    def max_vector_score(self, query: str, conversation_history: Sequence[Message], category: str | list[str] | None = None) -> float:
        _, peak = self.hybrid_search(query, conversation_history, top_k=5, category=category)
        return peak
