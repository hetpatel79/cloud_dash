"""Reranking of KB chunks via the NVIDIA NIM dedicated rerank-QA endpoint,
with an LLM-prompted JSON-scoring fallback when that endpoint is unavailable.
"""

from __future__ import annotations

import json
import os
import time
import re
from typing import Any, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from utils.openai_compat import chat_nvidia, nvidia_base_url

from config import load_settings
from models.schemas import KBChunk
from utils.logger import get_logger
from utils.trace import TraceContext

logger = get_logger("retrieval.reranker")

# NVIDIA's hosted "nvidia/llama-3.2-nv-rerankqa-1b-v2" is the model this repo's
# docs/comments reference. NVIDIA has since flagged it deprecated in favour of
# nvidia/llama-nemotron-rerank-1b-v2, though the older endpoint is still live
# as of this writing. Override via settings.yaml -> retrieval.rerank_model if
# NVIDIA retires it.
DEFAULT_NVIDIA_RERANK_MODEL = "nvidia/llama-3.2-nv-rerankqa-1b-v2"


def _robust_json_extract(text: str) -> list[dict] | None:
    """Robustly extracts a JSON array from text, even with markdown or conversational noise."""
    if not text:
        return None

    # Try literal JSON first
    try:
        data = json.loads(text.strip())
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Regex to find the first [ and last ]
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    # Try stripping common markdown artifacts
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"```(json)?", "", clean).replace("```", "").strip()
        try:
            data = json.loads(clean)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return None


class _LLMFallbackReranker:
    """Cross-encoder style reranking via LLM-prompted JSON scores.

    Used only when the dedicated NVIDIA rerank-QA endpoint is unavailable
    (no NVIDIA_API_KEY, or the endpoint call fails).
    """

    def __init__(self, model: str | None = None) -> None:
        self.model_name = model or "meta/llama-3.3-70b-instruct"
        try:
            self._llm = chat_nvidia(model=self.model_name, temperature=0.0, max_tokens=2048)
        except EnvironmentError:
            from utils.openai_compat import chat_groq
            self.model_name = "llama-3.3-70b-versatile"
            self._llm = chat_groq(model=self.model_name, temperature=0.0, max_tokens=2048)

    def rerank(self, query: str, chunks: Sequence[KBChunk]) -> list[KBChunk]:
        trace_id = TraceContext.get() or "unknown"
        if not chunks:
            return []

        sys = SystemMessage(
            content=(
                "You are an expert search relevance evaluator. Your task is to score KB chunk relevance to the user query. "
                "Return ONLY a JSON array of objects with keys article_id (string), chunk_index (integer), "
                "and score (number 0.0 to 10.0) for each chunk. "
                "Output ONLY the JSON array. Do not include markdown code fences or any other text."
            )
        )
        payload = [
            {
                "article_id": c.article_id,
                "chunk_index": c.chunk_index,
                "title": c.title,
                "excerpt": c.content[:1000],
            }
            for c in chunks
        ]
        human = HumanMessage(
            content=f"Query: {query}\n\nChunks to evaluate:\n{json.dumps(payload, ensure_ascii=False)}"
        )

        t0 = time.perf_counter()
        try:
            resp = self._llm.invoke([sys, human])
            text = (getattr(resp, "content", "") or "").strip()

            arr = _robust_json_extract(text)
            if not arr:
                logger.warning("rerank_json_extraction_failed", trace_id=trace_id, raw_output=text[:200])
                return list(chunks)

            score_map: dict[tuple[str, int], float] = {}
            for item in arr:
                if not isinstance(item, dict):
                    continue
                try:
                    aid = str(item.get("article_id", ""))
                    cidx = int(item.get("chunk_index", -1))
                    if aid == "" or cidx == -1:
                        continue
                    score = float(item.get("score", 0.0))
                    score_map[(aid, cidx)] = score
                except (TypeError, ValueError):
                    continue

            reranked_chunks = []
            for c in chunks:
                new_score = score_map.get((c.article_id, c.chunk_index), 0.0)
                reranked_chunks.append(c.model_copy(update={"score": new_score}))

            reranked_chunks.sort(key=lambda x: x.score, reverse=True)

            ms = int((time.perf_counter() - t0) * 1000)
            usage = getattr(resp, "usage_metadata", {}) or {}
            if not usage and hasattr(resp, "response_metadata"):
                usage = resp.response_metadata.get("usage", {})

            logger.info(
                "llm_call",
                trace_id=trace_id,
                model=self.model_name,
                prompt_tokens=int(usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0) or 0),
                completion_tokens=int(usage.get("output_tokens", 0) or usage.get("completion_tokens", 0) or 0),
                latency_ms=ms,
                kb_event="rerank_llm_fallback",
            )
            return reranked_chunks
        except Exception as exc:  # noqa: BLE001
            logger.error("rerank_llm_fallback_failed", trace_id=trace_id, error=str(exc))
            return list(chunks)


class Reranker:
    """Reranks KB chunks using the dedicated NVIDIA NIM rerank-QA model
    (nvidia/llama-3.2-nv-rerankqa-1b-v2). Falls back to an LLM-prompted
    JSON-scoring approach if the NVIDIA rerank endpoint is unavailable.
    """

    def __init__(self, model: str | None = None, fallback_model: str | None = None) -> None:
        settings = load_settings()
        self.model_name = settings.get("retrieval", {}).get("rerank_model", DEFAULT_NVIDIA_RERANK_MODEL)
        self._nim_rerank = None
        self._fallback: _LLMFallbackReranker | None = None
        self._fallback_model_name = fallback_model or model

        nvidia_key = os.getenv("NVIDIA_API_KEY")
        if nvidia_key:
            try:
                from langchain_nvidia_ai_endpoints import NVIDIARerank

                self._nim_rerank = NVIDIARerank(
                    model=self.model_name,
                    api_key=nvidia_key,
                    base_url=nvidia_base_url(),
                    top_n=20,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("nvidia_rerank_init_failed", error=str(exc))
                self._nim_rerank = None

    def _get_fallback(self) -> _LLMFallbackReranker:
        if self._fallback is None:
            self._fallback = _LLMFallbackReranker(model=self._fallback_model_name)
        return self._fallback

    def rerank(self, query: str, chunks: Sequence[KBChunk]) -> list[KBChunk]:
        trace_id = TraceContext.get() or "unknown"
        if not chunks:
            return []

        if self._nim_rerank is not None:
            try:
                from langchain_core.documents import Document

                docs = [
                    Document(
                        page_content=c.content,
                        metadata={"article_id": c.article_id, "chunk_index": c.chunk_index},
                    )
                    for c in chunks
                ]
                t0 = time.perf_counter()
                self._nim_rerank.top_n = len(docs)
                reranked_docs = self._nim_rerank.compress_documents(documents=docs, query=query)
                ms = int((time.perf_counter() - t0) * 1000)

                by_key = {(c.article_id, c.chunk_index): c for c in chunks}
                out: list[KBChunk] = []
                for d in reranked_docs:
                    key = (d.metadata.get("article_id"), d.metadata.get("chunk_index"))
                    src = by_key.get(key)
                    if src is None:
                        continue
                    out.append(src.model_copy(update={"score": float(d.metadata.get("relevance_score", 0.0))}))

                logger.info(
                    "llm_call",
                    trace_id=trace_id,
                    model=self.model_name,
                    latency_ms=ms,
                    kb_event="rerank_nim",
                )
                if out:
                    return out
                logger.warning("nvidia_rerank_empty_result", trace_id=trace_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("nvidia_rerank_call_failed", trace_id=trace_id, error=str(exc))

        # Fallback: no NVIDIA key, init failed, or the rerank call itself failed.
        return self._get_fallback().rerank(query, chunks)
