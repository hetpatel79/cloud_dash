"""LLM-based reranking of KB chunks."""

from __future__ import annotations

import json
import time
import re
from typing import Any, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from utils.openai_compat import chat_nvidia

from models.schemas import KBChunk
from utils.logger import get_logger
from utils.trace import TraceContext

logger = get_logger("retrieval.reranker")


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


class Reranker:
    """Cross-encoder style reranking via LLM scores."""

    def __init__(self, model: str | None = None) -> None:
        # Reranker stays on NVIDIA NIM — Groq has no dedicated reranker model.
        # Falls back gracefully to the NIM 70B chat model if the reranker endpoint
        # is unavailable, rather than crashing the whole pipeline.
        self.model_name = model or "meta/llama-3.3-70b-instruct"
        try:
            self._llm = chat_nvidia(model=self.model_name, temperature=0.0, max_tokens=2048)
        except EnvironmentError:
            # No NVIDIA key — fall back to Groq for reranking (lower quality but functional)
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
            
            # Update scores and sort
            reranked_chunks = []
            for c in chunks:
                new_score = score_map.get((c.article_id, c.chunk_index), 0.0)
                # Combine original retrieval score with LLM score (weighted or just replacement)
                # Here we replace as the LLM is expected to be more accurate
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
                kb_event="rerank",
            )
            return reranked_chunks
        except Exception as exc:  # noqa: BLE001
            logger.error("rerank_failed", trace_id=trace_id, error=str(exc))
            return list(chunks)
