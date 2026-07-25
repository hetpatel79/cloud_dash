"""Abstract base class for CloudDash agents."""

from __future__ import annotations

import time
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Sequence, Type, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from utils.openai_compat import chat_openai, use_nvidia_runtime, with_retries
from pydantic import BaseModel, Field

from config import AgentConfig
from models.schemas import AgentType, ConversationState, ExtractedEntities, KBChunk, Message, MessageRole
from retrieval.retriever import HybridRetriever
from utils.logger import get_logger
from utils.trace import TraceContext

T = TypeVar("T", bound=BaseModel)

class EntityExtraction(BaseModel):
    customer_id: str | None = None
    plan_type: str | None = None
    issue_type: str | None = None
    product_references: list[str] = Field(default_factory=list)
    urgency: str = "medium"
    sentiment: str = "neutral"


class BaseAgent(ABC):
    """Shared utilities and LLM wiring."""

    def __init__(self, agent_key: str, agents_cfg: dict[str, AgentConfig], retriever: HybridRetriever) -> None:
        self.agent_key = agent_key
        self.cfg = agents_cfg[agent_key]
        self.retriever = retriever
        self.log = get_logger(f"agents.{agent_key}")

    @property
    def agent_type(self) -> AgentType:
        return AgentType(self.agent_key)

    def build_messages(
        self,
        state: ConversationState,
        system_prompt: str,
        extra_context: str | None = None,
    ) -> list[BaseMessage]:
        parts = [system_prompt]
        # Add language instruction if customer is non-English
        detected_lang = getattr(state.entities, "detected_language", None) or "english"
        if detected_lang and detected_lang.lower() != "english":
            parts.append(
                f"IMPORTANT: The customer's message is in {detected_lang}. "
                f"Respond in {detected_lang} while using KB article content. "
                f"Translate KB article steps into {detected_lang} for the customer."
            )
        if extra_context:
            parts.append("### Retrieved knowledge\n" + extra_context)
        messages: list[BaseMessage] = [SystemMessage(content="\n\n".join(parts))]
        for m in state.messages[-20:]:
            if m.role.value == "user":
                messages.append(HumanMessage(content=m.content))
            elif m.role.value == "assistant":
                if m.content.strip():
                    messages.append(AIMessage(content=m.content))
        
        # If the last message is an AIMessage, it means the assistant responded,
        # but we are being called again (likely due to a handover in the same turn).
        # We must add a HumanMessage prompt so the new agent knows it needs to respond
        # to its respective part of the user's inquiry.
        if messages and isinstance(messages[-1], AIMessage):
            latest_user = next((m.content for m in reversed(state.messages) if m.role == MessageRole.USER), "")
            if latest_user:
                messages.append(HumanMessage(content=(
                    f"Please address the {self.agent_key}-related portion of my request: '{latest_user}'"
                )))
        return messages

    def format_citations(self, chunks: Sequence[KBChunk]) -> str:
        if not chunks:
            return ""
        seen_ids: set[str] = set()
        unique_chunks: list[KBChunk] = []
        for c in chunks:
            if c.article_id not in seen_ids:
                seen_ids.add(c.article_id)
                unique_chunks.append(c)
        return " ".join(f"[{c.article_id}: {c.title}]" for c in unique_chunks)

    def call_llm(self, messages: list[BaseMessage]) -> str:
        trace_id = TraceContext.get() or "unknown"
        
        providers = [self.cfg.provider, "nvidia", "openai"]
        seen = set()
        providers = [x for x in providers if x and not (x in seen or seen.add(x))]
        
        last_exc = None
        for prov in providers:
            try:
                model = chat_openai(
                    model=self.cfg.model,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    provider=prov,
                )
                t0 = time.perf_counter()

                @with_retries(max_attempts=3, base_delay=1.0, backoff_factor=2.0)
                def _invoke() -> str:
                    resp = model.invoke(messages)
                    ms = int((time.perf_counter() - t0) * 1000)
                    usage = getattr(resp, "usage_metadata", {}) or {}
                    if not usage and hasattr(resp, "response_metadata"):
                        usage = resp.response_metadata.get("usage", {})
                    self.log.info(
                        "llm_call",
                        trace_id=trace_id,
                        model=model.model_name if hasattr(model, "model_name") else self.cfg.model,
                        prompt_tokens=int(usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0) or 0),
                        completion_tokens=int(usage.get("output_tokens", 0) or usage.get("completion_tokens", 0) or 0),
                        latency_ms=ms,
                        kb_event="chat",
                    )
                    return (getattr(resp, "content", None) or "").strip()

                return _invoke()
            except Exception as exc:
                last_exc = exc
                if "rate limit" in str(exc).lower() or "429" in str(exc).lower():
                    self.log.warning("llm_call_rate_limited", provider=prov, error=str(exc))
                    continue
                raise
        
        self.log.error("llm_call_all_providers_failed", trace_id=trace_id, error=str(last_exc))
        return "I'm having trouble reaching the language model right now. Please try again shortly."

    def get_structured_output(self, schema: Type[T], messages: list[BaseMessage]) -> T:
        """Helper to get structured output with fallback for models that don't natively support it."""
        trace_id = TraceContext.get() or "unknown"
        
        providers = [self.cfg.provider, "nvidia", "openai"]
        seen = set()
        providers = [x for x in providers if x and not (x in seen or seen.add(x))]
        
        last_exc = None
        for prov in providers:
            try:
                model = chat_openai(model=self.cfg.model, temperature=0.0, provider=prov)
                
                try:
                    # Try native structured output first (with retries)
                    @with_retries(max_attempts=3, base_delay=1.0, backoff_factor=2.0)
                    def _structured_invoke() -> T | None:
                        structured_llm = model.with_structured_output(schema)
                        parsed = structured_llm.invoke(messages)
                        return parsed if isinstance(parsed, schema) else None

                    result = _structured_invoke()
                    if result is not None:
                        return result
                except Exception as exc:
                    if "rate limit" in str(exc).lower() or "429" in str(exc).lower():
                        raise
                    self.log.warning("native_structured_output_failed", trace_id=trace_id, error=str(exc))

                # Fallback: Ask for JSON in prompt and parse manually
                json_instr = f"\n\nReturn ONLY valid JSON matching this schema: {json.dumps(schema.model_json_schema())}. No markdown fences, no explanation."
                
                # Clone messages to avoid mutating original list
                msgs = list(messages)
                if isinstance(msgs[-1], HumanMessage):
                    msgs[-1] = HumanMessage(content=msgs[-1].content + json_instr)
                else:
                    msgs.append(HumanMessage(content=json_instr))
                    
                raw = self.call_llm(msgs)
                
                # Robust JSON extraction
                try:
                    # Strip markdown fences
                    clean = raw.strip()
                    if clean.startswith("```"):
                        clean = re.sub(r"```(json)?", "", clean).replace("```", "").strip()
                    
                    # Find first { and last }
                    start = clean.find("{")
                    end = clean.rfind("}")
                    if start != -1 and end != -1:
                        clean = clean[start : end + 1]
                        
                    return schema.model_validate_json(clean)
                except Exception as exc:
                    self.log.error("structured_output_fallback_failed", trace_id=trace_id, raw=raw, error=str(exc))
                    raise
            except Exception as exc:
                last_exc = exc
                if "rate limit" in str(exc).lower() or "429" in str(exc).lower():
                    self.log.warning("structured_output_rate_limited", provider=prov, error=str(exc))
                    continue
                raise
                
        raise last_exc

    def extract_entities(self, text: str) -> ExtractedEntities:
        try:
            parsed = self.get_structured_output(
                EntityExtraction,
                [
                    SystemMessage(content="Extract structured entities from the latest user message."),
                    HumanMessage(content=text),
                ]
            )
            return ExtractedEntities(
                customer_id=parsed.customer_id,
                plan_type=parsed.plan_type,
                issue_type=parsed.issue_type,
                product_references=parsed.product_references,
                urgency=parsed.urgency if parsed.urgency in {"low", "medium", "high"} else "medium",  # type: ignore[arg-type]
                sentiment=parsed.sentiment
                if parsed.sentiment in {"neutral", "frustrated", "satisfied"}
                else "neutral",  # type: ignore[arg-type]
            )
        except Exception:
            return ExtractedEntities()

    @abstractmethod
    def process(self, state: ConversationState) -> ConversationState:
        raise NotImplementedError
