"""Output-side safety checks."""

from __future__ import annotations

import re
from typing import Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from utils.openai_compat import chat_openai

from guardrails.input_guard import GuardResult
from models.schemas import KBChunk
from utils.logger import get_logger
from utils.trace import TraceContext

logger = get_logger("guardrails.output")

_CC = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)


def pii_redaction(text: str) -> str:
    redacted = _CC.sub("[REDACTED-CC]", text)
    redacted = _SSN.sub("[REDACTED-SSN]", redacted)
    redacted = _EMAIL.sub("[REDACTED-EMAIL]", redacted)
    return redacted


def hallucination_check(response: str, retrieved_chunks: Sequence[KBChunk], query: str) -> GuardResult:
    trace_id = TraceContext.get() or "unknown"
    llm = chat_openai(model="llama-3.1-8b-instant", temperature=0.0, provider="groq")
    kb = "\n".join(f"- {c.article_id}: {c.content[:400]}" for c in retrieved_chunks[:8])
    sys = SystemMessage(
        content=(
            "Given KB chunks and assistant response, list specific factual claims about CloudDash "
            "in the response that are NOT supported by the KB text. "
            "Return JSON {\"unsupported\": [str], \"has_issues\": true|false}."
        )
    )
    human = HumanMessage(content=f"User query: {query}\nKB:\n{kb}\nResponse:\n{response}")
    try:
        resp = llm.invoke([sys, human])
        content = getattr(resp, "content", "") or ""
        has_issues = "\"has_issues\": true" in content.replace(" ", "").lower()
        if has_issues:
            logger.warning(
                "guard_triggered",
                trace_id=trace_id,
                guard_type="hallucination_risk",
                violation=content[:500],
                sanitized=False,
            )
            disclaimer = (
                "\n\nNote: Some details may not be covered in our public documentation. "
                "Please verify critical facts in the CloudDash console or contact support."
            )
            return GuardResult(
                passed=False,
                violation_type="unsupported_claims",
                reason=content[:800],
                sanitized_input=response + disclaimer,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("hallucination_check_failed", trace_id=trace_id, error=str(exc))
        return GuardResult(passed=True, sanitized_input=response)
    return GuardResult(passed=True, sanitized_input=response)


class OutputGuardrail:
    def __init__(self, pii: bool = True, hallucination: bool = True) -> None:
        self.pii = pii
        self.hallucination = hallucination

    def check(self, response: str, retrieved_chunks: Sequence[KBChunk], query: str) -> tuple[str, GuardResult | None]:
        text = pii_redaction(response) if self.pii else response
        if not self.hallucination:
            return text, None
        gr = hallucination_check(text, retrieved_chunks, query)
        if not gr.passed and gr.sanitized_input:
            return gr.sanitized_input, gr
        return text, None
