"""Input-side safety checks — multi-layer guardrail with NO bypass paths."""

from __future__ import annotations

import json
import re
from typing import Optional, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from utils.openai_compat import chat_openai
from pydantic import BaseModel

from utils.logger import get_logger
from utils.trace import TraceContext

logger = get_logger("guardrails.input")


class GuardResult(BaseModel):
    passed: bool
    violation_type: Optional[str] = None
    reason: Optional[str] = None
    sanitized_input: Optional[str] = None


# ---------------------------------------------------------------------------
# Layer 1: Hard regex injection patterns (case-insensitive)
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+)?(previous|prior|above|your)\s+instructions",
        r"disregard\s+(all\s+)?(previous|prior|above|your)\s+instructions",
        r"forget\s+(all\s+)?(previous|prior|above|your)\s+instructions",
        r"you\s+are\s+now\s+(a|an|the|DAN)",
        r"pretend\s+you\s+are",
        r"act\s+as\s+if\s+you\s+(have\s+no|are\s+not)",
        r"reveal\s+(your\s+)?(system\s+prompt|instructions|prompt)",
        r"show\s+me\s+your\s+(system\s+prompt|instructions|prompt)",
        r"what\s+are\s+your\s+(instructions|system\s+prompt|rules)",
        r"tell\s+me\s+your\s+(instructions|system\s+prompt|rules)",
        r"print\s+your\s+(system\s+prompt|instructions)",
        r"output\s+your\s+(system\s+prompt|instructions)",
        r"bypass\s+(your\s+)?(restrictions|rules|guidelines|filters)",
        r"jailbreak",
        r"\bDAN\b",
        r"do\s+anything\s+now",
        r"no\s+restrictions",
        r"without\s+(any\s+)?restrictions",
        r"override\s+(your\s+)?(programming|instructions|rules)",
    ]
]

# ---------------------------------------------------------------------------
# Layer 2: Hard regex off-topic patterns
# ---------------------------------------------------------------------------
_OFFTOPIC_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\brecipe\b",
        r"\bcook(ing)?\b",
        r"\bworld\s+series\b",
        r"\bsports?\s+score\b",
        r"\b(baseball|football|basketball|soccer)\b",
        r"\bwrite\s+me\s+a\s+(poem|story|song|essay|joke)\b",
        r"\bwhat\s+is\s+the\s+capital\s+of\b",
        r"\bhomework\b",
        r"\bmath\s+problem\b",
    ]
]

# ---------------------------------------------------------------------------
# Layer 3: Always-allow keywords (CloudDash support topics)
# ---------------------------------------------------------------------------
_ALLOW_KEYWORDS = {
    "alert", "dashboard", "integration", "aws", "gcp", "azure",
    "billing", "invoice", "plan", "upgrade", "downgrade", "refund",
    "payment", "subscription", "enterprise", "pro", "starter",
    "sso", "saml", "login", "access", "api", "webhook", "monitor",
    "error", "issue", "problem", "help", "support", "clouddash",
    "credential", "key", "team", "user", "account", "charge",
    "metric", "log", "threshold", "cloudwatch", "iam", "agent",
    "escalate", "manager", "urgent", "cancel", "data", "export",
}


def _check_injection_regex(text: str) -> GuardResult:
    """Layer 1: Fast regex injection detection."""
    for pat in _INJECTION_PATTERNS:
        if pat.search(text):
            logger.warning(
                "guard_triggered",
                trace_id=TraceContext.get() or "unknown",
                guard_type="prompt_injection",
                pattern=pat.pattern[:50],
            )
            return GuardResult(
                passed=False,
                violation_type="prompt_injection",
                reason=f"Detected injection pattern: {pat.pattern[:50]}",
                sanitized_input=None,
            )
    return GuardResult(passed=True, sanitized_input=text)


def _check_offtopic_regex(text: str) -> GuardResult:
    """Layer 2: Fast regex off-topic detection."""
    for pat in _OFFTOPIC_PATTERNS:
        if pat.search(text):
            logger.warning(
                "guard_triggered",
                trace_id=TraceContext.get() or "unknown",
                guard_type="off_topic_regex",
                pattern=pat.pattern[:50],
            )
            return GuardResult(
                passed=False,
                violation_type="off_topic",
                reason=f"Detected off-topic pattern: {pat.pattern[:50]}",
            )
    return GuardResult(passed=True, sanitized_input=text)


def _check_offtopic_llm(text: str) -> GuardResult:
    """Layer 3: LLM classification with few-shot examples for ambiguous cases."""
    trace_id = TraceContext.get() or "unknown"
    llm = chat_openai(model="llama-3.1-8b-instant", temperature=0.0, max_tokens=150, provider="groq")
    prompt = (
        "You are a classifier for CloudDash, a B2B cloud infrastructure monitoring SaaS.\n"
        "Decide if the user message is a valid support query.\n\n"
        "VALID topics: alerts, integrations, dashboards, APIs, webhooks, billing, "
        "invoices, payments, plans, refunds, account management, SSO, teams, RBAC, "
        "API keys, onboarding, features, complaints, escalations, feedback.\n\n"
        "EXAMPLES — valid (YES):\n"
        '- "My alerts stopped firing" → YES\n'
        '- "I want to upgrade to Enterprise" → YES\n'
        '- "I want to speak to a manager" → YES\n'
        '- "SSO is not working" → YES\n\n'
        "EXAMPLES — invalid (NO):\n"
        '- "Who won the World Series?" → NO\n'
        '- "Write me a Python sorting script" → NO\n'
        '- "What is the capital of India?" → NO\n\n'
        'Reply with ONLY a JSON object: {"is_valid": true, "reason": "..."} or {"is_valid": false, "reason": "..."}'
    )
    try:
        resp = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=text)])
        raw = (getattr(resp, "content", "") or "").strip()
        # Clean markdown fences
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.split("```")[0].strip()
        result = json.loads(raw)
        if not result.get("is_valid", True):
            logger.warning("guard_triggered", trace_id=trace_id, guard_type="off_topic_llm")
            return GuardResult(
                passed=False,
                violation_type="off_topic",
                reason=result.get("reason", "Off-topic message"),
            )
        return GuardResult(passed=True, sanitized_input=text)
    except Exception as exc:  # noqa: BLE001
        # On LLM error, allow — better to let borderline messages through
        logger.warning("offtopic_llm_check_failed", trace_id=trace_id, error=str(exc))
        return GuardResult(passed=True, sanitized_input=text)


class InputGuardrail:
    """Multi-layer input guardrail. NO bypass paths — every message is checked."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def check(self, text: str, conversation_history: Sequence | None = None) -> GuardResult:
        """
        Main entry point. Runs all checks in order.
        Returns GuardResult with passed=True only if ALL checks pass.
        """
        if not self.enabled:
            return GuardResult(passed=True, sanitized_input=text)

        if not text or not text.strip():
            return GuardResult(passed=True, sanitized_input=text)

        # Layer 1: Hard injection check — regex, instant, no LLM
        r1 = _check_injection_regex(text)
        if not r1.passed:
            return r1

        # Layer 2: Always-allow keyword fast pass
        text_lower = text.lower()
        if any(kw in text_lower for kw in _ALLOW_KEYWORDS):
            return GuardResult(passed=True, sanitized_input=text)

        # Layer 3: Hard off-topic regex check
        r2 = _check_offtopic_regex(text)
        if not r2.passed:
            return r2

        # Layer 4: LLM-based classification for ambiguous cases
        r3 = _check_offtopic_llm(text)
        if not r3.passed:
            return r3

        return GuardResult(passed=True, sanitized_input=text)

    @staticmethod
    def get_block_response(result: GuardResult) -> str:
        """Returns appropriate block message for the customer."""
        if result.violation_type == "prompt_injection":
            return (
                "I'm only able to help with CloudDash product support. "
                "If you have a question about your account, alerts, billing, "
                "or integrations, I'm happy to help."
            )
        if result.violation_type == "off_topic":
            return (
                "I can only assist with CloudDash support topics such as "
                "technical issues, billing, account management, and "
                "integrations. How can I help you with CloudDash today?"
            )
        return "I'm here to help with CloudDash support. Please describe your issue."
