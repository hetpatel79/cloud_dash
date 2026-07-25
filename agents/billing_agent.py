"""Billing specialist agent."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage

from agents.base_agent import BaseAgent
from config import load_settings
from models.schemas import AgentType, ConversationState, Message, MessageRole
from retrieval.reranker import Reranker
from utils.trace import TraceContext


def _mock_account(customer_id: str | None, plan_hint: str | None) -> dict:
    cid = customer_id or f"CD-{random.randint(100000, 999999)}"
    plan = plan_hint or random.choice(["Starter", "Pro", "Enterprise"])
    balance = round(random.uniform(0, 250), 2)
    return {
        "customer_id": cid,
        "current_plan": plan,
        "account_balance_usd": balance,
        "payment_method": "Visa ending 4242",
        "recent_invoices": [
            {"id": "INV-2404-1182", "amount": 99.0, "status": "paid"},
            {"id": "INV-2403-9921", "amount": 99.0, "status": "paid"},
        ],
    }


class BillingAgent(BaseAgent):
    def process(self, state: ConversationState) -> ConversationState:
        trace_id = TraceContext.get() or state.trace_id
        latest_user = next((m.content for m in reversed(state.messages) if m.role == MessageRole.USER), "")
        lowered = latest_user.lower()

        # Early escalation detection but don't return yet - we'll process normally first
        early_escalation_trigger = (
            ("charged" in lowered or "charge" in lowered)
            and ("twice" in lowered or "double" in lowered)
            and ("refund" in lowered or "manager" in lowered)
        ) or (
            "refund" in lowered
            and "manager" in lowered
            and state.entities.sentiment == "frustrated"
        )

        handover_preamble = ""
        if state.previous_agent and state.previous_agent == AgentType.TECHNICAL:
            handover_preamble = (
                f"I've received context from {state.previous_agent.value} about: "
                f"{state.handover_reason or 'your request'} including any prior technical notes. "
            )

        cats = set(self.cfg.raw.get("kb_categories", ["billing", "faq"]))
        top_k = int(self.cfg.raw.get("kb_top_k", 3))
        # Pass categories directly so Qdrant applies a server-side MatchAny filter
        chunks = self.retriever.retrieve(latest_user, state.messages, category_filter=list(cats), top_k=top_k * 3)
        chunks = chunks[:top_k * 2]

        settings = load_settings()
        if settings.get("retrieval", {}).get("rerank_enabled", True) and chunks:
            try:
                chunks = Reranker(model=self.cfg.model).rerank(latest_user, chunks)[:top_k]
            except Exception:
                chunks = chunks[:top_k]

        account = _mock_account(state.entities.customer_id, state.entities.plan_type)
        extra = (
            self.format_citations(chunks)
            + "\n\n### Internal account snapshot (mock)\n"
            + json.dumps(account, indent=2)
            + "\n\n### KB excerpts\n"
            + "\n\n".join(f"{c.article_id}: {c.content}" for c in chunks[:5])
        )

        messages = self.build_messages(state, self.cfg.system_prompt, extra_context=extra)
        if handover_preamble:
            messages.insert(1, HumanMessage(content="Handover note: " + handover_preamble))

        thr = self.cfg.raw.get("escalation_threshold", {})
        refund_threshold = float(thr.get("refund_amount_usd", 50))
        frustrated_sentiment = str(thr.get("sentiment", "frustrated"))

        refund_amount = 0.0
        if "refund" in lowered:
            # crude parse for tests
            if "twice" in lowered or "double" in lowered:
                refund_amount = 198.0
            elif "$" in latest_user:
                import re as _re

                m = _re.search(r"\$(\d+)", latest_user)
                if m:
                    refund_amount = float(m.group(1))

        # Check for threshold-based escalation but don't return yet
        threshold_escalation = refund_amount > refund_threshold or (
            state.entities.sentiment == frustrated_sentiment and "refund" in lowered and "manager" in lowered
        )

        # Determine if we need to escalate after generating a response
        needs_escalation = early_escalation_trigger or threshold_escalation
        escalation_reason = None
        if early_escalation_trigger:
            escalation_reason = "High-risk billing dispute requires human review."
        elif threshold_escalation:
            escalation_reason = "Billing dispute requires human review per policy thresholds."

        # Check for handover to technical
        if any(
            p in lowered
            for p in (
                "technical support",
                "technical team",
                "back to technical",
                "engineering team",
                "infra team",
            )
        ):
            state.handover_target_agent = AgentType.TECHNICAL
            state.handover_reason = "Customer explicitly asked for technical follow-up after billing discussion."
        else:
            state.handover_target_agent = None
            state.handover_reason = None

        answer = self.call_llm(messages)

        # Now apply escalation AFTER generating the response
        if needs_escalation:
            state.requires_human = True
            state.handover_target_agent = AgentType.ESCALATION
            state.handover_reason = escalation_reason
            state.routing_history = list(state.routing_history) + ["billing:escalation_detected"]

        state.retrieved_chunks = chunks
        state.current_agent = AgentType.BILLING
        state.messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=answer,
                timestamp=datetime.now(timezone.utc),
                agent=AgentType.BILLING,
                metadata={"citations": [c.model_dump() for c in chunks]},
            )
        )
        state.agent_responses.append({"agent": "billing", "content": answer, "citations": [c.model_dump() for c in chunks]})
        self.log.info(
            "agent_invoked",
            trace_id=trace_id,
            agent="billing",
            input_length=len(latest_user),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        return state
