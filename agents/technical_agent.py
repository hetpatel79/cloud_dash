"""Technical support agent with hybrid RAG."""

from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.messages import HumanMessage

from agents.base_agent import BaseAgent
from config import load_settings
from models.schemas import AgentType, ConversationState, IntentType, Message, MessageRole
from retrieval.reranker import Reranker
from utils.trace import TraceContext


class TechnicalSupportAgent(BaseAgent):
    def process(self, state: ConversationState) -> ConversationState:
        trace_id = TraceContext.get() or state.trace_id
        latest_user = next((m.content for m in reversed(state.messages) if m.role == MessageRole.USER), "")
        handover_preamble = ""
        if state.previous_agent and state.previous_agent not in (AgentType.TRIAGE, AgentType.TECHNICAL):
            handover_preamble = (
                f"I've received context from {state.previous_agent.value} about: "
                f"{state.handover_reason or 'your request'}. "
            )

        cats = set(self.cfg.raw.get("kb_categories", ["troubleshooting", "api", "faq", "account"]))
        top_k = int(self.cfg.raw.get("kb_top_k", 5))
        # Pass the category list directly so Qdrant applies a server-side filter
        chunks, max_score = self.retriever.retrieve_with_vector_peak(
            latest_user, state.messages, category_filter=list(cats), top_k=top_k * 3
        )
        chunks = chunks[: max(top_k * 2, 8)]
        settings = load_settings()
        if settings.get("retrieval", {}).get("rerank_enabled", True) and chunks:
            try:
                chunks = Reranker(model=self.cfg.model).rerank(latest_user, chunks)[:top_k]
            except Exception:
                chunks = chunks[:top_k]

        no_kb_vector = max_score < 0.5
        citations_text = self.format_citations(chunks)
        extra = citations_text + "\n\n" + "\n\n".join(f"{c.article_id}: {c.content}" for c in chunks[:6])
        if no_kb_vector:
            extra = (
                "Vector similarity did not exceed 0.5 against the knowledge base. "
                "You MUST include the exact sentence: "
                "\"I don't have specific documentation on this\" "
                "and offer human escalation. Do not invent integrations or features."
            )

        messages = self.build_messages(state, self.cfg.system_prompt, extra_context=extra)
        if handover_preamble:
            messages.insert(1, HumanMessage(content="Handover note: " + handover_preamble))

        answer = self.call_llm(messages)

        # ---------------------------------------------------------------
        # Handover logic: primary path uses secondary_intents from Triage.
        # Fallback: keyword scan for cases where state was built without triage
        # (e.g. unit tests, direct agent invocation).
        # ---------------------------------------------------------------
        secondary = list(state.secondary_intents or [])
        lowered = latest_user.lower()

        def _has_billing_signal() -> bool:
            billing_kws = (
                "invoice", "charged", "charge", "double", "twice", "overcharge",
                "refund", "billing", "subscription", "upgrade", "downgrade",
                "enterprise", "plan change",
            )
            return any(kw in lowered for kw in billing_kws)

        if IntentType.BILLING in secondary:
            state.handover_target_agent = AgentType.BILLING
            state.handover_reason = (
                "Technical issue addressed. Customer also requested a plan/billing change "
                "— transferring to Billing Agent with full context."
            )
        elif IntentType.ACCOUNT in secondary:
            state.handover_target_agent = AgentType.TECHNICAL
            state.handover_reason = "Secondary account-access topic detected; re-routing."
        elif not secondary and _has_billing_signal():
            # Fallback: no secondary_intents provided (pre-triage path), but message
            # contains clear billing/upgrade language — infer handover.
            state.handover_target_agent = AgentType.BILLING
            state.handover_reason = (
                "Message contains billing/upgrade request alongside technical query "
                "— transferring to Billing Agent."
            )
        else:
            state.handover_target_agent = None
            state.handover_reason = None

        state.retrieved_chunks = chunks
        state.current_agent = AgentType.TECHNICAL
        state.messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=answer,
                timestamp=datetime.now(timezone.utc),
                agent=AgentType.TECHNICAL,
                metadata={"citations": [c.model_dump() for c in chunks]},
            )
        )
        state.agent_responses.append(
            {"agent": "technical", "content": answer, "citations": [c.model_dump() for c in chunks]}
        )
        self.log.info(
            "agent_invoked",
            trace_id=trace_id,
            agent="technical",
            input_length=len(latest_user),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        return state
