"""LangGraph orchestration for CloudDash multi-agent support."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

import structlog

from agents.billing_agent import BillingAgent
from agents.escalation_agent import EscalationAgent
from agents.state_utils import conversation_to_dict, dict_to_conversation
from agents.technical_agent import TechnicalSupportAgent
from agents.triage_agent import TriageAgent
from config import load_agents_config, load_settings
from guardrails.input_guard import InputGuardrail
from guardrails.output_guard import OutputGuardrail
from handover.protocol import HandoverProtocol
from models.schemas import AgentType, ConversationState, IntentType, Message, MessageRole
from retrieval.retriever import HybridRetriever
from utils.logger import get_logger
from utils.trace import TraceContext

logger = get_logger("agents.orchestrator")


class GraphState(TypedDict, total=False):
    """LangGraph-compatible mirror of ConversationState (JSON-serializable)."""

    conversation_id: str
    trace_id: str
    messages: list[dict[str, Any]]
    current_agent: str
    previous_agent: str | None
    intent: str | None
    secondary_intents: list[str]
    entities: dict[str, Any]
    retrieved_chunks: list[dict[str, Any]]
    handover_reason: str | None
    is_resolved: bool
    requires_human: bool
    escalation_package: dict[str, Any] | None
    agent_responses: list[dict[str, Any]]
    routing_history: list[str]
    handover_target_agent: str | None
    triage_confidence: float
    input_guard_failed: bool
    iteration_count: int


class SupportOrchestrator:
    """Compiles and runs the LangGraph state machine."""

    def __init__(self, retriever: HybridRetriever | None = None) -> None:
        self.agents_cfg = load_agents_config()
        self.settings = load_settings()
        self.retriever = retriever or HybridRetriever()
        self.input_guard = InputGuardrail(
            enabled=self.settings.get("guardrails", {}).get("input_enabled", True)
        )
        self.output_guard = OutputGuardrail(
            pii=self.settings.get("guardrails", {}).get("pii_redaction", True),
            hallucination=self.settings.get("guardrails", {}).get("hallucination_check", True),
        )
        self.triage = TriageAgent("triage", self.agents_cfg, self.retriever)
        self.technical = TechnicalSupportAgent("technical", self.agents_cfg, self.retriever)
        self.billing = BillingAgent("billing", self.agents_cfg, self.retriever)
        self.escalation = EscalationAgent("escalation", self.agents_cfg, self.retriever)
        self.handover_protocol = HandoverProtocol()
        self.graph = self._build_graph()

    def _to_conv(self, state: GraphState) -> ConversationState:
        return dict_to_conversation(dict(state))

    def _to_graph(self, conv: ConversationState) -> GraphState:
        return conversation_to_dict(conv)  # type: ignore[return-value]

    def _input_guard_node(self, state: GraphState) -> GraphState:
        conv = self._to_conv(state)
        latest = next((m.content for m in reversed(conv.messages) if m.role == MessageRole.USER), "")
        history = [{"role": m.role.value, "content": m.content} for m in conv.messages]
        res = self.input_guard.check(latest, conversation_history=history)
        if not res.passed:
            conv.input_guard_failed = True
            conv.routing_history = list(conv.routing_history) + ["input_guard:blocked"]
            block_msg = self.input_guard.get_block_response(res)
            conv.messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=block_msg,
                    timestamp=datetime.now(timezone.utc),
                    agent=AgentType.TRIAGE,
                )
            )
            conv.current_agent = AgentType.TRIAGE
            logger.warning(
                "guard_blocked",
                violation_type=res.violation_type,
                reason=res.reason,
                trace_id=conv.trace_id,
            )
        else:
            conv.input_guard_failed = False
        return self._to_graph(conv)

    def _safety_check_node(self, state: GraphState) -> GraphState:
        """Bug 1: Max iterations guard."""
        conv = self._to_conv(state)
        if conv.iteration_count > 5:
            conv.requires_human = True
            conv.handover_reason = "Max iterations exceeded - routing to escalation"
            conv.routing_history.append("orchestrator:max_iterations_exceeded")
        return self._to_graph(conv)

    def _triage_node(self, state: GraphState) -> GraphState:
        conv = self._to_conv(state)
        conv.iteration_count += 1

        # Always re-triage to detect intent changes in follow-up messages
        # (e.g., "I was charged twice" → BILLING, then "my dashboard won't load" → TECHNICAL)
        conv = self.triage.process(conv)
        return self._to_graph(conv)

    def _technical_node(self, state: GraphState) -> GraphState:
        conv = self._to_conv(state)
        conv.iteration_count += 1
        if conv.handover_target_agent == AgentType.TECHNICAL and conv.current_agent == AgentType.BILLING:
            d = conversation_to_dict(conv)
            if self.handover_protocol.validate_handover(AgentType.BILLING, AgentType.TECHNICAL):
                try:
                    d = self.handover_protocol.execute_handover(
                        d, AgentType.TECHNICAL, conv.handover_reason or "Billing handover"
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("technical_handover_failed", error=str(exc))
                    d = self.handover_protocol.handle_failed_handover(d, str(exc))
            conv = dict_to_conversation(d)
        try:
            conv = self.technical.process(conv)
        except Exception as exc:  # noqa: BLE001
            logger.error("technical_agent_failed", error=str(exc))
            conv = self.handover_protocol.handle_failed_handover(conversation_to_dict(conv), str(exc))
            conv = dict_to_conversation(conv)
        return self._to_graph(conv)

    def _billing_node(self, state: GraphState) -> GraphState:
        conv = self._to_conv(state)
        conv.iteration_count += 1
        d = conversation_to_dict(conv)
        if conv.handover_target_agent == AgentType.BILLING and conv.current_agent == AgentType.TECHNICAL:
            if self.handover_protocol.validate_handover(AgentType.TECHNICAL, AgentType.BILLING):
                try:
                    d = self.handover_protocol.execute_handover(
                        d, AgentType.BILLING, conv.handover_reason or "Technical handover"
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("billing_handover_failed", error=str(exc))
                    d = self.handover_protocol.handle_failed_handover(d, str(exc))
            conv = dict_to_conversation(d)
        try:
            conv = self.billing.process(conv)
        except Exception as exc:  # noqa: BLE001
            logger.error("billing_agent_failed", error=str(exc))
            conv = self.handover_protocol.handle_failed_handover(conversation_to_dict(conv), str(exc))
            conv = dict_to_conversation(conv)
        return self._to_graph(conv)

    def _escalation_node(self, state: GraphState) -> GraphState:
        conv = self._to_conv(state)
        conv.iteration_count += 1
        try:
            conv = self.escalation.process(conv)
        except Exception as exc:  # noqa: BLE001
            logger.error("escalation_agent_failed", error=str(exc))
            conv = self.handover_protocol.handle_failed_handover(conversation_to_dict(conv), str(exc))
            conv = dict_to_conversation(conv)
        return self._to_graph(conv)

    def _output_guard_node(self, state: GraphState) -> GraphState:
        conv = self._to_conv(state)
        latest_user = next((m.content for m in reversed(conv.messages) if m.role == MessageRole.USER), "")
        for idx in range(len(conv.messages) - 1, -1, -1):
            msg = conv.messages[idx]
            if msg.role != MessageRole.ASSISTANT:
                continue
            text, gr = self.output_guard.check(msg.content, conv.retrieved_chunks, latest_user)
            final_text = gr.sanitized_input if gr and not gr.passed and gr.sanitized_input else text
            conv.messages[idx] = msg.model_copy(update={"content": final_text})
            break
        
        # Set is_resolved: False if escalated to human, True if handled by agent
        if conv.requires_human or conv.handover_target_agent == AgentType.ESCALATION:
            conv.is_resolved = False
        else:
            # Resolved if we have an agent response and no unmet requirements
            has_agent_response = any(m.role == MessageRole.ASSISTANT for m in conv.messages)
            conv.is_resolved = has_agent_response and not conv.requires_human
        
        return self._to_graph(conv)

    def _route_input(self, state: GraphState) -> Literal["triage", "output_guard"]:
        if state.get("input_guard_failed"):
            return "output_guard"
        return "triage"

    def _route_triage(self, state: GraphState) -> Literal["safety_check", "output_guard"]:
        """Route to safety check before proceeding to agents."""
        if state.get("input_guard_failed"):
            return "output_guard"
        return "safety_check"

    def _route_safety_check(self, state: GraphState) -> Literal["technical", "billing", "escalation", "output_guard"]:
        if state.get("requires_human"):
            return "escalation"
            
        intent = state.get("intent")
        if intent == IntentType.TECHNICAL.value:
            return "technical"
        if intent == IntentType.BILLING.value:
            return "billing"
        if intent == IntentType.ESCALATION.value:
            return "escalation"
        if intent in {IntentType.ACCOUNT.value, IntentType.GENERAL.value}:
            return "technical"
        
        # Low confidence or unknown intent
        if float(state.get("triage_confidence", 1.0)) < 0.7:
            return "output_guard"
        
        return "technical"  # Default to technical for fallthrough

    def _route_from_agent(self, state: GraphState) -> Literal["technical", "billing", "escalation", "output_guard"]:
        """Bug 1 Fix: Always exit to output_guard unless explicit handover is required."""
        if state.get("requires_human"):
            return "escalation"
            
        target = state.get("handover_target_agent")
        current = state.get("current_agent")
        
        if target and target != current:
            if target == AgentType.BILLING.value:
                return "billing"
            if target == AgentType.TECHNICAL.value:
                return "technical"
        
        return "output_guard"

    def _build_graph(self) -> Any:
        graph = StateGraph(GraphState)
        graph.add_node("input_guard", self._input_guard_node)
        graph.add_node("safety_check", self._safety_check_node)
        graph.add_node("triage", self._triage_node)
        graph.add_node("technical", self._technical_node)
        graph.add_node("billing", self._billing_node)
        graph.add_node("escalation", self._escalation_node)
        graph.add_node("output_guard", self._output_guard_node)

        graph.add_edge(START, "input_guard")
        graph.add_conditional_edges("input_guard", self._route_input, {"triage": "triage", "output_guard": "output_guard"})
        
        graph.add_conditional_edges("triage", self._route_triage, {"safety_check": "safety_check", "output_guard": "output_guard"})
        
        graph.add_conditional_edges(
            "safety_check",
            self._route_safety_check,
            {"technical": "technical", "billing": "billing", "escalation": "escalation", "output_guard": "output_guard"}
        )

        graph.add_conditional_edges(
            "technical",
            self._route_from_agent,
            {"billing": "billing", "escalation": "escalation", "output_guard": "output_guard"}
        )
        graph.add_conditional_edges(
            "billing",
            self._route_from_agent,
            {"technical": "technical", "escalation": "escalation", "output_guard": "output_guard"}
        )
        
        graph.add_edge("escalation", "output_guard")
        graph.add_edge("output_guard", END)
        return graph.compile()

    def run_conversation(self, state: ConversationState) -> ConversationState:
        TraceContext.set(state.trace_id)
        structlog.contextvars.bind_contextvars(
            trace_id=state.trace_id,
            conversation_id=state.conversation_id,
            agent="orchestrator",
        )
        try:
            initial: GraphState = self._to_graph(state)
            final = self.graph.invoke(initial, config={"recursion_limit": 25})
            return dict_to_conversation(dict(final))
        finally:
            structlog.contextvars.clear_contextvars()

    async def stream_conversation(self, state: ConversationState):
        TraceContext.set(state.trace_id)
        initial: GraphState = self._to_graph(state)
        async for chunk in self.graph.astream(initial, config={"recursion_limit": 25}):
            yield chunk


def build_default_orchestrator() -> SupportOrchestrator:
    return SupportOrchestrator()
