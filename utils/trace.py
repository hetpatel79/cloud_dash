"""Trace and conversation identifiers."""

from __future__ import annotations

import uuid
from contextvars import ContextVar, Token
from typing import Any

_trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)


def generate_trace_id() -> str:
    return f"trace-{uuid.uuid4()}"


def generate_conversation_id() -> str:
    return f"conv-{uuid.uuid4()}"


class TraceContext:
    """Holds current trace_id for request-scoped logging."""

    @staticmethod
    def get() -> str | None:
        return _trace_id_var.get()

    @staticmethod
    def set(trace_id: str) -> Token[Any]:
        return _trace_id_var.set(trace_id)

    @staticmethod
    def reset(token: Token[Any] | None = None) -> None:
        if token is not None:
            _trace_id_var.reset(token)
        else:
            _trace_id_var.set(None)
