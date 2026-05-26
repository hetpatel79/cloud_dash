"""
LLM provider compatibility layer.

Routing table:
  provider=groq   → ChatGroq  (Llama via Groq — fast, reliable, for all agent calls)
  provider=nvidia → ChatNVIDIA (NVIDIA NIM — kept for embeddings & reranker)

NVIDIA NIM is still used for:
  - Embeddings  : nvidia/nv-embedqa-e5-v5   (retrieval/embedder.py)
  - Reranker    : nvidia/llama-3.2-nv-rerankqa-1b-v2  (retrieval/reranker.py)

All agent chat calls (triage, technical, billing, escalation, guardrails,
query-rewriting) go through Groq.
"""

from __future__ import annotations

import logging
import os
import time
from functools import wraps
from typing import Any, Callable, TypeVar

DEFAULT_NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_CHAT_MODEL = "meta/llama-3.3-70b-instruct"
DEFAULT_NVIDIA_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"

DEFAULT_GROQ_CHAT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GROQ_FAST_MODEL = "llama-3.1-8b-instant"

_log = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Retry / backoff decorator
# ---------------------------------------------------------------------------

def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    retryable_markers = (
        "429", "rate limit", "too many requests",
        "504", "502", "503",
        "timeout", "timed out", "connection error", "connection reset",
    )
    return any(m in msg for m in retryable_markers)


def with_retries(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator: retry with exponential backoff on transient provider errors."""
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    if attempt == max_attempts or not _is_retryable(exc):
                        raise
                    _log.warning(
                        "llm_transient_error_retry",
                        extra={
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "delay_s": delay,
                            "error": str(exc)[:200],
                        },
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
            raise RuntimeError("Unreachable")  # pragma: no cover
        return wrapper  # type: ignore[return-value]
    return decorator


# ---------------------------------------------------------------------------
# Key / URL helpers
# ---------------------------------------------------------------------------

def _groq_api_key() -> str | None:
    return os.getenv("GROQ_API_KEY")


def _nvidia_api_key() -> str | None:
    return os.getenv("NVIDIA_API_KEY")


def _openai_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY")


def use_groq_runtime() -> bool:
    """True when a Groq API key is present — preferred for all chat calls."""
    return bool(_groq_api_key())


def use_nvidia_runtime() -> bool:
    """True when an NVIDIA NIM key is present — used for embeddings / reranker."""
    return bool(_nvidia_api_key())


def nvidia_embeddings_enabled() -> bool:
    return use_nvidia_runtime()


def nvidia_base_url() -> str:
    return os.getenv("NVIDIA_API_BASE") or os.getenv("NVIDIA_BASE_URL") or DEFAULT_NVIDIA_API_BASE


# ---------------------------------------------------------------------------
# Model name resolution
# ---------------------------------------------------------------------------

def resolve_chat_model(configured_model: str, provider: str | None = None) -> str:
    """
    Map a model name to the appropriate provider's name.
    If provider=groq, always return the Groq model variant.
    If provider=nvidia (or no Groq key), return the configured NIM name.
    """
    eff_provider = provider or ("groq" if use_groq_runtime() else "nvidia")

    if eff_provider == "groq":
        # Remap any NIM model name to the Groq equivalent
        nim_to_groq: dict[str, str] = {
            "meta/llama-3.3-70b-instruct": "llama-3.3-70b-versatile",
            "meta/llama-3.1-8b-instruct":  "llama-3.1-8b-instant",
        }
        # Honour an explicit env override
        env_model = os.getenv("GROQ_CHAT_MODEL")
        if env_model:
            return env_model
        return nim_to_groq.get(configured_model, DEFAULT_GROQ_CHAT_MODEL)

    if eff_provider == "nvidia":
        groq_to_nim: dict[str, str] = {
            "llama-3.3-70b-versatile": "meta/llama-3.3-70b-instruct",
            "llama-3.1-8b-instant":  "meta/llama-3.1-8b-instruct",
        }
        # NVIDIA / OpenAI path
        env_model = os.getenv("NVIDIA_CHAT_MODEL")
        if use_nvidia_runtime() and env_model:
            return env_model
        return groq_to_nim.get(configured_model, configured_model)
    
    return configured_model


def resolve_embedding_model(yaml_model: str) -> str:
    if nvidia_embeddings_enabled():
        return os.getenv("NVIDIA_EMBEDDING_MODEL", DEFAULT_NVIDIA_EMBEDDING_MODEL)
    return yaml_model


# ---------------------------------------------------------------------------
# LLM factories
# ---------------------------------------------------------------------------

def chat_groq(
    *,
    model: str,
    temperature: float,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> Any:
    """Build a ChatGroq client. Fast, reliable for all agent chat calls."""
    from langchain_groq import ChatGroq  # lazy import — optional dep

    key = _groq_api_key()
    if not key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Get a free key at https://console.groq.com and add it to .env"
        )
    kw: dict[str, Any] = {"api_key": key, **kwargs}
    if max_tokens is not None:
        kw["max_tokens"] = max_tokens
    return ChatGroq(model=model, temperature=temperature, **kw)


def chat_nvidia(
    *,
    model: str,
    temperature: float,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> Any:
    """Build a ChatNVIDIA client. Used only for reranker / fallback."""
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    key = _nvidia_api_key()
    if not key:
        raise EnvironmentError("NVIDIA_API_KEY is not set.")

    timeout = float(os.getenv("LLM_REQUEST_TIMEOUT_SEC", "120"))
    model_kw = {"timeout": timeout, **kwargs}
    if max_tokens is not None:
        model_kw["max_tokens"] = max_tokens

    return ChatNVIDIA(
        model=model,
        temperature=temperature,
        api_key=key,
        base_url=nvidia_base_url(),
        **model_kw,
    )


def chat_openai_fallback(
    *,
    model: str,
    temperature: float,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> Any:
    """OpenAI fallback when neither Groq nor NVIDIA keys are present."""
    from langchain_openai import ChatOpenAI

    kw: dict[str, Any] = {}
    key = _openai_api_key()
    if key:
        kw["api_key"] = key
    if max_tokens is not None:
        kw["max_tokens"] = max_tokens
    return ChatOpenAI(model=model, temperature=temperature, **kw, **kwargs)


def chat_openai(
    *,
    model: str,
    temperature: float,
    max_tokens: int | None = None,
    provider: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Unified LLM factory.  Resolution order:
      1. explicit provider kwarg
      2. GROQ_API_KEY present → Groq
      3. NVIDIA_API_KEY present → NVIDIA NIM
      4. OPENAI_API_KEY / bare OpenAI
    """
    eff_provider = provider or ("groq" if use_groq_runtime() else
                                 "nvidia" if use_nvidia_runtime() else "openai")

    resolved_model = resolve_chat_model(model, provider=eff_provider)

    if eff_provider == "groq":
        return chat_groq(model=resolved_model, temperature=temperature,
                         max_tokens=max_tokens, **kwargs)
    if eff_provider == "nvidia":
        return chat_nvidia(model=resolved_model, temperature=temperature,
                           max_tokens=max_tokens, **kwargs)
    return chat_openai_fallback(model=resolved_model, temperature=temperature,
                                max_tokens=max_tokens, **kwargs)
