"""Configuration loader for CloudDash — YAML-backed settings and per-agent config.

This package was missing from the published repository; reconstructed from the
config keys actually referenced across agents/, retrieval/, guardrails/ and api/
(self.cfg.*, load_settings().get(...), cfg.raw.get(...)).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

_ROOT = Path(__file__).resolve().parent
_SETTINGS_PATH = _ROOT / "settings.yaml"
_AGENTS_PATH = _ROOT / "agents.yaml"


class AgentConfig(BaseModel):
    """Per-agent configuration: model, generation params, prompt, and raw extras."""

    model: str
    temperature: float = 0.2
    max_tokens: int | None = 1024
    provider: str | None = None
    system_prompt: str = ""
    raw: dict[str, Any] = {}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def load_settings() -> dict[str, Any]:
    """Global settings: retrieval, qdrant, guardrails, logging."""
    return _load_yaml(_SETTINGS_PATH)


@lru_cache(maxsize=1)
def load_agents_config() -> dict[str, AgentConfig]:
    """Per-agent configuration, keyed by agent_key (triage/technical/billing/escalation)."""
    raw = _load_yaml(_AGENTS_PATH)
    agents_raw: dict[str, Any] = raw.get("agents", raw)
    out: dict[str, AgentConfig] = {}
    for key, cfg in agents_raw.items():
        cfg = dict(cfg or {})
        known = {"model", "temperature", "max_tokens", "provider", "system_prompt"}
        extras = {k: v for k, v in cfg.items() if k not in known}
        out[key] = AgentConfig(
            model=cfg.get("model", "llama-3.3-70b-versatile"),
            temperature=cfg.get("temperature", 0.2),
            max_tokens=cfg.get("max_tokens", 1024),
            provider=cfg.get("provider"),
            system_prompt=cfg.get("system_prompt", ""),
            raw=extras,
        )
    return out


def reload_config() -> None:
    """Test/dev helper to clear cached config after editing YAML on disk."""
    load_settings.cache_clear()
    load_agents_config.cache_clear()
