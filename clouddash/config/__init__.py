"""Load YAML configuration for agents and platform settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent


@dataclass
class AgentConfig:
    """Wraps a single agent block from agents.yaml."""

    key: str
    raw: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.raw.get("name", self.key))

    @property
    def model(self) -> str:
        return str(self.raw.get("model", "meta/llama-3.3-70b-instruct"))

    @property
    def temperature(self) -> float:
        return float(self.raw.get("temperature", 0.2))

    @property
    def max_tokens(self) -> int:
        return int(self.raw.get("max_tokens", 800))

    @property
    def system_prompt(self) -> str:
        return str(self.raw.get("system_prompt", "")).strip()

    @property
    def provider(self) -> str:
        """LLM provider: 'groq' | 'nvidia' | 'openai'. Defaults to 'groq'."""
        return str(self.raw.get("provider", "groq"))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.raw)


def load_agents_config(path: Path | None = None) -> dict[str, AgentConfig]:
    path = path or CONFIG_DIR / "agents.yaml"
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    agents = data.get("agents", {})
    return {k: AgentConfig(key=k, raw=dict(v or {})) for k, v in agents.items()}


def load_settings(path: Path | None = None) -> dict[str, Any]:
    path = path or CONFIG_DIR / "settings.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
