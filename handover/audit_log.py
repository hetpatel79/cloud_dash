"""Handover audit logging."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models.schemas import HandoverPayload
from utils.logger import get_logger

logger = get_logger("handover.audit")

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "logs" / "handovers.jsonl"


class HandoverAuditLog:
    def __init__(self) -> None:
        self._memory: dict[str, list[dict[str, Any]]] = {}
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    def log_handover(self, payload: HandoverPayload) -> None:
        entry = {
            "type": "handover",
            "conversation_id": payload.context_snapshot.get("conversation_id", ""),
            "handover_id": payload.handover_id,
            "timestamp": payload.timestamp.isoformat(),
            "source": payload.source_agent.value,
            "target": payload.target_agent.value,
            "reason": payload.reason,
            "conversation_summary": payload.conversation_summary,
            "entities": payload.entities.model_dump(),
            "messages": [m.model_dump(mode="json") for m in payload.message_history],
            "context_snapshot": payload.context_snapshot,
            "priority": payload.priority,
        }
        cid = str(entry.get("conversation_id", ""))
        self._memory.setdefault(cid, []).append(entry)
        try:
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.error("handover_log_write_failed", error=str(exc))

    def log_escalation(self, escalation_package: dict[str, Any], conversation_id: str = "") -> None:
        entry = {"type": "escalation", "conversation_id": conversation_id, **escalation_package}
        self._memory.setdefault(conversation_id, []).append(entry)
        try:
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.error("escalation_log_write_failed", error=str(exc))

    def get_handover_history(self, conversation_id: str) -> list[dict[str, Any]]:
        disk: list[dict[str, Any]] = []
        if LOG_PATH.exists():
            with LOG_PATH.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("conversation_id") == conversation_id or obj.get("context_snapshot", {}).get(
                        "conversation_id"
                    ) == conversation_id:
                        disk.append(obj)
        return self._memory.get(conversation_id, []) + disk


AUDIT = HandoverAuditLog()
