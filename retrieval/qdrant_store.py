"""Qdrant vector store wrapper with lazy singleton client."""

from __future__ import annotations

import threading
from typing import Any, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from config import load_settings


class QdrantStore:
    """Singleton-style store around qdrant-client."""

    _lock = threading.Lock()
    _client: QdrantClient | None = None
    _collection_ready: set[str] = set()

    def __init__(self) -> None:
        self._settings = load_settings()
        qcfg = self._settings.get("qdrant", {})
        self.collection_name: str = qcfg.get("collection_name", "clouddash_kb")
        self.vector_size: int = int(qcfg.get("vector_size", 1536))
        self.mode: str = str(qcfg.get("mode", "memory"))
        self.url: str = str(qcfg.get("url", ""))

    @classmethod
    def instance(cls) -> "QdrantStore":
        with cls._lock:
            # simple module-level singleton
            if not hasattr(cls, "_singleton"):
                cls._singleton = cls()
            return cls._singleton  # type: ignore[attr-defined]

    def _get_client(self) -> QdrantClient:
        with self._lock:
            if QdrantStore._client is None:
                if self.mode == "url" and self.url:
                    QdrantStore._client = QdrantClient(url=self.url)
                else:
                    QdrantStore._client = QdrantClient(":memory:")
            return QdrantStore._client

    def ensure_collection(self) -> None:
        client = self._get_client()
        with self._lock:
            if self.collection_name in QdrantStore._collection_ready:
                return
            exists = False
            try:
                cols = client.get_collections().collections
                exists = any(c.name == self.collection_name for c in cols)
            except Exception:
                exists = False
            if not exists:
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qm.VectorParams(size=self.vector_size, distance=qm.Distance.COSINE),
                )
            QdrantStore._collection_ready.add(self.collection_name)

    def upsert_chunks(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, Any]],
    ) -> None:
        if vectors:
            self.vector_size = len(vectors[0])
        self.ensure_collection()
        client = self._get_client()
        points = [
            qm.PointStruct(id=pid, vector=list(vec), payload=dict(pl))
            for pid, vec, pl in zip(ids, vectors, payloads)
        ]
        client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        filter_by_category: str | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if query_vector:
            self.vector_size = len(query_vector)
        try:
            self.ensure_collection()
            client = self._get_client()
            flt: qm.Filter | None = None
            if filter_by_category:
                # Accept both a single string and a list for backward compatibility
                cats = [filter_by_category] if isinstance(filter_by_category, str) else list(filter_by_category)
                flt = qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="category",
                            match=qm.MatchAny(any=cats),
                        )
                    ]
                )
            query_vec = list(query_vector)
            # qdrant-client 2.x removed client.search(); use query_points (vector = nearest).
            if hasattr(client, "query_points"):
                resp = client.query_points(
                    collection_name=self.collection_name,
                    query=query_vec,
                    limit=top_k,
                    query_filter=flt,
                    with_payload=True,
                )
                hits = resp.points
            else:
                hits = client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vec,
                    limit=top_k,
                    query_filter=flt,
                    with_payload=True,
                )
            results: list[dict[str, Any]] = []
            for h in hits:
                payload = dict(h.payload or {})
                payload["_score"] = float(h.score)
                results.append(payload)
            return results
        except Exception as exc:  # noqa: BLE001
            from utils.logger import get_logger
            get_logger("retrieval.qdrant_store").warning("qdrant_search_failed", error=str(exc))
            return []

    def get_collection_info(self, ensure: bool = True) -> dict[str, Any]:
        client = self._get_client()
        if ensure:
            self.ensure_collection()
        try:
            info = client.get_collection(self.collection_name)
        except Exception:
            return {
                "name": self.collection_name,
                "points_count": 0,
                "status": "missing",
            }
        points_count = getattr(info, "points_count", None)
        return {
            "name": self.collection_name,
            "points_count": points_count,
            "status": str(getattr(info, "status", "")),
        }


def reset_dev_client() -> None:
    """Test helper to clear singleton."""
    with QdrantStore._lock:
        QdrantStore._client = None
        QdrantStore._collection_ready.clear()
        if hasattr(QdrantStore, "_singleton"):
            delattr(QdrantStore, "_singleton")
