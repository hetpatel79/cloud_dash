"""Embeddings via NVIDIA NIM or OpenAI."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Sequence

from langchain_openai import OpenAIEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from config import load_settings
from utils.openai_compat import nvidia_base_url, resolve_embedding_model, use_nvidia_runtime


@lru_cache(maxsize=1)
def _get_nvidia_embeddings(model: str) -> NVIDIAEmbeddings:
    return NVIDIAEmbeddings(
        model=model,
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url=nvidia_base_url(),
        truncate="NONE"
    )


@lru_cache(maxsize=1)
def _get_openai_embeddings(model: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model)


@lru_cache(maxsize=8192)
def _embed_query_cached(model: str, text: str, is_nvidia: bool) -> tuple[float, ...]:
    if is_nvidia:
        return tuple(_get_nvidia_embeddings(model).embed_query(text))
    return tuple(_get_openai_embeddings(model).embed_query(text))


class Embedder:
    """LangChain-compatible embeddings wrapper."""

    def __init__(self, model: str | None = None) -> None:
        settings = load_settings()
        default_model = settings.get("retrieval", {}).get("embedding_model", "text-embedding-3-small")
        self.model = resolve_embedding_model(model or default_model)
        self.is_nvidia = use_nvidia_runtime()

    def embed_query(self, text: str) -> list[float]:
        return list(_embed_query_cached(self.model, text, self.is_nvidia))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        texts_list = list(texts)
        if not texts_list:
            return []
        if self.is_nvidia:
            return _get_nvidia_embeddings(self.model).embed_documents(texts_list)
        return _get_openai_embeddings(self.model).embed_documents(texts_list)
