"""Ingest KB JSON articles into Qdrant and BM25."""

from __future__ import annotations

import sys
from pathlib import Path

# Running as `python knowledge_base/ingest.py` puts `knowledge_base/` on sys.path[0], not the repo root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")
load_dotenv()

import json
import pickle
import re
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from config import load_settings
from retrieval.embedder import Embedder
from retrieval.qdrant_store import QdrantStore

ARTICLES_DIR = Path(__file__).resolve().parent / "articles"
BM25_OUT = Path(__file__).resolve().parent / "bm25_index.pkl"


def load_articles() -> list[dict]:
    files = sorted(ARTICLES_DIR.glob("KB-*.json"))
    articles: list[dict] = []
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            articles.append(json.load(f))
    return articles


def main() -> None:
    settings = load_settings()
    qcfg = settings.get("qdrant", {})
    collection = qcfg.get("collection_name", "clouddash_kb")

    articles = load_articles()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    texts: list[str] = []
    metadatas: list[dict] = []
    for art in articles:
        doc = f"# {art['title']}\n{art['content']}"
        chunks = splitter.split_text(doc)
        for idx, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append(
                {
                    "article_id": art["id"],
                    "title": art["title"],
                    "category": art["category"],
                    "tags": art.get("tags", []),
                    "chunk_index": idx,
                    "applies_to": art.get("applies_to", []),
                    "content": chunk,
                }
            )

    embedder = Embedder()
    vectors = embedder.embed_documents(texts)

    store = QdrantStore.instance()
    store.collection_name = collection
    store.vector_size = len(vectors[0]) if vectors else int(qcfg.get("vector_size", 1536))
    store.ensure_collection()

    ids: list[str] = []
    payloads: list[dict] = []
    for meta, vec in zip(metadatas, vectors):
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{meta['article_id']}:{meta['chunk_index']}"))
        ids.append(pid)
        payloads.append(
            {
                "article_id": meta["article_id"],
                "title": meta["title"],
                "category": meta["category"],
                "tags": meta["tags"],
                "chunk_index": meta["chunk_index"],
                "applies_to": meta["applies_to"],
                "content": meta["content"],
            }
        )
    store.upsert_chunks(ids=ids, vectors=vectors, payloads=payloads)

    tokenized = [[t for t in re.split(r"\W+", doc.lower()) if t] for doc in texts]
    bm25 = BM25Okapi(tokenized)
    with BM25_OUT.open("wb") as f:
        pickle.dump({"bm25": bm25, "metadatas": metadatas, "documents": texts}, f)

    info = store.get_collection_info()
    print("Ingestion complete")
    print(f"Articles: {len(articles)}")
    print(f"Chunks: {len(texts)}")
    print(f"Collection: {info}")


if __name__ == "__main__":
    main()
