"""
ChromaDB vector store wrapper for VoiceOS file indexing.

Provides a thin interface over a persistent ChromaDB collection so the rest
of the codebase doesn't need to import chromadb directly.

Collection:  voice_os_files
Persist dir: ~/.voice_os/chroma_db/
Embeddings:  sentence-transformers/all-MiniLM-L6-v2  (~80 MB, downloaded once)

All public methods degrade gracefully when chromadb / sentence-transformers
are not installed — they log a warning and return empty / zero results.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PERSIST_DIR = Path.home() / ".voice_os" / "chroma_db"
_COLLECTION_NAME = "voice_os_files"


class VectorStore:
    """
    Thin wrapper around a persistent ChromaDB collection.

    Build once via the module-level ``get_vector_store()`` singleton helper.
    """

    def __init__(self) -> None:
        self._client = None
        self._collection = None
        self._ef = None          # embedding function instance
        self._ready = False
        self._init()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _init(self) -> None:
        try:
            import chromadb
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )
        except ImportError:
            logger.warning(
                "chromadb / sentence-transformers not installed — "
                "file content search will be unavailable. "
                "Run: pip install chromadb sentence-transformers"
            )
            return

        try:
            _PERSIST_DIR.mkdir(parents=True, exist_ok=True)
            self._ef = SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self._client = chromadb.PersistentClient(path=str(_PERSIST_DIR))
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )
            self._ready = True
            logger.info(
                "VectorStore ready — collection '%s' has %d documents.",
                _COLLECTION_NAME,
                self._collection.count(),
            )
        except Exception as exc:
            logger.error("VectorStore init failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Return True when ChromaDB is available and the collection is open."""
        return self._ready

    def count(self) -> int:
        """Number of documents currently in the collection."""
        if not self._ready:
            return 0
        try:
            return self._collection.count()
        except Exception as exc:
            logger.warning("VectorStore.count: %s", exc)
            return 0

    def upsert(self, documents: List[Dict[str, Any]]) -> None:
        """
        Insert or update a batch of documents.

        Each dict must contain:
            path     (str)  — absolute file path, used as a stable document ID
            text     (str)  — text to embed
            metadata (dict) — arbitrary key/value pairs stored alongside the vector
        """
        if not self._ready or not documents:
            return
        try:
            ids = [_path_id(d["path"]) for d in documents]
            texts = [d["text"] for d in documents]
            metas = [d.get("metadata", {}) for d in documents]
            self._collection.upsert(ids=ids, documents=texts, metadatas=metas)
            logger.debug("VectorStore: upserted %d documents.", len(documents))
        except Exception as exc:
            logger.error("VectorStore.upsert: %s", exc, exc_info=True)

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic similarity search.

        Returns a list of metadata dicts (each contains at least ``path`` and
        ``filename``) for the top ``n_results`` matches ordered by relevance.
        Returns an empty list when the store is empty or unavailable.
        """
        if not self._ready:
            return []
        try:
            total = self._collection.count()
            if total == 0:
                return []
            actual_n = min(n_results, total)
            kwargs: Dict[str, Any] = {
                "query_texts": [query],
                "n_results": actual_n,
                "include": ["metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where
            result = self._collection.query(**kwargs)
            return list(result.get("metadatas", [[]])[0])
        except Exception as exc:
            logger.error("VectorStore.search: %s", exc, exc_info=True)
            return []

    def delete_by_path(self, path: str) -> None:
        """Remove a single document by its file path."""
        if not self._ready:
            return
        try:
            self._collection.delete(ids=[_path_id(path)])
        except Exception as exc:
            logger.warning("VectorStore.delete_by_path(%r): %s", path, exc)

    def clear(self) -> None:
        """Wipe all documents from the collection and recreate it empty."""
        if not self._ready:
            return
        try:
            self._client.delete_collection(_COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("VectorStore cleared.")
        except Exception as exc:
            logger.error("VectorStore.clear: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _path_id(path: str) -> str:
    """Return a stable 32-char hex ID derived from the file path."""
    return hashlib.sha256(path.encode("utf-8", errors="replace")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Return the shared VectorStore instance (created on first call)."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
