"""DeepPersona vector-store adapter used by :class:`SocialAgent`.

The framework imports this module from ``MultiAgent4Collusion-master/to_add_engine``.
The vector store itself lives at the repository root so it can be rebuilt without
modifying the upstream simulation package.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "agent_profiles"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE = PROJECT_ROOT / "deeppersona_ai" / "vector_store"
DEFAULT_MODEL_CACHE = PROJECT_ROOT / ".runtime" / "huggingface" / "hub"


def load_vector_store():
    """Open the persistent DeepPersona collection expected by the framework."""
    store_path = Path(
        os.getenv("DEEP_PERSONA_VECTOR_STORE", str(DEFAULT_STORE))
    ).expanduser().resolve()
    metadata_path = store_path / "chroma.sqlite3"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"DeepPersona Chroma metadata is missing: {metadata_path}. "
            "Run deeppersona_ai/profile_chunker.py and build_vector_store.py first."
        )
    client = chromadb.PersistentClient(path=str(store_path))
    return client.get_collection(COLLECTION_NAME)


@lru_cache(maxsize=4)
def _encoder(model_name: str) -> SentenceTransformer:
    """Reuse the same local encoder that was recorded in collection metadata."""
    cache_folder = Path(
        os.getenv("DEEP_PERSONA_MODEL_CACHE", str(DEFAULT_MODEL_CACHE))
    ).expanduser().resolve()
    return SentenceTransformer(
        model_name,
        cache_folder=str(cache_folder),
        local_files_only=True,
    )


def get_agent_specific_memory(
    collection,
    agent_id: int,
    current_topic: str,
    top_k: int = 7,
) -> list[dict]:
    """Retrieve only the current agent's persona chunks for the live context."""
    count = collection.count()
    if count == 0:
        return []

    metadata = collection.metadata or {}
    model_name = os.getenv(
        "DEEP_PERSONA_EMBEDDING_MODEL",
        str(metadata.get("embedding_model") or "all-mpnet-base-v2"),
    )
    query = current_topic.strip() or "general social media context"
    query_embedding = _encoder(model_name).encode([query])[0].tolist()
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(max(int(top_k), 1), count),
        where={"agent_id": int(agent_id)},
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "text": text,
            "section": item_metadata.get("section", "unknown"),
            "dist": float(distance),
        }
        for text, item_metadata, distance in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )
    ]
