"""Lightweight runtime adapter for DeepPersona context retrieval.

The OASIS agent calls this module at action time.  It deliberately uses the
already generated ``chunked_profiles.json`` and TF-IDF so loading a persona
does not require ChromaDB, a model download, or a particular PyTorch build.
The offline embedding builder can still be used for larger experiments, but
it is not a prerequisite for the simulation to start.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_CHUNKS_PATH = Path(__file__).resolve().parent / "chunked_profiles.json"


@dataclass(frozen=True)
class PersonaIndex:
    chunks_by_agent: dict[int, list[dict[str, Any]]]


def load_vector_store(chunks_path: str | Path | None = None) -> PersonaIndex:
    """Load and validate the DeepPersona chunks used by the simulator.

    The function name is kept for compatibility with the existing OASIS
    integration, although the returned object is an in-memory lexical index.
    """

    configured_path = os.getenv("DEEP_PERSONA_CHUNKS_PATH")
    path = Path(chunks_path or configured_path or DEFAULT_CHUNKS_PATH)
    if not path.is_file():
        raise FileNotFoundError(
            f"DeepPersona chunks not found: {path}. "
            "Run deeppersona_ai/profile_chunker.py first."
        )

    with path.open("r", encoding="utf-8") as handle:
        chunks = json.load(handle)
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"DeepPersona chunks must be a non-empty list: {path}")

    grouped: dict[int, list[dict[str, Any]]] = {}
    for position, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"Invalid chunk at index {position}: expected object")
        try:
            agent_id = int(chunk["agent_id"])
            section = str(chunk["section"]).strip()
            text = str(chunk["text"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid DeepPersona chunk at index {position}") from exc
        if not section or not text:
            continue
        grouped.setdefault(agent_id, []).append(
            {"agent_id": agent_id, "section": section, "text": text}
        )

    if not grouped:
        raise ValueError(f"No usable DeepPersona chunks found in {path}")
    return PersonaIndex(chunks_by_agent=grouped)


def get_agent_specific_memory(
    collection: PersonaIndex,
    agent_id: int,
    current_topic: str,
    top_k: int = 7,
) -> list[dict[str, Any]]:
    """Return context-relevant chunks for one agent, including its summary."""

    if top_k <= 0:
        return []
    chunks = collection.chunks_by_agent.get(int(agent_id), [])
    if not chunks:
        return []

    documents = [chunk["text"] for chunk in chunks]
    query = (current_topic or "").strip()
    if query:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(documents + [query])
        similarities = cosine_similarity(matrix[:-1], matrix[-1]).ravel()
    else:
        similarities = [0.0] * len(chunks)

    ranked = sorted(
        enumerate(chunks),
        key=lambda item: (-float(similarities[item[0]]), item[0]),
    )

    selected_indexes: list[int] = []
    summary_index = next(
        (idx for idx, chunk in enumerate(chunks)
         if chunk["section"] == "summary"),
        None,
    )
    if summary_index is not None:
        selected_indexes.append(summary_index)
    for idx, _ in ranked:
        if idx not in selected_indexes:
            selected_indexes.append(idx)
        if len(selected_indexes) >= min(top_k, len(chunks)):
            break

    return [
        {
            **chunks[idx],
            "dist": 1.0 - float(similarities[idx]),
        }
        for idx in selected_indexes
    ]
