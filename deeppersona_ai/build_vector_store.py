"""
build_vector_store.py

Reads chunked_profiles.json, converts each text chunk into a vector using a local embedding model,
and stores it into a ChromaDB file database.

Dependencies installation:
  pip install sentence-transformers chromadb

Output: E:\fraud-detection2\vector_store\chroma.sqlite3
"""

import json
import os
import sys
import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb


# ---------- Configuration ----------
CHUNKED_DATA_PATH = "chunked_profiles.json"
VECTOR_STORE_DIR = "vector_store"
COLLECTION_NAME = "agent_profiles"
EMBEDDING_MODEL = os.getenv(
    "DEEP_PERSONA_EMBEDDING_MODEL", "all-mpnet-base-v2"
)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_CACHE = os.getenv(
    "DEEP_PERSONA_MODEL_CACHE",
    os.path.join(PROJECT_ROOT, ".runtime", "huggingface", "hub"),
)
# EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # Alternative (better for Chinese, also 384 dimensions)
# ---------- End of Configuration ----------


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=script_dir / CHUNKED_DATA_PATH
    )
    parser.add_argument(
        "--output-dir", type=Path, default=script_dir / VECTOR_STORE_DIR
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chunked_path = args.input.expanduser().resolve()
    store_path = args.output_dir.expanduser().resolve()

    # 1. Load chunked data
    if not os.path.exists(chunked_path):
        print(f"[ERROR] {chunked_path} not found, please run profile_chunker.py first")
        sys.exit(1)

    with open(chunked_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Read {len(chunks)} chunks")

    # 2. Load local embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL} ...")
    try:
        encoder = SentenceTransformer(
            EMBEDDING_MODEL,
            cache_folder=MODEL_CACHE,
            local_files_only=True,
        )
    except Exception as local_error:
        print(f"  Local model unavailable, trying the configured model source: {local_error}")
        encoder = SentenceTransformer(
            EMBEDDING_MODEL,
            cache_folder=MODEL_CACHE,
        )
    dim = encoder.get_sentence_embedding_dimension()
    print(f"  Model dimension: {dim}")

    # 3. Compute embeddings (CPU inference, finishes 40 texts in seconds)
    texts = [c["text"] for c in chunks]

    # Ensure text exists
    empty_idx = [i for i, t in enumerate(texts) if not t.strip()]
    if empty_idx:
        print(f"  [WARN] Found {len(empty_idx)} empty text chunks: {empty_idx}")
        print(f"    These chunks will be skipped")
        # Filter empty chunks
        texts = [t for i, t in enumerate(texts) if i not in empty_idx]
        chunks = [c for i, c in enumerate(chunks) if i not in empty_idx]

    print(f"\nComputing embeddings for {len(texts)} texts ...")
    embeddings = encoder.encode(texts, show_progress_bar=True)
    print(f"  Embedding shape: {embeddings.shape}")

    # 4. Write to ChromaDB
    os.makedirs(store_path, exist_ok=True)
    db = chromadb.PersistentClient(path=str(store_path))

    # Delete existing collection with the same name (ensure idempotence)
    try:
        db.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing collection: {COLLECTION_NAME}")
    except (ValueError, chromadb.errors.NotFoundError):
        pass

    collection = db.create_collection(
        COLLECTION_NAME,
        metadata={"embedding_model": EMBEDDING_MODEL},
    )

    # Prepare data
    ids = [f"agent_{c['agent_id']}_{c['section']}" for c in chunks]
    metadatas = [
        {"agent_id": c["agent_id"], "section": c["section"]}
        for c in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )

    print(f"\n Vector database creation completed")
    print(f"  Storage path: {store_path}")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Total vector count: {collection.count()}")

    # 5. Quick retrieval test
    print("\n" + "=" * 60)
    print("Retrieval Test")
    print("=" * 60)

    test_queries = [
        "economic researcher studying currency devaluation in Yemen",
        "woodworking and traditional carpentry in Slovakia",
        "supply chain management logistics negotiation",
        "elderly Singaporean gardening balcony chili kangkong",
        "teenager in Uruguay rescues stray dog",
    ]

    for query in test_queries:
        query_vec = encoder.encode([query])
        results = collection.query(
            query_embeddings=query_vec,
            n_results=2,
        )

        print(f"\nQuery: \"{query}\"")
        print(f"  Top-2 matches:")
        for doc_id, dist, meta in zip(
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            print(f"    [{doc_id}] Distance={dist:.4f}  agent={meta['agent_id']} section={meta['section']}")

    print(f"\n[OK] Build completed")


if __name__ == "__main__":
    main()
