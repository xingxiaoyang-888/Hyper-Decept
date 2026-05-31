"""
build_vector_store.py

读取 chunked_profiles.json，用本地 embedding 模型将每块文本转为向量，
存入 ChromaDB 文件数据库。

依赖安装:
  pip install sentence-transformers chromadb

输出: E:\fraud-detection2\vector_store\chroma.sqlite3
"""

import json
import os
import sys

from sentence_transformers import SentenceTransformer
import chromadb


# ---------- 配置 ----------
CHUNKED_DATA_PATH = "chunked_profiles.json"
VECTOR_STORE_DIR = "vector_store"
COLLECTION_NAME = "agent_profiles"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"    # 384 维，~80MB，CPU 推理，免费
# EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # 备选（中文更优，同样384维）
# ---------- 配置结束 ----------


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chunked_path = os.path.join(script_dir, CHUNKED_DATA_PATH)
    store_path = os.path.join(script_dir, VECTOR_STORE_DIR)

    # 1. 加载 chunked 数据
    if not os.path.exists(chunked_path):
        print(f"[ERROR] 未找到 {chunked_path}，请先运行 profile_chunker.py")
        sys.exit(1)

    with open(chunked_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"读取 {len(chunks)} 个 chunks")

    # 2. 加载本地 embedding 模型
    print(f"\n加载 embedding 模型: {EMBEDDING_MODEL} ...")
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    dim = encoder.get_sentence_embedding_dimension()
    print(f"  模型维度: {dim}")

    # 3. 计算 embedding（CPU 推理，40 条文本秒级完成）
    texts = [c["text"] for c in chunks]

    # 确保文本存在
    empty_idx = [i for i, t in enumerate(texts) if not t.strip()]
    if empty_idx:
        print(f"  [WARN] 发现 {len(empty_idx)} 个空文本 chunk: {empty_idx}")
        print(f"     这些 chunk 将被跳过")
        # 过滤空 chunk
        texts = [t for i, t in enumerate(texts) if i not in empty_idx]
        chunks = [c for i, c in enumerate(chunks) if i not in empty_idx]

    print(f"\n计算 {len(texts)} 个 embedding ...")
    embeddings = encoder.encode(texts, show_progress_bar=True)
    print(f"  embedding 形状: {embeddings.shape}")

    # 4. 写入 ChromaDB
    os.makedirs(store_path, exist_ok=True)
    db = chromadb.PersistentClient(path=store_path)

    # 删除已存在的同名 collection（保证幂等）
    try:
        db.delete_collection(COLLECTION_NAME)
        print(f"  删除已有 collection: {COLLECTION_NAME}")
    except (ValueError, chromadb.errors.NotFoundError):
        pass

    collection = db.create_collection(COLLECTION_NAME)

    # 准备数据
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

    print(f"\n[OK] 向量数据库创建完成")
    print(f"  存储路径: {store_path}")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  总向量数: {collection.count()}")

    # 5. 快速检索测试
    print("\n" + "=" * 60)
    print("检索测试")
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

        print(f"\n查询: \"{query}\"")
        print(f"  Top-2 匹配:")
        for doc_id, dist, meta in zip(
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            print(f"    [{doc_id}] 距离={dist:.4f}  agent={meta['agent_id']} section={meta['section']}")

    print(f"\n[OK] 构建完成")


if __name__ == "__main__":
    main()
