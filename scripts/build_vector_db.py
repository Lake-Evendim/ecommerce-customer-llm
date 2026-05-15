#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.rag.vector_store import EmbeddingModel, FaissVectorStore, load_kb_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS vector DB from kb_all.jsonl")
    parser.add_argument("--config", default="configs/rag.yaml", help="RAG config path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    emb_cfg = cfg["embedding"]

    kb_path = Path(paths["kb_path"])
    docs = load_kb_documents(kb_path)
    print(f"读取知识库文档数：{len(docs)}")

    texts = [doc.text_for_embedding for doc in docs]
    doc_rows = [doc.to_json() for doc in docs]

    embedder = EmbeddingModel(
        model_name_or_path=emb_cfg["model_name_or_path"],
        device=emb_cfg.get("device", "auto"),
        normalize_embeddings=bool(emb_cfg.get("normalize_embeddings", True)),
        query_instruction=emb_cfg.get("query_instruction", ""),
    )
    embeddings = embedder.encode_documents(texts, batch_size=int(emb_cfg.get("batch_size", 32)))
    print(f"embedding shape：{embeddings.shape}")

    store = FaissVectorStore.build(embeddings=embeddings, documents=doc_rows)
    store.save(index_path=paths["faiss_index_path"], doc_store_path=paths["doc_store_path"])

    print("向量库构建完成：")
    print(f"- {paths['faiss_index_path']}")
    print(f"- {paths['doc_store_path']}")


if __name__ == "__main__":
    main()
