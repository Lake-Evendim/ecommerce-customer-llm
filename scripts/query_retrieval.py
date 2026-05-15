#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from src.rag.prompt_builder import build_rag_prompt
from src.rag.retriever import Retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query FAISS retriever")
    parser.add_argument("--config", default="configs/rag.yaml")
    parser.add_argument("--query", required=True)
    parser.add_argument("--category", default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--show_prompt", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = Retriever(config_path=args.config)
    docs = retriever.retrieve(query=args.query, top_k=args.top_k, category=args.category)

    print("检索结果：")
    for i, doc in enumerate(docs, start=1):
        print(json.dumps(
            {
                "rank": i,
                "score": doc.get("score"),
                "doc_id": doc.get("doc_id"),
                "chunk_id": doc.get("chunk_id"),
                "doc_type": doc.get("doc_type"),
                "category": doc.get("category"),
                "title": doc.get("title"),
                "content_preview": str(doc.get("content", ""))[:120],
            },
            ensure_ascii=False,
            indent=2,
        ))

    if args.show_prompt:
        print("\n====== RAG Prompt 预览 ======\n")
        print(build_rag_prompt(user_query=args.query, retrieved_docs=docs))


if __name__ == "__main__":
    main()
