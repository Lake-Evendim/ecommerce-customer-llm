#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import yaml
from tqdm import tqdm

from src.rag.retriever import Retriever
from src.rag.vector_store import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval with eval_set.jsonl")
    parser.add_argument("--config", default="configs/rag.yaml")
    parser.add_argument("--top_k", type=int, default=None)
    return parser.parse_args()


def normalize_doc_ids(values: Any) -> Set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        return {values}
    if isinstance(values, list):
        return {str(v) for v in values if str(v).strip()}
    return {str(values)}


def hit_at_k(expected: Set[str], retrieved: List[Dict[str, Any]], k: int) -> bool:
    if not expected:
        return False
    retrieved_ids = {str(doc.get("doc_id", "")) for doc in retrieved[:k]}
    retrieved_chunk_ids = {str(doc.get("chunk_id", "")) for doc in retrieved[:k]}
    return bool(expected & (retrieved_ids | retrieved_chunk_ids))


def infer_badcase_type(expected: Set[str], retrieved: List[Dict[str, Any]], sample: Dict[str, Any]) -> str | None:
    if not expected:
        return "missing_expected_docs_label"
    if not retrieved:
        return "rag_no_result"
    if not hit_at_k(expected, retrieved, len(retrieved)):
        return "rag_wrong_doc"

    top1_category = retrieved[0].get("category")
    sample_category = sample.get("category")
    if sample_category and top1_category and sample_category != top1_category:
        return "rag_category_mismatch"
    return None


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    retrieval_cfg = cfg.get("retrieval", {})
    top_k = int(args.top_k or retrieval_cfg.get("top_k", 5))

    eval_rows = read_jsonl(paths["eval_path"])
    if retrieval_cfg.get("eval_only_need_rag", True):
        eval_rows = [row for row in eval_rows if bool(row.get("need_rag", False))]

    retriever = Retriever(config_path=args.config)
    results: List[Dict[str, Any]] = []
    badcases: List[Dict[str, Any]] = []

    for sample in tqdm(eval_rows, desc="Retrieval eval"):
        query = sample.get("query") or sample.get("user_query") or ""
        if not query.strip():
            continue

        category = sample.get("category")
        retrieved = retriever.retrieve(query=query, top_k=top_k, category=category)
        expected = normalize_doc_ids(sample.get("expected_docs"))

        row = {
            "id": sample.get("id"),
            "category": category,
            "eval_type": sample.get("eval_type"),
            "difficulty": sample.get("difficulty"),
            "query": query,
            "expected_docs": sorted(expected),
            "retrieved_docs": [
                {
                    "rank": i + 1,
                    "doc_id": doc.get("doc_id"),
                    "chunk_id": doc.get("chunk_id"),
                    "category": doc.get("category"),
                    "title": doc.get("title"),
                    "score": doc.get("score"),
                }
                for i, doc in enumerate(retrieved)
            ],
            "hit@1": hit_at_k(expected, retrieved, 1),
            "hit@3": hit_at_k(expected, retrieved, min(3, top_k)),
            "hit@5": hit_at_k(expected, retrieved, min(5, top_k)),
            "top1_score": retrieved[0].get("score") if retrieved else None,
            "top1_category": retrieved[0].get("category") if retrieved else None,
            "badcase_type": infer_badcase_type(expected, retrieved, sample),
        }
        results.append(row)
        if row["badcase_type"]:
            badcases.append(row)

    Path(paths["retrieval_eval_output"]).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths["retrieval_eval_output"], results)
    write_jsonl(paths["retrieval_badcase_output"], badcases)

    total = len(results)
    hit1 = safe_mean([1.0 if r["hit@1"] else 0.0 for r in results])
    hit3 = safe_mean([1.0 if r["hit@3"] else 0.0 for r in results])
    hit5 = safe_mean([1.0 if r["hit@5"] else 0.0 for r in results])
    top1_scores = [r["top1_score"] for r in results if isinstance(r.get("top1_score"), float)]
    badcase_counter = Counter(r["badcase_type"] for r in badcases)

    by_category = defaultdict(list)
    for r in results:
        by_category[r.get("category") or "unknown"].append(r)

    lines = []
    lines.append("# RAG Retrieval Evaluation Summary")
    lines.append("")
    lines.append(f"- total samples: {total}")
    lines.append(f"- recall@1: {hit1:.4f}")
    lines.append(f"- recall@3: {hit3:.4f}")
    lines.append(f"- recall@5: {hit5:.4f}")
    lines.append(f"- avg top1 score: {safe_mean(top1_scores):.4f}")
    lines.append(f"- badcase count: {len(badcases)}")
    lines.append("")
    lines.append("## Badcase Types")
    for key, value in badcase_counter.most_common():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## By Category")
    lines.append("| category | count | recall@1 | recall@3 | recall@5 |")
    lines.append("|---|---:|---:|---:|---:|")
    for category, rows in sorted(by_category.items()):
        lines.append(
            f"| {category} | {len(rows)} | "
            f"{safe_mean([1.0 if r['hit@1'] else 0.0 for r in rows]):.4f} | "
            f"{safe_mean([1.0 if r['hit@3'] else 0.0 for r in rows]):.4f} | "
            f"{safe_mean([1.0 if r['hit@5'] else 0.0 for r in rows]):.4f} |"
        )

    summary_path = Path(paths["retrieval_summary_output"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("检索评测完成：")
    print(f"- 明细：{paths['retrieval_eval_output']}")
    print(f"- 汇总：{paths['retrieval_summary_output']}")
    print(f"- Badcase：{paths['retrieval_badcase_output']}")


if __name__ == "__main__":
    main()
