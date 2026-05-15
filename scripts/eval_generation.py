from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.model_loader import ModelConfig, load_tokenizer_and_model
from src.llm.generator import GenerationConfig, ChatGenerator, build_rag_messages
from src.risk.rules import (
    pre_check_user_query,
    post_check_answer,
    retrieval_check,
    make_safe_fallback,
)
from src.evaluation.generation_metrics import (
    evaluate_one_generation,
    summarize_generation_results,
)

# 复用 infer_rag.py 中的兼容函数
from scripts.infer_rag import load_existing_retriever, call_retriever


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_badcases(path: Path, rows: List[Dict[str, Any]], model_type: str):
    badcases = []
    for r in rows:
        reasons = []
        if r.get("empty_answer"):
            reasons.append("empty_answer")
        if r.get("must_include_missing"):
            reasons.append("missing_must_include")
        if r.get("must_not_include_violations"):
            reasons.append("must_not_include_violation")
        if not r.get("need_human_correct", True):
            reasons.append("need_human_wrong")
        if r.get("post_has_risk"):
            reasons.append("post_risk")

        if reasons:
            badcases.append({
                "id": r.get("id"),
                "model": model_type,
                "category": r.get("category"),
                "eval_type": r.get("eval_type"),
                "query": r.get("query"),
                "answer": r.get("answer"),
                "badcase_reasons": reasons,
                "must_include_missing": r.get("must_include_missing", []),
                "must_not_include_violations": r.get("must_not_include_violations", []),
                "expected_need_human": r.get("expected_need_human"),
                "predicted_need_human": r.get("predicted_need_human"),
                "post_risk_flags": r.get("post_risk_flags", []),
            })

    write_jsonl(path, badcases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/infer.yaml")
    parser.add_argument("--model_type", choices=["base", "base_rag"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="调试时可只跑前 N 条")
    parser.add_argument("--safe_mode", action="store_true", help="启用风控兜底输出")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    eval_path = cfg["paths"]["eval_path"]
    outputs_dir = Path(cfg["paths"].get("outputs_dir", "outputs/eval_results"))
    badcases_dir = Path(cfg["paths"].get("badcases_dir", "outputs/badcases"))

    samples = list(read_jsonl(eval_path))
    if args.limit:
        samples = samples[: args.limit]

    model_cfg = ModelConfig(**cfg["model"])
    gen_cfg = GenerationConfig(**cfg.get("generation", {}))
    tokenizer, model = load_tokenizer_and_model(model_cfg)
    generator = ChatGenerator(tokenizer, model, gen_cfg)

    retriever = None
    rag_cfg = cfg.get("rag", {})
    if args.model_type == "base_rag":
        retriever = load_existing_retriever(rag_cfg.get("rag_config_path", "configs/rag.yaml"))

    rows = []
    for sample in tqdm(samples, desc=f"Evaluating {args.model_type}"):
        query = sample["query"]

        pre = pre_check_user_query(query) if cfg.get("risk", {}).get("enable_pre_check", True) else {
            "need_human": False, "risk_flags": [], "risk_level": "low"
        }

        retrieved_docs = []
        retrieval_status = None

        if args.model_type == "base":
            if args.safe_mode and pre["need_human"]:
                answer = make_safe_fallback(query, pre["risk_flags"])
            else:
                answer = generator.generate(query)

        else:
            top_k = int(rag_cfg.get("top_k", 5))
            score_threshold = float(rag_cfg.get("score_threshold", 0.35))
            retrieved_docs = call_retriever(retriever, query, top_k=top_k)
            retrieval_status = retrieval_check(retrieved_docs, score_threshold=score_threshold)

            fallback_flags = []
            fallback_flags.extend(pre.get("risk_flags", []))
            fallback_flags.extend(retrieval_status.get("risk_flags", []))

            if args.safe_mode and (pre["need_human"] or not retrieval_status["retrieval_reliable"]):
                answer = make_safe_fallback(query, fallback_flags)
            else:
                messages = build_rag_messages(query, retrieved_docs)
                answer = generator.generate_from_messages(messages)

        metrics = evaluate_one_generation(sample, answer)
        metrics.update({
            "model_type": args.model_type,
            "pre_risk": pre,
            "retrieved_docs": retrieved_docs,
            "retrieval_status": retrieval_status,
            "safe_mode": args.safe_mode,
        })
        rows.append(metrics)

    suffix = "_safe" if args.safe_mode else ""
    result_path = outputs_dir / f"{args.model_type}_generation_eval{suffix}.jsonl"
    summary_path = outputs_dir / f"{args.model_type}_generation_summary{suffix}.md"
    badcase_path = badcases_dir / f"{args.model_type}_generation_badcases{suffix}.jsonl"

    write_jsonl(result_path, rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        summarize_generation_results(rows, f"{args.model_type}{suffix}"),
        encoding="utf-8",
    )
    dump_badcases(badcase_path, rows, f"{args.model_type}{suffix}")

    print(f"Saved eval rows: {result_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved badcases: {badcase_path}")


if __name__ == "__main__":
    main()
