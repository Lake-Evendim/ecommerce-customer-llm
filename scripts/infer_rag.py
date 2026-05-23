from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

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


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _convert_doc(doc: Any) -> Dict[str, Any]:
    """
    兼容不同 retriever 返回结构：
    - dict
    - dataclass / object
    """
    if isinstance(doc, dict):
        return doc
    out = {}
    for k in ["doc_id", "chunk_id", "doc_type", "category", "title", "content", "score"]:
        if hasattr(doc, k):
            out[k] = getattr(doc, k)
    return out


def load_existing_retriever(rag_config_path: str):
    """
    尽量兼容你上一阶段的 RAG 检索器。

    推荐你的 src/rag/retriever.py 提供：
        retriever = Retriever(config_path="configs/rag.yaml")
        docs = retriever.retrieve(query, top_k=5)

    如果你的类名或方法名不同，优先改这里。
    """
    try:
        from src.rag.retriever import Retriever
    except Exception as e:
        raise ImportError(
            "无法导入 src.rag.retriever.Retriever。请确认上一阶段 RAG 检索模块存在，"
            "或者在 scripts/infer_rag.py 的 load_existing_retriever 中适配你的类名。"
        ) from e

    try:
        return Retriever(config_path=rag_config_path)
    except TypeError:
        try:
            return Retriever(rag_config_path)
        except TypeError:
            return Retriever()


def call_retriever(retriever, query: str, top_k: int) -> List[Dict[str, Any]]:
    for method_name in ["retrieve", "search", "query"]:
        if hasattr(retriever, method_name):
            method = getattr(retriever, method_name)
            try:
                docs = method(query=query, top_k=top_k)
            except TypeError:
                try:
                    docs = method(query, top_k)
                except TypeError:
                    docs = method(query)
            return [_convert_doc(d) for d in docs]
    raise AttributeError("Retriever must have one of methods: retrieve/search/query")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/infer.yaml")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument(
        "--model_type",
        default="dpo_rag",
        choices=["base_rag", "sft_rag", "dpo_rag"],
        help="用于标识当前推理模型类型，不影响模型加载，模型实际由 config 决定。",
    )
    parser.add_argument(
        "--safe_mode",
        action="store_true",
        help="命中高风险或低可信检索时直接输出兜底回答",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    rag_cfg = cfg.get("rag", {})
    top_k = args.top_k or int(rag_cfg.get("top_k", 5))
    score_threshold = float(rag_cfg.get("score_threshold", 0.35))
    rag_config_path = rag_cfg.get("rag_config_path", "configs/rag.yaml")

    pre = pre_check_user_query(args.query) if cfg.get("risk", {}).get("enable_pre_check", True) else {
        "need_human": False, "risk_flags": [], "risk_level": "low"
    }

    retriever = load_existing_retriever(rag_config_path)
    retrieved_docs = call_retriever(retriever, args.query, top_k=top_k)
    retrieval_status = retrieval_check(retrieved_docs, score_threshold=score_threshold)

    fallback_flags = []
    fallback_flags.extend(pre.get("risk_flags", []))
    fallback_flags.extend(retrieval_status.get("risk_flags", []))

    if args.safe_mode and (pre["need_human"] or not retrieval_status["retrieval_reliable"]):
        result = {
            "model_type": args.model_type,
            "query": args.query,
            "answer": make_safe_fallback(args.query, fallback_flags),
            "retrieved_docs": retrieved_docs,
            "pre_risk": pre,
            "retrieval_status": retrieval_status,
            "post_risk": {"has_risk": False, "risk_flags": [], "risk_level": "low"},
            "safe_fallback": True,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    model_cfg = ModelConfig(**cfg["model"])
    gen_cfg = GenerationConfig(**cfg.get("generation", {}))

    tokenizer, model = load_tokenizer_and_model(model_cfg)
    generator = ChatGenerator(tokenizer, model, gen_cfg)

    messages = build_rag_messages(args.query, retrieved_docs)
    answer = generator.generate_from_messages(messages)

    post = post_check_answer(answer) if cfg.get("risk", {}).get("enable_post_check", True) else {
        "has_risk": False, "risk_flags": [], "risk_level": "low"
    }

    result = {
        "model_type": args.model_type,
        "query": args.query,
        "answer": answer,
        "retrieved_docs": retrieved_docs,
        "pre_risk": pre,
        "retrieval_status": retrieval_status,
        "post_risk": post,
        "safe_fallback": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
