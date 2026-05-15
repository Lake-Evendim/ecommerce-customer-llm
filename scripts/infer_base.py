from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.model_loader import ModelConfig, load_tokenizer_and_model
from src.llm.generator import GenerationConfig, ChatGenerator
from src.risk.rules import pre_check_user_query, post_check_answer, make_safe_fallback


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/infer.yaml")
    parser.add_argument("--query", required=True)
    parser.add_argument("--safe_mode", action="store_true", help="命中高风险时直接输出兜底回答")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    pre = pre_check_user_query(args.query) if cfg.get("risk", {}).get("enable_pre_check", True) else {
        "need_human": False, "risk_flags": [], "risk_level": "low"
    }

    if args.safe_mode and pre["need_human"]:
        result = {
            "model_type": "base",
            "query": args.query,
            "answer": make_safe_fallback(args.query, pre["risk_flags"]),
            "pre_risk": pre,
            "post_risk": {"has_risk": False, "risk_flags": [], "risk_level": "low"},
            "safe_fallback": True,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    model_cfg = ModelConfig(**cfg["model"])
    gen_cfg = GenerationConfig(**cfg.get("generation", {}))

    tokenizer, model = load_tokenizer_and_model(model_cfg)
    generator = ChatGenerator(tokenizer, model, gen_cfg)
    answer = generator.generate(args.query)

    post = post_check_answer(answer) if cfg.get("risk", {}).get("enable_post_check", True) else {
        "has_risk": False, "risk_flags": [], "risk_level": "low"
    }

    result = {
        "model_type": "base",
        "query": args.query,
        "answer": answer,
        "pre_risk": pre,
        "post_risk": post,
        "safe_fallback": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
