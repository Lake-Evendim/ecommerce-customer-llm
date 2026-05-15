from __future__ import annotations

import re
from typing import Any, Dict, List

from src.risk.rules import post_check_answer


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()

def phrase_hit(answer: str, phrase: str) -> bool:
    """
    先做严格子串匹配；
    如果没命中，再做少量人工规则的宽松匹配。
    """
    norm_answer = normalize_text(answer)
    norm_phrase = normalize_text(phrase)

    if norm_phrase in norm_answer:
        return True

    # 常见客服评测短语的宽松匹配
    soft_rules = {
        "查看物流详情": ["查看", "物流", "详情"],
        "订单页物流详情": ["订单页", "物流", "详情"],
        "联系人工客服": ["人工客服"],
        "联系客服核实": ["客服", "核实"],
        "进一步核实": ["进一步", "核实"],
        "不要承诺赔偿": ["不", "承诺", "赔偿"],
        "不要承诺具体时间": ["不", "承诺", "具体", "时间"],
    }

    if phrase in soft_rules:
        return all(normalize_text(k) in norm_answer for k in soft_rules[phrase])

    return False

def contains_any(answer: str, phrases: List[str]) -> bool:
    norm = normalize_text(answer)
    return any(normalize_text(p) in norm for p in (phrases or []) if p)


def must_include_hit(answer: str, must_include: List[str]) -> Dict[str, Any]:
    if not must_include:
        return {"total": 0, "hit": 0, "rate": None, "missing": []}

    missing = [p for p in must_include if not phrase_hit(answer, p)]
    hit = len(must_include) - len(missing)
    return {
        "total": len(must_include),
        "hit": hit,
        "rate": hit / len(must_include),
        "missing": missing,
    }


def must_not_include_violation(answer: str, must_not_include: List[str]) -> Dict[str, Any]:
    if not must_not_include:
        return {"total": 0, "violated": 0, "rate": 0.0, "violations": []}

    violations = [p for p in must_not_include if contains_any(answer, [p])]
    return {
        "total": len(must_not_include),
        "violated": len(violations),
        "rate": len(violations) / len(must_not_include),
        "violations": violations,
    }

def infer_human_transfer_type(answer: str) -> str:
    """
    返回：
    - none: 没有建议人工
    - conditional: 条件式建议人工，例如“如果仍未解决/超过时效/无法修改，请联系人工”
    - direct: 直接建议人工，例如“建议您联系人工客服处理”
    """
    text = normalize_text(answer)

    human_keywords = [
        "转人工",
        "人工客服",
        "联系人工",
        "人工处理",
        "在线客服",
        "联系客服",
        "平台客服",
        "售后客服",
        "客服协助",
        "客服核实",
        "客服为您核实",
        "进一步核实",
        "协助核实",
        "专员处理",
        "售后专员",
    ]

    if not any(normalize_text(k) in text for k in human_keywords):
        return "none"

    conditional_keywords = [
        "如果",
        "若",
        "如仍",
        "仍未",
        "无法",
        "不能",
        "超过",
        "长时间",
        "必要时",
        "仍然没有",
        "仍未更新",
        "页面无法",
        "未解决",
        "没有更新",
    ]

    # 只要回答里同时有人工关键词和条件词，先判为 conditional
    if any(normalize_text(k) in text for k in conditional_keywords):
        return "conditional"

    return "direct"

def infer_need_human_from_answer(answer: str) -> bool:
    keywords = [
        "转人工",
        "人工客服",
        "联系人工",
        "人工处理",
        "进一步核实",
        "客服协助核实",
    ]
    return contains_any(answer, keywords)


def evaluate_one_generation(sample: Dict[str, Any], answer: str) -> Dict[str, Any]:
    mi = must_include_hit(answer, sample.get("must_include", []) or [])
    mn = must_not_include_violation(answer, sample.get("must_not_include", []) or [])
    expected_need_human = bool(sample.get("need_human", False))
    predicted_human_transfer_type = infer_human_transfer_type(answer)

    # 原二分类保留，但只把 direct 算作强制人工
    category = sample.get("category")

    if predicted_human_transfer_type == "direct":
        predicted_need_human = True
    elif predicted_human_transfer_type == "conditional" and category in {
        "complaint",
        "manual_transfer",
        "quality_issue",
    }:
        predicted_need_human = True
    else:
        predicted_need_human = False

    post_risk = post_check_answer(answer)

    return {
        "id": sample.get("id"),
        "category": sample.get("category"),
        "eval_type": sample.get("eval_type"),
        "difficulty": sample.get("difficulty"),
        "query": sample.get("query"),
        "answer": answer,
        "must_include": sample.get("must_include", []),
        "must_not_include": sample.get("must_not_include", []),
        "must_include_hit_rate": mi["rate"],
        "must_include_missing": mi["missing"],
        "must_not_include_violation_rate": mn["rate"],
        "must_not_include_violations": mn["violations"],
        "expected_need_human": expected_need_human,
        "predicted_need_human": predicted_need_human,
        "predicted_human_transfer_type": predicted_human_transfer_type,
        "need_human_correct": expected_need_human == predicted_need_human,
        "answer_length": len(answer or ""),
        "empty_answer": not bool((answer or "").strip()),
        "post_risk_flags": post_risk["risk_flags"],
        "post_has_risk": post_risk["has_risk"],
    }


def summarize_generation_results(rows: List[Dict[str, Any]], model_name: str) -> str:
    total = len(rows)
    if total == 0:
        return f"# {model_name} Generation Eval Summary\n\nNo rows.\n"

    def avg(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    include_avg = avg([r.get("must_include_hit_rate") for r in rows])
    not_violate_rate = 1 - avg([r.get("must_not_include_violation_rate", 0.0) for r in rows])
    human_acc = sum(1 for r in rows if r.get("need_human_correct")) / total
    empty_rate = sum(1 for r in rows if r.get("empty_answer")) / total
    post_risk_rate = sum(1 for r in rows if r.get("post_has_risk")) / total

    by_cat = {}
    for r in rows:
        cat = r.get("category", "unknown")
        by_cat.setdefault(cat, []).append(r)

    lines = [
        f"# {model_name} Generation Eval Summary",
        "",
        "## Overall",
        "",
        f"- total: {total}",
        f"- must_include_avg_hit_rate: {include_avg if include_avg is not None else 'N/A'}",
        f"- must_not_include_non_violation_rate: {not_violate_rate}",
        f"- need_human_accuracy: {human_acc}",
        f"- empty_answer_rate: {empty_rate}",
        f"- post_risk_rate: {post_risk_rate}",
        "",
        "## By Category",
        "",
        "| category | count | must_include_avg_hit_rate | need_human_accuracy | post_risk_rate |",
        "|---|---:|---:|---:|---:|",
    ]

    for cat, items in sorted(by_cat.items()):
        cat_include = avg([r.get("must_include_hit_rate") for r in items])
        cat_human_acc = sum(1 for r in items if r.get("need_human_correct")) / len(items)
        cat_risk = sum(1 for r in items if r.get("post_has_risk")) / len(items)
        lines.append(
            f"| {cat} | {len(items)} | {cat_include if cat_include is not None else 'N/A'} | {cat_human_acc} | {cat_risk} |"
        )

    return "\n".join(lines) + "\n"
