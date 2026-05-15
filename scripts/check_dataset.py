#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
全量数据质量验收脚本。

检查对象：
    data/interim/cleaned_sft.jsonl
    data/processed/sft_train.jsonl
    data/processed/sft_valid.jsonl
    data/eval/eval_set.jsonl
    data/interim/dpo_candidates.jsonl
    data/processed/dpo_train.jsonl
    data/processed/dpo_valid.jsonl
    data/knowledge_base/kb_all.jsonl

输出：
    docs/dataset_quality_report.md
    docs/dataset_quality_report.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


VALID_CATEGORIES = {
    "return_refund", "logistics", "product_info", "quality_issue",
    "invoice", "coupon_price", "complaint", "manual_transfer",
}
VALID_EVAL_TYPES = {"sft", "rag", "safety", "human_transfer"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_REJECTED_TYPES = {
    "fabricated_policy", "compensation_commitment", "wrong_intent",
    "missing_action", "rude_tone", "fake_order_or_logistics_status",
    "over_marketing", "no_human_transfer", "unsafe_overpromise",
}
VALID_SEVERITIES = {"low", "medium", "high"}
VALID_KB_DOC_TYPES = {"faq", "sop", "product"}

PHONE_PATTERNS = [r"(?<!\d)1[3-9]\d{9}(?!\d)"]
ID_CARD_PATTERNS = [r"\b\d{17}[\dXx]\b", r"\b\d{15}\b"]
EMAIL_PATTERNS = [r"[\w\.-]+@[\w\.-]+\.\w+"]
ORDER_ID_PATTERNS = [
    r"订单号\s*[:：]?\s*[A-Za-z0-9_-]{5,}",
    r"订单\s*[:：]?\s*[A-Za-z0-9_-]{8,}",
    r"单号\s*[:：]?\s*[A-Za-z0-9_-]{8,}",
]
RISKY_ASSISTANT_PATTERNS = [
    r"(一定|肯定|保证|百分百|必须).{0,10}(退款|退货|送达|赔偿|赔付|解决|处理)",
    r"(明天|今天|后天).{0,8}(一定|肯定|保证).{0,8}(到|送达|退款|处理)",
    r"(三倍|十倍|假一赔十|全额).{0,8}(赔偿|赔付|退款)",
    r"(无条件退款|无条件退货|包退|包换|包赔)",
    r"(赔|赔偿|赔付|补偿)\s*\d+(?:\.\d+)?\s*(元|块钱|人民币|rmb|RMB)",
]
BAD_ANSWER_PHRASES = {"不知道", "不清楚", "随便", "自己看", "无法回答", "我也不知道", "不归我管"}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_jsonl(path: Path, required: bool = True) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    items: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    if not path.exists():
        if required:
            errors.append({"type": "missing_file", "path": str(path), "line": None, "message": "file does not exist"})
        return items, errors

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as e:
                errors.append({"type": "json_decode_error", "path": str(path), "line": line_no, "message": str(e)})
                continue
            if not isinstance(item, dict):
                errors.append({"type": "not_json_object", "path": str(path), "line": line_no, "message": "JSONL item is not an object"})
                continue
            item["_file"] = str(path)
            item["_line"] = line_no
            items.append(item)
    return items, errors


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", "", text)
    text = text.replace("？", "?").replace("！", "!").replace("。", ".")
    text = text.replace("，", ",").replace("、", ",")
    text = text.replace("：", ":").replace("；", ";")
    return text


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def contains_any_pattern(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def contains_sensitive_info(text: str) -> Tuple[bool, str]:
    checks = [("phone", PHONE_PATTERNS), ("id_card", ID_CARD_PATTERNS), ("email", EMAIL_PATTERNS), ("order_id", ORDER_ID_PATTERNS)]
    for name, patterns in checks:
        if contains_any_pattern(text, patterns):
            return True, name
    return False, "ok"


def contains_risky_assistant_commitment(text: str) -> Tuple[bool, str]:
    if contains_any_pattern(text, RISKY_ASSISTANT_PATTERNS):
        return True, "risky_assistant_commitment"
    return False, "ok"


def count_by(items: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    return dict(Counter(str(item.get(key, "unknown")) for item in items))


def get_messages(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = item.get("messages", [])
    return messages if isinstance(messages, list) else []


def extract_text_by_role_from_messages(messages: Sequence[Dict[str, Any]], role: str) -> str:
    parts: List[str] = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == role:
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
    return "\n".join(parts).strip()


def extract_sft_texts(item: Dict[str, Any]) -> Tuple[str, str, str]:
    messages = get_messages(item)
    system = extract_text_by_role_from_messages(messages, "system")
    user = extract_text_by_role_from_messages(messages, "user")
    assistant = extract_text_by_role_from_messages(messages, "assistant")
    return system, user, assistant


def extract_prompt_user(prompt: Sequence[Dict[str, Any]]) -> str:
    return extract_text_by_role_from_messages(prompt, "user")


def extract_assistant_text(msg_list: Sequence[Dict[str, Any]]) -> str:
    return extract_text_by_role_from_messages(msg_list, "assistant")


def add_issue(
    issues: List[Dict[str, Any]], severity: str, dataset: str, issue_type: str,
    message: str, item_id: Optional[str] = None, path: Optional[str] = None,
    line: Optional[int] = None, extra: Optional[Dict[str, Any]] = None,
) -> None:
    issue = {"severity": severity, "dataset": dataset, "issue_type": issue_type, "message": message}
    if item_id is not None:
        issue["id"] = item_id
    if path is not None:
        issue["path"] = path
    if line is not None:
        issue["line"] = line
    if extra:
        issue["extra"] = extra
    issues.append(issue)


def add_file_errors_as_issues(issues: List[Dict[str, Any]], dataset: str, errors: Sequence[Dict[str, Any]]) -> None:
    for error in errors:
        add_issue(issues, "error", dataset, error.get("type", "file_error"), error.get("message", ""), path=error.get("path"), line=error.get("line"))


def validate_sft_item(item: Dict[str, Any], dataset: str, issues: List[Dict[str, Any]]) -> bool:
    ok = True
    item_id = str(item.get("id", ""))
    if item.get("category") not in VALID_CATEGORIES:
        ok = False
        add_issue(issues, "error", dataset, "invalid_category", f"invalid category: {item.get('category')}", item_id)

    messages = get_messages(item)
    if not messages:
        add_issue(issues, "error", dataset, "missing_messages", "messages is missing or not a list", item_id)
        return False

    roles = [msg.get("role") for msg in messages if isinstance(msg, dict)]
    if len(roles) < 3:
        ok = False
        add_issue(issues, "error", dataset, "too_few_messages", "messages has fewer than 3 roles", item_id)
    if roles[:3] != ["system", "user", "assistant"]:
        ok = False
        add_issue(issues, "error", dataset, "bad_role_order", f"role order is {roles[:3]}", item_id)

    for msg in messages:
        if not isinstance(msg, dict):
            ok = False
            add_issue(issues, "error", dataset, "message_not_object", "message is not an object", item_id)
            continue
        if msg.get("role") not in {"system", "user", "assistant"}:
            ok = False
            add_issue(issues, "error", dataset, "invalid_role", f"invalid role: {msg.get('role')}", item_id)
        if not isinstance(msg.get("content"), str) or not msg.get("content", "").strip():
            ok = False
            add_issue(issues, "error", dataset, "empty_content", "message content is empty or not string", item_id)

    system, user, assistant = extract_sft_texts(item)
    if len(user) < 4:
        ok = False
        add_issue(issues, "error", dataset, "user_too_short", "user text too short", item_id)
    if len(assistant) < 20:
        ok = False
        add_issue(issues, "error", dataset, "assistant_too_short", "assistant text too short", item_id)

    sensitive, sensitive_type = contains_sensitive_info(user + "\n" + assistant)
    if sensitive:
        ok = False
        add_issue(issues, "error", dataset, f"sensitive_{sensitive_type}", "sensitive info detected", item_id)

    risky, risky_type = contains_risky_assistant_commitment(assistant)
    if risky:
        ok = False
        add_issue(issues, "error", dataset, risky_type, "risky assistant commitment detected", item_id)

    if any(phrase in assistant for phrase in BAD_ANSWER_PHRASES):
        ok = False
        add_issue(issues, "error", dataset, "bad_answer_phrase", "assistant contains bad phrase", item_id)
    if not system:
        ok = False
        add_issue(issues, "error", dataset, "missing_system_text", "system text is empty", item_id)
    return ok


def check_sft_dataset(name: str, items: Sequence[Dict[str, Any]], issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_count = 0
    id_counter = Counter(str(item.get("id", "")) for item in items if item.get("id"))
    for item_id, count in id_counter.items():
        if count > 1:
            add_issue(issues, "error", name, "duplicate_id", f"duplicate id appears {count} times", item_id)

    user_answer_keys = Counter()
    for item in items:
        _, user, assistant = extract_sft_texts(item)
        if user and assistant:
            user_answer_keys[normalize_text(user) + "||" + normalize_text(assistant)] += 1
    for _, count in user_answer_keys.items():
        if count > 1:
            add_issue(issues, "warning", name, "duplicate_user_answer_pair", f"duplicate user-answer pair appears {count} times")

    for item in items:
        if validate_sft_item(item, name, issues):
            valid_count += 1

    return {
        "total": len(items),
        "valid_format": valid_count,
        "category_distribution": count_by(items, "category"),
        "source_distribution": count_by(items, "source"),
        "seed_id_count": len({str(item.get("seed_id")) for item in items if item.get("seed_id")}),
    }


def check_sft_seed_leakage(train_items: Sequence[Dict[str, Any]], valid_items: Sequence[Dict[str, Any]], issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    train_seed_ids = {str(item.get("seed_id")) for item in train_items if item.get("seed_id")}
    valid_seed_ids = {str(item.get("seed_id")) for item in valid_items if item.get("seed_id")}
    leakage = sorted(train_seed_ids & valid_seed_ids)
    if leakage:
        add_issue(issues, "error", "sft_train_valid", "seed_id_leakage", f"{len(leakage)} seed_id values appear in both train and valid", extra={"seed_ids": leakage[:50]})
    return {"train_seed_ids": len(train_seed_ids), "valid_seed_ids": len(valid_seed_ids), "leakage_count": len(leakage), "leakage_examples": leakage[:50]}


def kb_doc_id(item: Dict[str, Any]) -> str:
    return str(item.get("chunk_id") or item.get("product_id") or item.get("id") or "")


def validate_kb_item(item: Dict[str, Any], issues: List[Dict[str, Any]]) -> bool:
    ok = True
    item_id = kb_doc_id(item)
    if not item_id:
        ok = False
        add_issue(issues, "error", "kb_all", "missing_doc_id", "missing chunk_id/product_id/id")
    doc_type = item.get("doc_type")
    if doc_type not in VALID_KB_DOC_TYPES:
        ok = False
        add_issue(issues, "error", "kb_all", "invalid_doc_type", f"invalid doc_type: {doc_type}", item_id)
    category = item.get("category")
    if category not in VALID_CATEGORIES and doc_type != "product":
        ok = False
        add_issue(issues, "error", "kb_all", "invalid_category", f"invalid category: {category}", item_id)
    title = item.get("title")
    if not isinstance(title, str) or not title.strip():
        ok = False
        add_issue(issues, "error", "kb_all", "missing_title", "title missing or empty", item_id)
    content = item.get("content")
    if not isinstance(content, str) or len(content.strip()) < 10:
        ok = False
        add_issue(issues, "error", "kb_all", "content_too_short", "content missing or too short", item_id)
    sensitive, sensitive_type = contains_sensitive_info(str(title) + "\n" + str(content))
    if sensitive:
        ok = False
        add_issue(issues, "error", "kb_all", f"sensitive_{sensitive_type}", "sensitive info detected", item_id)
    return ok


def check_kb(items: Sequence[Dict[str, Any]], issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_count = 0
    doc_ids = [kb_doc_id(item) for item in items if kb_doc_id(item)]
    for doc_id, count in Counter(doc_ids).items():
        if count > 1:
            add_issue(issues, "error", "kb_all", "duplicate_doc_id", f"doc_id appears {count} times", doc_id)
    content_counter = Counter(normalize_text(item.get("content", "")) for item in items if item.get("content"))
    for _, count in content_counter.items():
        if count > 1:
            add_issue(issues, "warning", "kb_all", "duplicate_content", f"duplicate content appears {count} times")
    for item in items:
        if validate_kb_item(item, issues):
            valid_count += 1
    return {
        "total": len(items),
        "valid_format": valid_count,
        "doc_type_distribution": count_by(items, "doc_type"),
        "category_distribution": count_by(items, "category"),
        "unique_doc_ids": len(set(doc_ids)),
    }


def validate_eval_item(item: Dict[str, Any], kb_doc_ids: Set[str], issues: List[Dict[str, Any]]) -> bool:
    ok = True
    item_id = str(item.get("id", ""))
    required = ["id", "category", "eval_type", "difficulty", "query", "reference_answer", "must_include", "must_not_include", "need_human", "need_rag", "risk_tags"]
    for field in required:
        if field not in item:
            ok = False
            add_issue(issues, "error", "eval_set", f"missing_{field}", f"missing field: {field}", item_id)

    if item.get("category") not in VALID_CATEGORIES:
        ok = False
        add_issue(issues, "error", "eval_set", "invalid_category", f"invalid category: {item.get('category')}", item_id)
    if item.get("eval_type") not in VALID_EVAL_TYPES:
        ok = False
        add_issue(issues, "error", "eval_set", "invalid_eval_type", f"invalid eval_type: {item.get('eval_type')}", item_id)
    if item.get("difficulty") not in VALID_DIFFICULTIES:
        ok = False
        add_issue(issues, "error", "eval_set", "invalid_difficulty", f"invalid difficulty: {item.get('difficulty')}", item_id)

    query = item.get("query")
    if not isinstance(query, str) or len(query.strip()) < 4:
        ok = False
        add_issue(issues, "error", "eval_set", "bad_query", "query too short or not string", item_id)
    ref = item.get("reference_answer")
    if not isinstance(ref, str) or len(ref.strip()) < 10:
        ok = False
        add_issue(issues, "error", "eval_set", "bad_reference_answer", "reference_answer too short or not string", item_id)

    for field in ["must_include", "must_not_include", "risk_tags"]:
        value = item.get(field)
        if not isinstance(value, list) or not [x for x in value if str(x).strip()]:
            ok = False
            add_issue(issues, "error", "eval_set", f"empty_{field}", f"{field} empty or not list", item_id)

    if not isinstance(item.get("need_human"), bool):
        ok = False
        add_issue(issues, "error", "eval_set", "need_human_not_bool", "need_human is not bool", item_id)
    if not isinstance(item.get("need_rag"), bool):
        ok = False
        add_issue(issues, "error", "eval_set", "need_rag_not_bool", "need_rag is not bool", item_id)
    if item.get("eval_type") == "rag" and item.get("need_rag") is not True:
        ok = False
        add_issue(issues, "error", "eval_set", "rag_eval_type_need_rag_false", "eval_type=rag requires need_rag=true", item_id)
    if item.get("eval_type") == "human_transfer" and item.get("need_human") is not True:
        ok = False
        add_issue(issues, "error", "eval_set", "human_transfer_need_human_false", "eval_type=human_transfer requires need_human=true", item_id)

    expected_docs = item.get("expected_docs", [])
    if item.get("need_rag") is True:
        if not isinstance(expected_docs, list) or not expected_docs:
            ok = False
            add_issue(issues, "error", "eval_set", "need_rag_missing_expected_docs", "need_rag=true but expected_docs empty", item_id)
        else:
            missing_docs = [str(doc_id) for doc_id in expected_docs if str(doc_id) not in kb_doc_ids]
            if missing_docs:
                ok = False
                add_issue(issues, "error", "eval_set", "expected_docs_not_in_kb", f"expected_docs not found in kb_all: {missing_docs[:5]}", item_id, extra={"missing_docs": missing_docs[:20]})

    sensitive, sensitive_type = contains_sensitive_info(str(query) + "\n" + str(ref))
    if sensitive:
        ok = False
        add_issue(issues, "error", "eval_set", f"sensitive_{sensitive_type}", "sensitive info detected", item_id)
    return ok


def check_eval_set(items: Sequence[Dict[str, Any]], kb_doc_ids: Set[str], issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_count = 0
    id_counter = Counter(str(item.get("id", "")) for item in items if item.get("id"))
    for item_id, count in id_counter.items():
        if count > 1:
            add_issue(issues, "error", "eval_set", "duplicate_id", f"duplicate id appears {count} times", item_id)
    query_counter = Counter(normalize_text(item.get("query", "")) for item in items if item.get("query"))
    for _, count in query_counter.items():
        if count > 1:
            add_issue(issues, "error", "eval_set", "duplicate_query", f"duplicate query appears {count} times")
    for item in items:
        if validate_eval_item(item, kb_doc_ids, issues):
            valid_count += 1
    return {
        "total": len(items),
        "valid_format": valid_count,
        "category_distribution": count_by(items, "category"),
        "eval_type_distribution": count_by(items, "eval_type"),
        "difficulty_distribution": count_by(items, "difficulty"),
        "need_human_distribution": count_by(items, "need_human"),
        "need_rag_distribution": count_by(items, "need_rag"),
        "risk_tag_distribution": dict(Counter(tag for item in items for tag in item.get("risk_tags", []) if isinstance(item.get("risk_tags"), list))),
    }


def chosen_has_risky_commitment(chosen_text: str) -> bool:
    return contains_any_pattern(chosen_text, RISKY_ASSISTANT_PATTERNS)


def validate_dpo_item(item: Dict[str, Any], dataset: str, issues: List[Dict[str, Any]], similarity_threshold: float) -> bool:
    ok = True
    item_id = str(item.get("id", ""))
    required = ["id", "category", "source", "source_id", "rejected_type", "severity", "prompt", "chosen", "rejected", "preference_reason"]
    for field in required:
        if field not in item:
            ok = False
            add_issue(issues, "error", dataset, f"missing_{field}", f"missing field: {field}", item_id)
    if item.get("category") not in VALID_CATEGORIES:
        ok = False
        add_issue(issues, "error", dataset, "invalid_category", f"invalid category: {item.get('category')}", item_id)
    if item.get("rejected_type") not in VALID_REJECTED_TYPES:
        ok = False
        add_issue(issues, "error", dataset, "invalid_rejected_type", f"invalid rejected_type: {item.get('rejected_type')}", item_id)
    if item.get("severity") not in VALID_SEVERITIES:
        ok = False
        add_issue(issues, "error", dataset, "invalid_severity", f"invalid severity: {item.get('severity')}", item_id)

    prompt = item.get("prompt", [])
    chosen = item.get("chosen", [])
    rejected = item.get("rejected", [])

    if not isinstance(prompt, list) or len(prompt) < 2:
        ok = False
        add_issue(issues, "error", dataset, "bad_prompt", "prompt should contain system and user", item_id)
    else:
        prompt_roles = [msg.get("role") for msg in prompt if isinstance(msg, dict)]
        if "system" not in prompt_roles or "user" not in prompt_roles:
            ok = False
            add_issue(issues, "error", dataset, "prompt_missing_system_or_user", "prompt missing system or user", item_id)

    if not isinstance(chosen, list) or len(chosen) != 1 or chosen[0].get("role") != "assistant":
        ok = False
        add_issue(issues, "error", dataset, "bad_chosen", "chosen should contain one assistant message", item_id)
    if not isinstance(rejected, list) or len(rejected) != 1 or rejected[0].get("role") != "assistant":
        ok = False
        add_issue(issues, "error", dataset, "bad_rejected", "rejected should contain one assistant message", item_id)

    chosen_text = extract_assistant_text(chosen) if isinstance(chosen, list) else ""
    rejected_text = extract_assistant_text(rejected) if isinstance(rejected, list) else ""

    if len(chosen_text) < 10:
        ok = False
        add_issue(issues, "error", dataset, "chosen_too_short", "chosen too short", item_id)
    if len(rejected_text) < 4:
        ok = False
        add_issue(issues, "error", dataset, "rejected_too_short", "rejected too short", item_id)
    if len(rejected_text) > 200:
        ok = False
        add_issue(issues, "error", dataset, "rejected_too_long", "rejected too long", item_id)
    if chosen_text and rejected_text and similarity(chosen_text, rejected_text) >= similarity_threshold:
        ok = False
        add_issue(issues, "error", dataset, "chosen_rejected_too_similar", f"chosen/rejected similarity >= {similarity_threshold}", item_id)
    if chosen_has_risky_commitment(chosen_text):
        ok = False
        add_issue(issues, "error", dataset, "chosen_has_risky_commitment", "chosen contains risky commitment", item_id)

    sensitive, sensitive_type = contains_sensitive_info(json.dumps({k: v for k, v in item.items() if not k.startswith("_")}, ensure_ascii=False))
    if sensitive:
        ok = False
        add_issue(issues, "error", dataset, f"sensitive_{sensitive_type}", "sensitive info detected", item_id)

    reason = item.get("preference_reason")
    if not isinstance(reason, str) or len(reason.strip()) < 8:
        ok = False
        add_issue(issues, "error", dataset, "bad_preference_reason", "preference_reason too short or not string", item_id)
    return ok


def check_dpo_dataset(name: str, items: Sequence[Dict[str, Any]], issues: List[Dict[str, Any]], similarity_threshold: float) -> Dict[str, Any]:
    valid_count = 0
    id_counter = Counter(str(item.get("id", "")) for item in items if item.get("id"))
    for item_id, count in id_counter.items():
        if count > 1:
            add_issue(issues, "error", name, "duplicate_id", f"duplicate id appears {count} times", item_id)
    dpo_key_counter = Counter()
    for item in items:
        prompt_user = extract_prompt_user(item.get("prompt", [])) if isinstance(item.get("prompt"), list) else ""
        rejected_text = extract_assistant_text(item.get("rejected", [])) if isinstance(item.get("rejected"), list) else ""
        key = normalize_text(prompt_user) + "||" + str(item.get("rejected_type")) + "||" + normalize_text(rejected_text)
        if key.strip("|"):
            dpo_key_counter[key] += 1
    for _, count in dpo_key_counter.items():
        if count > 1:
            add_issue(issues, "warning", name, "duplicate_dpo_key", f"duplicate DPO key appears {count} times")
    for item in items:
        if validate_dpo_item(item, name, issues, similarity_threshold):
            valid_count += 1
    return {
        "total": len(items),
        "valid_format": valid_count,
        "category_distribution": count_by(items, "category"),
        "source_distribution": count_by(items, "source"),
        "rejected_type_distribution": count_by(items, "rejected_type"),
        "severity_distribution": count_by(items, "severity"),
    }


def check_eval_vs_sft_overlap(eval_items, sft_train, sft_valid, issues, threshold: float) -> Dict[str, Any]:
    sft_user_texts = []
    for dataset_name, items in [("sft_train", sft_train), ("sft_valid", sft_valid)]:
        for item in items:
            _, user, _ = extract_sft_texts(item)
            if user:
                sft_user_texts.append((dataset_name, user))
    exact_overlap = 0
    near_overlap = 0
    examples = []
    sft_norms = {normalize_text(text): dataset_name for dataset_name, text in sft_user_texts}
    for eval_item in eval_items:
        query = str(eval_item.get("query", ""))
        if not query:
            continue
        norm = normalize_text(query)
        if norm in sft_norms:
            exact_overlap += 1
            if len(examples) < 20:
                examples.append({"eval_id": eval_item.get("id"), "query": query, "overlap_type": "exact", "sft_dataset": sft_norms[norm]})
            continue
        for dataset_name, sft_user in sft_user_texts:
            if similarity(query, sft_user) >= threshold:
                near_overlap += 1
                if len(examples) < 20:
                    examples.append({"eval_id": eval_item.get("id"), "query": query, "overlap_type": "near", "sft_dataset": dataset_name, "sft_user": sft_user})
                break
    if exact_overlap > 0:
        add_issue(issues, "warning", "cross_dataset", "eval_sft_exact_overlap", f"{exact_overlap} eval queries exactly overlap with SFT user texts", extra={"examples": examples[:10]})
    if near_overlap > 0:
        add_issue(issues, "warning", "cross_dataset", "eval_sft_near_overlap", f"{near_overlap} eval queries are near-duplicates of SFT user texts", extra={"examples": examples[:10]})
    return {"exact_overlap": exact_overlap, "near_overlap": near_overlap, "examples": examples}


def check_thresholds(stats: Dict[str, Any], thresholds: Dict[str, int], issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = {}
    checks = [
        ("sft_train_min", stats["sft_train"]["total"], thresholds["sft_train_min"], "sft_train"),
        ("sft_valid_min", stats["sft_valid"]["total"], thresholds["sft_valid_min"], "sft_valid"),
        ("eval_set_min", stats["eval_set"]["total"], thresholds["eval_set_min"], "eval_set"),
        ("dpo_total_min", stats["dpo_train"]["total"] + stats["dpo_valid"]["total"], thresholds["dpo_total_min"], "dpo"),
        ("dpo_valid_min", stats["dpo_valid"]["total"], thresholds["dpo_valid_min"], "dpo_valid"),
        ("kb_all_min", stats["kb_all"]["total"], thresholds["kb_all_min"], "kb_all"),
    ]
    for check_name, actual, expected, dataset in checks:
        passed = actual >= expected
        results[check_name] = {"actual": actual, "expected_min": expected, "passed": passed}
        if not passed:
            add_issue(issues, "error", dataset, "below_minimum_count", f"{check_name}: actual={actual}, expected_min={expected}")
    return results


def issue_summary(issues: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "total": len(issues),
        "by_severity": dict(Counter(issue["severity"] for issue in issues)),
        "by_dataset": dict(Counter(issue["dataset"] for issue in issues)),
        "by_type": dict(Counter(issue["issue_type"] for issue in issues)),
    }


def write_distribution_table(f, title: str, distribution: Dict[str, int]) -> None:
    f.write(f"### {title}\n\n")
    if not distribution:
        f.write("- 无\n\n")
        return
    f.write("| 项 | 数量 |\n|---|---:|\n")
    for key, value in sorted(distribution.items()):
        f.write(f"| {key} | {value} |\n")
    f.write("\n")


def write_dataset_section(f, title: str, stats: Dict[str, Any]) -> None:
    f.write(f"## {title}\n\n")
    f.write(f"- 样本数：{stats.get('total', 0)}\n")
    f.write(f"- 格式有效数：{stats.get('valid_format', 0)}\n\n")
    for key, value in stats.items():
        if key.endswith("_distribution") and isinstance(value, dict):
            write_distribution_table(f, key, value)


def write_markdown_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        status = "PASS" if report["passed"] else "FAIL"
        f.write("# 数据集质量验收报告\n\n")
        f.write(f"## 1. 总体结论：{status}\n\n")
        f.write(f"- 错误数：{report['issue_summary']['by_severity'].get('error', 0)}\n")
        f.write(f"- 警告数：{report['issue_summary']['by_severity'].get('warning', 0)}\n")
        f.write(f"- Issue 总数：{report['issue_summary']['total']}\n\n")

        f.write("## 2. 数量门槛检查\n\n")
        f.write("| 检查项 | 实际值 | 最低要求 | 是否通过 |\n|---|---:|---:|---|\n")
        for name, result in report["threshold_results"].items():
            f.write(f"| {name} | {result['actual']} | {result['expected_min']} | {'PASS' if result['passed'] else 'FAIL'} |\n")
        f.write("\n")

        write_dataset_section(f, "3. SFT Train", report["datasets"]["sft_train"])
        write_dataset_section(f, "4. SFT Valid", report["datasets"]["sft_valid"])

        f.write("## 5. SFT Seed 泄漏检查\n\n")
        sft_leakage = report["cross_checks"]["sft_seed_leakage"]
        f.write(f"- train seed_id 数：{sft_leakage['train_seed_ids']}\n")
        f.write(f"- valid seed_id 数：{sft_leakage['valid_seed_ids']}\n")
        f.write(f"- 泄漏数量：{sft_leakage['leakage_count']}\n")
        if sft_leakage["leakage_examples"]:
            f.write("- 泄漏示例：\n")
            for seed_id in sft_leakage["leakage_examples"][:20]:
                f.write(f"  - {seed_id}\n")
        f.write("\n")

        write_dataset_section(f, "6. Eval Set", report["datasets"]["eval_set"])
        write_dataset_section(f, "7. DPO Candidates", report["datasets"]["dpo_candidates"])
        write_dataset_section(f, "8. DPO Train", report["datasets"]["dpo_train"])
        write_dataset_section(f, "9. DPO Valid", report["datasets"]["dpo_valid"])
        write_dataset_section(f, "10. RAG KB", report["datasets"]["kb_all"])

        f.write("## 11. Eval 与 SFT 重复检查\n\n")
        overlap = report["cross_checks"]["eval_vs_sft_overlap"]
        f.write(f"- 精确重复数：{overlap['exact_overlap']}\n")
        f.write(f"- 近重复数：{overlap['near_overlap']}\n")
        if overlap["examples"]:
            f.write("- 示例：\n")
            for ex in overlap["examples"][:10]:
                f.write(f"  - {ex}\n")
        f.write("\n")

        f.write("## 12. Issue 汇总\n\n")
        write_distribution_table(f, "按严重程度", report["issue_summary"]["by_severity"])
        write_distribution_table(f, "按数据集", report["issue_summary"]["by_dataset"])
        write_distribution_table(f, "按问题类型", report["issue_summary"]["by_type"])

        f.write("## 13. Issue 明细\n\n")
        if not report["issues"]:
            f.write("- 无\n")
        else:
            f.write("| 严重程度 | 数据集 | 类型 | id | 信息 |\n|---|---|---|---|---|\n")
            for issue in report["issues"][:300]:
                message = str(issue.get("message", "")).replace("|", "/")
                f.write(f"| {issue.get('severity')} | {issue.get('dataset')} | {issue.get('issue_type')} | {issue.get('id', '')} | {message} |\n")
            if len(report["issues"]) > 300:
                f.write("\n仅展示前 300 条 issue，完整内容见 JSON 报告。\n")


def parse_args() -> argparse.Namespace:
    root = get_project_root()
    parser = argparse.ArgumentParser(description="Full dataset quality gate for SFT / Eval / DPO / RAG KB.")

    parser.add_argument("--cleaned-sft", type=Path, default=root / "data" / "interim" / "cleaned_sft.jsonl")
    parser.add_argument("--sft-train", type=Path, default=root / "data" / "processed" / "sft_train.jsonl")
    parser.add_argument("--sft-valid", type=Path, default=root / "data" / "processed" / "sft_valid.jsonl")
    parser.add_argument("--eval-set", type=Path, default=root / "data" / "eval" / "eval_set.jsonl")
    parser.add_argument("--dpo-candidates", type=Path, default=root / "data" / "interim" / "dpo_candidates.jsonl")
    parser.add_argument("--dpo-train", type=Path, default=root / "data" / "processed" / "dpo_train.jsonl")
    parser.add_argument("--dpo-valid", type=Path, default=root / "data" / "processed" / "dpo_valid.jsonl")
    parser.add_argument("--kb-all", type=Path, default=root / "data" / "knowledge_base" / "kb_all.jsonl")
    parser.add_argument("--report-md", type=Path, default=root / "docs" / "dataset_quality_report.md")
    parser.add_argument("--report-json", type=Path, default=root / "docs" / "dataset_quality_report.json")

    parser.add_argument("--sft-train-min", type=int, default=1800)
    parser.add_argument("--sft-valid-min", type=int, default=200)
    parser.add_argument("--eval-set-min", type=int, default=200)
    parser.add_argument("--dpo-total-min", type=int, default=300)
    parser.add_argument("--dpo-valid-min", type=int, default=20)
    parser.add_argument("--kb-all-min", type=int, default=80)
    parser.add_argument("--dpo-similarity-threshold", type=float, default=0.82)
    parser.add_argument("--eval-sft-overlap-threshold", type=float, default=0.94)
    parser.add_argument("--warning-as-fail", action="store_true")
    parser.add_argument("--no-fail-exit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    issues: List[Dict[str, Any]] = []

    paths = {
        "cleaned_sft": args.cleaned_sft,
        "sft_train": args.sft_train,
        "sft_valid": args.sft_valid,
        "eval_set": args.eval_set,
        "dpo_candidates": args.dpo_candidates,
        "dpo_train": args.dpo_train,
        "dpo_valid": args.dpo_valid,
        "kb_all": args.kb_all,
    }

    loaded: Dict[str, List[Dict[str, Any]]] = {}
    for name, path in paths.items():
        required = name != "cleaned_sft"
        items, errors = read_jsonl(path, required=required)
        loaded[name] = items
        add_file_errors_as_issues(issues, name, errors)

    kb_stats = check_kb(loaded["kb_all"], issues)
    kb_doc_ids = {kb_doc_id(item) for item in loaded["kb_all"] if kb_doc_id(item)}

    datasets = {
        "cleaned_sft": check_sft_dataset("cleaned_sft", loaded["cleaned_sft"], issues),
        "sft_train": check_sft_dataset("sft_train", loaded["sft_train"], issues),
        "sft_valid": check_sft_dataset("sft_valid", loaded["sft_valid"], issues),
        "eval_set": check_eval_set(loaded["eval_set"], kb_doc_ids, issues),
        "dpo_candidates": check_dpo_dataset("dpo_candidates", loaded["dpo_candidates"], issues, args.dpo_similarity_threshold),
        "dpo_train": check_dpo_dataset("dpo_train", loaded["dpo_train"], issues, args.dpo_similarity_threshold),
        "dpo_valid": check_dpo_dataset("dpo_valid", loaded["dpo_valid"], issues, args.dpo_similarity_threshold),
        "kb_all": kb_stats,
    }

    cross_checks = {
        "sft_seed_leakage": check_sft_seed_leakage(loaded["sft_train"], loaded["sft_valid"], issues),
        "eval_vs_sft_overlap": check_eval_vs_sft_overlap(
            loaded["eval_set"], loaded["sft_train"], loaded["sft_valid"], issues,
            threshold=args.eval_sft_overlap_threshold,
        ),
    }

    thresholds = {
        "sft_train_min": args.sft_train_min,
        "sft_valid_min": args.sft_valid_min,
        "eval_set_min": args.eval_set_min,
        "dpo_total_min": args.dpo_total_min,
        "dpo_valid_min": args.dpo_valid_min,
        "kb_all_min": args.kb_all_min,
    }
    threshold_results = check_thresholds(datasets, thresholds, issues)

    error_count = sum(1 for issue in issues if issue["severity"] == "error")
    warning_count = sum(1 for issue in issues if issue["severity"] == "warning")
    passed = error_count == 0 and (warning_count == 0 if args.warning_as_fail else True)

    report = {
        "passed": passed,
        "paths": {name: str(path) for name, path in paths.items()},
        "thresholds": thresholds,
        "threshold_results": threshold_results,
        "datasets": datasets,
        "cross_checks": cross_checks,
        "issue_summary": issue_summary(issues),
        "issues": issues,
    }

    write_json(args.report_json, report)
    write_markdown_report(args.report_md, report)

    print("[SUMMARY]")
    print(f"  status: {'PASS' if passed else 'FAIL'}")
    print(f"  errors: {error_count}")
    print(f"  warnings: {warning_count}")
    print(f"  report md: {args.report_md}")
    print(f"  report json: {args.report_json}")
    for name, stats in datasets.items():
        print(f"  {name}: total={stats.get('total', 0)}, valid_format={stats.get('valid_format', 0)}")
    if not passed and not args.no_fail_exit:
        sys.exit(1)


if __name__ == "__main__":
    main()
