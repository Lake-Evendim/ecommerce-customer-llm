#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SFT 数据二次清洗与验收脚本。

输入：
    data/interim/sft_candidates.jsonl

输出：
    data/interim/cleaned_sft.jsonl
    docs/sft_clean_report.md
    docs/sft_clean_report.json

定位：
    expand_sft_with_llm.py 负责基于 sft_seed.jsonl 生成扩写样本；
    clean_sft_data.py 只清洗 sft_candidates.jsonl，不把黄金样本并入训练集。
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


VALID_CATEGORIES = {
    "return_refund",
    "logistics",
    "product_info",
    "quality_issue",
    "invoice",
    "coupon_price",
    "complaint",
    "manual_transfer",
}

EXPECTED_ROLES = ["system", "user", "assistant"]

PHONE_PATTERNS = [r"(?<!\d)1[3-9]\d{9}(?!\d)"]
ID_CARD_PATTERNS = [r"\b\d{17}[\dXx]\b", r"\b\d{15}\b"]
EMAIL_PATTERNS = [r"[\w\.-]+@[\w\.-]+\.\w+"]
ORDER_ID_PATTERNS = [
    r"订单号\s*[:：]?\s*[A-Za-z0-9_-]{5,}",
    r"订单\s*[:：]?\s*[A-Za-z0-9_-]{8,}",
    r"单号\s*[:：]?\s*[A-Za-z0-9_-]{8,}",
]

RISKY_MONEY_COMMITMENT_PATTERNS = [
    r"(赔|赔偿|赔付|补偿|补|返|退)\s*\d+(?:\.\d+)?\s*(元|块钱|人民币|rmb|RMB)",
    r"\d+(?:\.\d+)?\s*(元|块钱|人民币|rmb|RMB).{0,8}(赔|赔偿|赔付|补偿|补|返|退)",
]

RISKY_POLICY_PATTERNS = [
    r"(一定|肯定|保证|百分百|必须).{0,10}(退款|退货|送达|赔偿|赔付|解决|处理)",
    r"(明天|今天|后天).{0,8}(一定|肯定|保证).{0,8}(到|送达|退款|处理)",
    r"(三倍|十倍|假一赔十|全额).{0,8}(赔偿|赔付|退款)",
    r"(无条件退款|无条件退货|包退|包换|包赔)",
]

BAD_ANSWER_PHRASES = ["不知道", "不清楚", "随便", "自己看", "无法回答", "我也不知道", "不归我管"]
ANSWER_LIKE_MARKERS = ["您好", "建议您", "您可以", "很抱歉", "请您", "我们会", "平台会"]


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_jsonl(path: Path, required: bool = False) -> Tuple[List[Dict[str, Any]], Counter]:
    items: List[Dict[str, Any]] = []
    errors: Counter = Counter()

    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required input file not found: {path}")
        errors["missing_file"] += 1
        return items, errors

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                errors["json_decode_error"] += 1
                print(f"[WARN] JSON decode error: {path}, line {line_no}")
                continue

            if not isinstance(item, dict):
                errors["not_json_object"] += 1
                print(f"[WARN] JSON item is not object: {path}, line {line_no}")
                continue

            item["_input_file"] = str(path)
            item["_input_line"] = line_no
            items.append(item)

    return items, errors


def write_jsonl(path: Path, items: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            cleaned_item = {k: v for k, v in item.items() if not k.startswith("_")}
            f.write(json.dumps(cleaned_item, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_messages(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = item.get("messages", [])
    if not isinstance(messages, list):
        return []
    return messages


def extract_text_by_role(item: Dict[str, Any], role: str) -> str:
    parts: List[str] = []
    for msg in get_messages(item):
        if isinstance(msg, dict) and msg.get("role") == role:
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
    return "\n".join(parts).strip()


def extract_user_and_answer(item: Dict[str, Any]) -> Tuple[str, str]:
    return extract_text_by_role(item, "user"), extract_text_by_role(item, "assistant")


def extract_system_text(item: Dict[str, Any]) -> str:
    return extract_text_by_role(item, "system")


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "", text)
    text = text.replace("？", "?").replace("！", "!").replace("。", ".")
    text = text.replace("，", ",").replace("、", ",")
    text = text.replace("：", ":").replace("；", ";")
    return text


def normalize_for_dedup(user_text: str, answer_text: str) -> str:
    return normalize_text(user_text) + "||" + normalize_text(answer_text)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def contains_any_pattern(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def contains_sensitive_info(text: str) -> Tuple[bool, str]:
    checks = [
        ("phone", PHONE_PATTERNS),
        ("id_card", ID_CARD_PATTERNS),
        ("email", EMAIL_PATTERNS),
        ("order_id", ORDER_ID_PATTERNS),
    ]
    for reason, patterns in checks:
        if contains_any_pattern(text, patterns):
            return True, reason
    return False, "ok"


def contains_risky_commitment(text: str) -> Tuple[bool, str]:
    if contains_any_pattern(text, RISKY_MONEY_COMMITMENT_PATTERNS):
        return True, "risky_money_commitment"
    if contains_any_pattern(text, RISKY_POLICY_PATTERNS):
        return True, "risky_policy_commitment"
    return False, "ok"


def user_looks_like_assistant_answer(user_text: str) -> bool:
    question_markers = [
        "吗", "么", "嘛", "？", "?", "怎么", "咋", "如何", "为啥", "为什么",
        "能不能", "可不可以", "可以吗", "行不行", "是不是", "有没有",
        "多久", "多少", "哪里", "在哪", "怎么办", "啥时候", "会不会",
    ]
    if any(marker in user_text for marker in question_markers):
        return False

    strong_markers = [
        "感谢您的理解",
        "祝您生活愉快",
        "我们会尽快",
        "请您耐心等待",
        "很抱歉给您带来不便",
        "您可以在订单详情页",
        "建议您先查看订单页",
    ]
    if any(marker in user_text for marker in strong_markers):
        return True

    marker_count = sum(1 for marker in ANSWER_LIKE_MARKERS if marker in user_text)
    return marker_count >= 2


def validate_messages(item: Dict[str, Any]) -> Tuple[bool, str]:
    messages = get_messages(item)

    if not messages:
        return False, "missing_or_invalid_messages"

    roles = [msg.get("role") for msg in messages if isinstance(msg, dict)]

    if len(roles) < 3:
        return False, "too_few_messages"
    if "system" not in roles:
        return False, "missing_system"
    if "user" not in roles:
        return False, "missing_user"
    if "assistant" not in roles:
        return False, "missing_assistant"

    if roles[:3] != EXPECTED_ROLES:
        return False, "bad_role_order"

    for msg in messages:
        if not isinstance(msg, dict):
            return False, "message_not_object"

        role = msg.get("role")
        content = msg.get("content")

        if role not in {"system", "user", "assistant"}:
            return False, "invalid_role"
        if not isinstance(content, str):
            return False, "content_not_string"
        if not content.strip():
            return False, "empty_content"

    return True, "ok"


def validate_item(
    item: Dict[str, Any],
    min_user_len: int,
    min_answer_len: int,
    max_user_len: int,
    max_answer_len: int,
    strict_category: bool,
    check_user_answer_style: bool,
) -> Tuple[bool, str]:
    ok, reason = validate_messages(item)
    if not ok:
        return False, reason

    category = item.get("category")
    if strict_category and category not in VALID_CATEGORIES:
        return False, "invalid_category"

    user_text, answer_text = extract_user_and_answer(item)
    system_text = extract_system_text(item)

    if len(user_text) < min_user_len:
        return False, "user_too_short"
    if len(user_text) > max_user_len:
        return False, "user_too_long"
    if len(answer_text) < min_answer_len:
        return False, "answer_too_short"
    if len(answer_text) > max_answer_len:
        return False, "answer_too_long"

    sensitive, sensitive_reason = contains_sensitive_info(user_text + "\n" + answer_text)
    if sensitive:
        return False, f"sensitive_{sensitive_reason}"

    risky, risky_reason = contains_risky_commitment(answer_text)
    if risky:
        return False, risky_reason

    if any(p in answer_text for p in BAD_ANSWER_PHRASES):
        return False, "bad_answer_phrase"

    if check_user_answer_style and user_looks_like_assistant_answer(user_text):
        return False, "user_looks_like_answer"

    if not system_text:
        return False, "empty_system"

    return True, "ok"


def is_near_duplicate_against_seen(
    user_text: str,
    answer_text: str,
    seen_user_texts_by_answer: Dict[str, List[str]],
    threshold: float,
) -> bool:
    answer_key = normalize_text(answer_text)
    for old_user in seen_user_texts_by_answer.get(answer_key, []):
        if similarity(user_text, old_user) >= threshold:
            return True
    return False


def build_drop_example(
    item: Dict[str, Any],
    reason: str,
    user_text: str,
    answer_text: str,
) -> Dict[str, Any]:
    return {
        "id": item.get("id"),
        "seed_id": item.get("seed_id"),
        "category": item.get("category"),
        "source": item.get("source"),
        "reason": reason,
        "user": user_text[:200],
        "assistant": answer_text[:200],
        "input_file": item.get("_input_file"),
        "input_line": item.get("_input_line"),
    }


def clean_items(
    raw_items: Sequence[Dict[str, Any]],
    min_user_len: int,
    min_answer_len: int,
    max_user_len: int,
    max_answer_len: int,
    near_duplicate_threshold: float,
    enable_near_duplicate_filter: bool,
    strict_category: bool,
    check_user_answer_style: bool,
) -> Tuple[List[Dict[str, Any]], Counter, List[Dict[str, Any]]]:
    items = list(raw_items)

    cleaned: List[Dict[str, Any]] = []
    drop_reasons: Counter = Counter()
    dropped_examples: List[Dict[str, Any]] = []

    seen_pair_keys = set()
    seen_user_texts_by_answer: Dict[str, List[str]] = defaultdict(list)

    for item in items:
        ok, reason = validate_item(
            item=item,
            min_user_len=min_user_len,
            min_answer_len=min_answer_len,
            max_user_len=max_user_len,
            max_answer_len=max_answer_len,
            strict_category=strict_category,
            check_user_answer_style=check_user_answer_style,
        )
        user_text, answer_text = extract_user_and_answer(item)

        if not ok:
            drop_reasons[reason] += 1
            if len(dropped_examples) < 200:
                dropped_examples.append(build_drop_example(item, reason, user_text, answer_text))
            continue

        pair_key = normalize_for_dedup(user_text, answer_text)
        if pair_key in seen_pair_keys:
            reason = "duplicate_user_answer_pair"
            drop_reasons[reason] += 1
            if len(dropped_examples) < 200:
                dropped_examples.append(build_drop_example(item, reason, user_text, answer_text))
            continue

        if enable_near_duplicate_filter and is_near_duplicate_against_seen(
            user_text=user_text,
            answer_text=answer_text,
            seen_user_texts_by_answer=seen_user_texts_by_answer,
            threshold=near_duplicate_threshold,
        ):
            reason = "near_duplicate_same_answer"
            drop_reasons[reason] += 1
            if len(dropped_examples) < 200:
                dropped_examples.append(build_drop_example(item, reason, user_text, answer_text))
            continue

        seen_pair_keys.add(pair_key)
        seen_user_texts_by_answer[normalize_text(answer_text)].append(user_text)
        cleaned.append(item)

    return cleaned, drop_reasons, dropped_examples


def count_by(items: Sequence[Dict[str, Any]], key: str) -> Counter:
    return Counter(str(item.get(key, "unknown")) for item in items)


def count_seed_retention(items: Sequence[Dict[str, Any]]) -> Counter:
    counter: Counter = Counter()
    for item in items:
        seed_id = item.get("seed_id")
        if seed_id:
            counter[str(seed_id)] += 1
    return counter


def length_stats(values: Sequence[int]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "avg": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": round(sum(values) / len(values), 2),
    }


def build_report(
    raw_items: Sequence[Dict[str, Any]],
    cleaned_items: Sequence[Dict[str, Any]],
    input_errors: Counter,
    drop_reasons: Counter,
    dropped_examples: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    user_lengths: List[int] = []
    answer_lengths: List[int] = []

    for item in cleaned_items:
        user_text, answer_text = extract_user_and_answer(item)
        user_lengths.append(len(user_text))
        answer_lengths.append(len(answer_text))

    return {
        "summary": {
            "raw_items": len(raw_items),
            "cleaned_items": len(cleaned_items),
            "dropped_items": len(raw_items) - len(cleaned_items),
            "drop_ratio": round((len(raw_items) - len(cleaned_items)) / max(len(raw_items), 1), 4),
        },
        "input_errors": dict(input_errors),
        "drop_reasons": dict(drop_reasons),
        "category_distribution": dict(count_by(cleaned_items, "category")),
        "source_distribution": dict(count_by(cleaned_items, "source")),
        "seed_retention": dict(count_seed_retention(cleaned_items)),
        "length_stats": {
            "user": length_stats(user_lengths),
            "assistant": length_stats(answer_lengths),
        },
        "dropped_examples": list(dropped_examples),
    }


def write_markdown_report(path: Path, report: Dict[str, Any], output_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    summary = report["summary"]
    input_errors = report["input_errors"]
    drop_reasons = report["drop_reasons"]
    category_distribution = report["category_distribution"]
    source_distribution = report["source_distribution"]
    seed_retention = report["seed_retention"]
    length = report["length_stats"]

    with path.open("w", encoding="utf-8") as f:
        f.write("# SFT 数据清洗报告\n\n")

        f.write("## 1. 总览\n\n")
        f.write(f"- 原始样本数：{summary['raw_items']}\n")
        f.write(f"- 清洗后样本数：{summary['cleaned_items']}\n")
        f.write(f"- 删除样本数：{summary['dropped_items']}\n")
        f.write(f"- 删除比例：{summary['drop_ratio']}\n")
        f.write(f"- 输出文件：`{output_path}`\n\n")

        f.write("## 2. 输入解析错误\n\n")
        if input_errors:
            for k, v in sorted(input_errors.items()):
                f.write(f"- {k}: {v}\n")
        else:
            f.write("- 无\n")
        f.write("\n")

        f.write("## 3. 删除原因统计\n\n")
        if drop_reasons:
            for k, v in sorted(drop_reasons.items(), key=lambda x: (-x[1], x[0])):
                f.write(f"- {k}: {v}\n")
        else:
            f.write("- 无\n")
        f.write("\n")

        f.write("## 4. 类别分布\n\n")
        f.write("| 类别 | 数量 |\n|---|---:|\n")
        for k, v in sorted(category_distribution.items()):
            f.write(f"| {k} | {v} |\n")
        f.write("\n")

        f.write("## 5. 来源分布\n\n")
        f.write("| 来源 | 数量 |\n|---|---:|\n")
        for k, v in sorted(source_distribution.items()):
            f.write(f"| {k} | {v} |\n")
        f.write("\n")

        f.write("## 6. 文本长度统计\n\n")
        f.write("| 字段 | 样本数 | 最短 | 最长 | 平均 |\n|---|---:|---:|---:|---:|\n")
        f.write(
            f"| user | {length['user']['count']} | {length['user']['min']} | "
            f"{length['user']['max']} | {length['user']['avg']} |\n"
        )
        f.write(
            f"| assistant | {length['assistant']['count']} | {length['assistant']['min']} | "
            f"{length['assistant']['max']} | {length['assistant']['avg']} |\n\n"
        )

        f.write("## 7. Seed 扩写保留数量\n\n")
        if seed_retention:
            f.write("| seed_id | 保留扩写数 |\n|---|---:|\n")
            for k, v in sorted(seed_retention.items()):
                f.write(f"| {k} | {v} |\n")
        else:
            f.write("- 没有检测到带 seed_id 的扩写样本。\n")
        f.write("\n")

        f.write("## 8. 清洗规则说明\n\n")
        f.write("本脚本执行以下确定性清洗：\n\n")
        f.write("1. 检查 `messages` 是否存在且包含 `system/user/assistant`；\n")
        f.write("2. 检查角色顺序是否为 `system -> user -> assistant`；\n")
        f.write("3. 检查 `category` 是否属于项目定义的 8 个类别；\n")
        f.write("4. 删除 user 或 assistant 过短、过长的样本；\n")
        f.write("5. 删除包含手机号、身份证号、邮箱、明显订单号的样本；\n")
        f.write("6. 删除 assistant 中明显承诺退款、赔偿、送达或无条件处理的高风险样本；\n")
        f.write("7. 删除完全重复的 user-answer 样本；\n")
        f.write("8. 可选删除同一 answer 下的近重复 user 问法。\n\n")

        f.write("## 9. 删除样例\n\n")
        examples = report.get("dropped_examples", [])
        if not examples:
            f.write("- 无\n")
        else:
            for i, ex in enumerate(examples[:30], 1):
                f.write(f"### {i}. {ex.get('reason')}\n\n")
                f.write(f"- id: `{ex.get('id')}`\n")
                f.write(f"- seed_id: `{ex.get('seed_id')}`\n")
                f.write(f"- category: `{ex.get('category')}`\n")
                f.write(f"- source: `{ex.get('source')}`\n")
                f.write(f"- user: {ex.get('user')}\n")
                f.write(f"- assistant: {ex.get('assistant')}\n\n")


def parse_args() -> argparse.Namespace:
    root = get_project_root()
    parser = argparse.ArgumentParser(description="Clean and validate SFT seed + LLM-expanded candidate data.")

    parser.add_argument("--candidate-input", type=Path, default=root / "data" / "interim" / "sft_candidates.jsonl")
    parser.add_argument("--output", type=Path, default=root / "data" / "interim" / "cleaned_sft.jsonl")
    parser.add_argument("--report-md", type=Path, default=root / "docs" / "sft_clean_report.md")
    parser.add_argument("--report-json", type=Path, default=root / "docs" / "sft_clean_report.json")
    parser.add_argument("--min-user-len", type=int, default=4)
    parser.add_argument("--min-answer-len", type=int, default=20)
    parser.add_argument("--max-user-len", type=int, default=160)
    parser.add_argument("--max-answer-len", type=int, default=800)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.94)
    parser.add_argument("--disable-near-duplicate-filter", action="store_true")
    parser.add_argument("--allow-unknown-category", action="store_true")
    parser.add_argument("--disable-user-answer-style-check", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not (0 < args.near_duplicate_threshold <= 1):
        raise ValueError("--near-duplicate-threshold must be in (0, 1].")

    candidate_items, candidate_errors = read_jsonl(args.candidate_input, required=True)

    input_errors = Counter()
    input_errors.update({f"candidate_{k}": v for k, v in candidate_errors.items()})

    raw_items = candidate_items

    cleaned, drop_reasons, dropped_examples = clean_items(
        raw_items=raw_items,
        min_user_len=args.min_user_len,
        min_answer_len=args.min_answer_len,
        max_user_len=args.max_user_len,
        max_answer_len=args.max_answer_len,
        near_duplicate_threshold=args.near_duplicate_threshold,
        enable_near_duplicate_filter=not args.disable_near_duplicate_filter,
        strict_category=not args.allow_unknown_category,
        check_user_answer_style=not args.disable_user_answer_style_check,
    )

    write_jsonl(args.output, cleaned)

    report = build_report(
        raw_items=raw_items,
        cleaned_items=cleaned,
        input_errors=input_errors,
        drop_reasons=drop_reasons,
        dropped_examples=dropped_examples,
    )

    write_json(args.report_json, report)
    write_markdown_report(args.report_md, report, args.output)

    print("[SUMMARY]")
    print(f"  candidate input: {args.candidate_input} ({len(candidate_items)} items)")
    print(f"  raw items: {len(raw_items)}")
    print(f"  cleaned items: {len(cleaned)}")
    print(f"  dropped items: {len(raw_items) - len(cleaned)}")
    print(f"  output: {args.output}")
    print(f"  markdown report: {args.report_md}")
    print(f"  json report: {args.report_json}")

    if drop_reasons:
        print("  drop reasons:")
        for reason, count in drop_reasons.most_common():
            print(f"    - {reason}: {count}")


if __name__ == "__main__":
    main()
