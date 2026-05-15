#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SFT 训练集 / 验证集划分脚本。

输入：
    data/interim/cleaned_sft.jsonl

输出：
    data/processed/sft_train.jsonl
    data/processed/sft_valid.jsonl
    docs/sft_split_report.md
    docs/sft_split_report.json

设计目标：
    1. 按 category 分层划分，尽量保持训练集和验证集类别分布一致；
    2. 优先按 seed_id 分组划分，避免同一条黄金样本的扩写同时进入 train 和 valid；
    3. 如果样本没有 seed_id，则使用自身 id 作为分组 key；
    4. 输出划分报告，便于检查类别分布、来源分布和 seed 泄漏情况。
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


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


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    items: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = line.strip()
            if not raw:
                continue

            try:
                item = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}, line {line_no}: {e}") from e

            if not isinstance(item, dict):
                raise ValueError(f"JSONL item must be object at {path}, line {line_no}")

            items.append(item)

    return items


def write_jsonl(path: Path, items: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_messages(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = item.get("messages", [])
    if not isinstance(messages, list):
        return []
    return messages


def validate_item(item: Dict[str, Any]) -> Tuple[bool, str]:
    category = item.get("category")
    if category not in VALID_CATEGORIES:
        return False, "invalid_category"

    messages = get_messages(item)
    if not messages:
        return False, "missing_messages"

    roles = [msg.get("role") for msg in messages if isinstance(msg, dict)]
    if "user" not in roles:
        return False, "missing_user"
    if "assistant" not in roles:
        return False, "missing_assistant"

    return True, "ok"


def get_group_key(item: Dict[str, Any]) -> str:
    seed_id = item.get("seed_id")
    if seed_id:
        return f"seed::{seed_id}"

    item_id = item.get("id")
    if item_id:
        return f"item::{item_id}"

    return f"object::{id(item)}"


def get_group_category(items: Sequence[Dict[str, Any]]) -> str:
    counter = Counter(str(item.get("category", "unknown")) for item in items)
    return counter.most_common(1)[0][0]


def build_groups(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for item in items:
        grouped[get_group_key(item)].append(item)

    groups: List[Dict[str, Any]] = []

    for group_key, group_items in grouped.items():
        groups.append({
            "group_key": group_key,
            "category": get_group_category(group_items),
            "size": len(group_items),
            "items": group_items,
            "category_counter": dict(Counter(str(x.get("category", "unknown")) for x in group_items)),
        })

    return groups


def stratified_group_split(
    groups: Sequence[Dict[str, Any]],
    valid_ratio: float,
    seed: int,
    min_valid_groups_per_category: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not (0 < valid_ratio < 1):
        raise ValueError("--valid-ratio must be in (0, 1).")

    rng = random.Random(seed)

    groups_by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for group in groups:
        groups_by_category[group["category"]].append(group)

    train_groups: List[Dict[str, Any]] = []
    valid_groups: List[Dict[str, Any]] = []

    for category, category_groups in groups_by_category.items():
        category_groups = list(category_groups)
        rng.shuffle(category_groups)

        n_groups = len(category_groups)

        if n_groups == 1:
            train_groups.extend(category_groups)
            continue

        n_valid = int(round(n_groups * valid_ratio))
        n_valid = max(min_valid_groups_per_category, n_valid)
        n_valid = min(n_valid, n_groups - 1)

        valid_groups.extend(category_groups[:n_valid])
        train_groups.extend(category_groups[n_valid:])

    rng.shuffle(train_groups)
    rng.shuffle(valid_groups)

    return train_groups, valid_groups


def flatten_groups(groups: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for group in groups:
        items.extend(group["items"])
    return items


def count_by(items: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    return dict(Counter(str(item.get(key, "unknown")) for item in items))


def count_groups_by_category(groups: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(str(group.get("category", "unknown")) for group in groups))


def detect_seed_leakage(train_items: Sequence[Dict[str, Any]], valid_items: Sequence[Dict[str, Any]]) -> List[str]:
    train_seed_ids = {str(item.get("seed_id")) for item in train_items if item.get("seed_id")}
    valid_seed_ids = {str(item.get("seed_id")) for item in valid_items if item.get("seed_id")}
    return sorted(train_seed_ids & valid_seed_ids)


def group_category_conflicts(groups: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    conflicts: List[Dict[str, Any]] = []

    for group in groups:
        category_counter = group.get("category_counter", {})
        if len(category_counter) > 1:
            conflicts.append({
                "group_key": group["group_key"],
                "category_counter": category_counter,
                "size": group["size"],
            })

    return conflicts


def build_report(
    input_path: Path,
    train_output: Path,
    valid_output: Path,
    raw_items: Sequence[Dict[str, Any]],
    valid_items: Sequence[Dict[str, Any]],
    train_items: Sequence[Dict[str, Any]],
    groups: Sequence[Dict[str, Any]],
    train_groups: Sequence[Dict[str, Any]],
    valid_groups: Sequence[Dict[str, Any]],
    invalid_items: Sequence[Dict[str, Any]],
    valid_ratio: float,
    seed: int,
) -> Dict[str, Any]:
    seed_leakage = detect_seed_leakage(train_items, valid_items)

    return {
        "config": {
            "input": str(input_path),
            "train_output": str(train_output),
            "valid_output": str(valid_output),
            "valid_ratio": valid_ratio,
            "seed": seed,
            "split_strategy": "stratified_group_split_by_category_and_seed_id",
        },
        "summary": {
            "raw_items": len(raw_items),
            "valid_format_items": len(valid_items) + len(train_items),
            "invalid_items": len(invalid_items),
            "total_groups": len(groups),
            "train_items": len(train_items),
            "valid_items": len(valid_items),
            "train_groups": len(train_groups),
            "valid_groups": len(valid_groups),
            "actual_valid_item_ratio": round(len(valid_items) / max(len(train_items) + len(valid_items), 1), 4),
            "actual_valid_group_ratio": round(len(valid_groups) / max(len(train_groups) + len(valid_groups), 1), 4),
            "seed_leakage_count": len(seed_leakage),
        },
        "item_category_distribution": {
            "all": count_by(list(train_items) + list(valid_items), "category"),
            "train": count_by(train_items, "category"),
            "valid": count_by(valid_items, "category"),
        },
        "group_category_distribution": {
            "all": count_groups_by_category(groups),
            "train": count_groups_by_category(train_groups),
            "valid": count_groups_by_category(valid_groups),
        },
        "source_distribution": {
            "all": count_by(list(train_items) + list(valid_items), "source"),
            "train": count_by(train_items, "source"),
            "valid": count_by(valid_items, "source"),
        },
        "seed_leakage": seed_leakage,
        "group_category_conflicts": group_category_conflicts(groups),
        "invalid_items": list(invalid_items[:100]),
    }


def write_distribution_table(f, distribution: Dict[str, Dict[str, int]]) -> None:
    all_keys = sorted(
        set(distribution.get("all", {}).keys())
        | set(distribution.get("train", {}).keys())
        | set(distribution.get("valid", {}).keys())
    )

    f.write("| 类别/来源 | All | Train | Valid |\n")
    f.write("|---|---:|---:|---:|\n")

    for key in all_keys:
        all_count = distribution.get("all", {}).get(key, 0)
        train_count = distribution.get("train", {}).get(key, 0)
        valid_count = distribution.get("valid", {}).get(key, 0)
        f.write(f"| {key} | {all_count} | {train_count} | {valid_count} |\n")

    f.write("\n")


def write_markdown_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    summary = report["summary"]
    config = report["config"]

    with path.open("w", encoding="utf-8") as f:
        f.write("# SFT Train / Valid 划分报告\n\n")

        f.write("## 1. 配置\n\n")
        f.write(f"- 输入文件：`{config['input']}`\n")
        f.write(f"- 训练集输出：`{config['train_output']}`\n")
        f.write(f"- 验证集输出：`{config['valid_output']}`\n")
        f.write(f"- 验证集比例：{config['valid_ratio']}\n")
        f.write(f"- 随机种子：{config['seed']}\n")
        f.write(f"- 划分策略：`{config['split_strategy']}`\n\n")

        f.write("## 2. 总览\n\n")
        f.write(f"- 原始样本数：{summary['raw_items']}\n")
        f.write(f"- 格式有效样本数：{summary['valid_format_items']}\n")
        f.write(f"- 格式无效样本数：{summary['invalid_items']}\n")
        f.write(f"- 分组总数：{summary['total_groups']}\n")
        f.write(f"- 训练集样本数：{summary['train_items']}\n")
        f.write(f"- 验证集样本数：{summary['valid_items']}\n")
        f.write(f"- 训练集 group 数：{summary['train_groups']}\n")
        f.write(f"- 验证集 group 数：{summary['valid_groups']}\n")
        f.write(f"- 实际验证集样本比例：{summary['actual_valid_item_ratio']}\n")
        f.write(f"- 实际验证集 group 比例：{summary['actual_valid_group_ratio']}\n")
        f.write(f"- seed_id 泄漏数量：{summary['seed_leakage_count']}\n\n")

        f.write("## 3. 样本类别分布\n\n")
        write_distribution_table(f, report["item_category_distribution"])

        f.write("## 4. Group 类别分布\n\n")
        write_distribution_table(f, report["group_category_distribution"])

        f.write("## 5. 来源分布\n\n")
        write_distribution_table(f, report["source_distribution"])

        f.write("## 6. Seed 泄漏检查\n\n")
        seed_leakage = report.get("seed_leakage", [])
        if seed_leakage:
            f.write("以下 seed_id 同时出现在 train 和 valid 中，需要检查：\n\n")
            for seed_id in seed_leakage:
                f.write(f"- {seed_id}\n")
        else:
            f.write("- 未发现 seed_id 同时出现在 train 和 valid。\n")
        f.write("\n")

        f.write("## 7. Group 内类别冲突\n\n")
        conflicts = report.get("group_category_conflicts", [])
        if conflicts:
            f.write("以下 group 内存在多个 category，建议检查数据源：\n\n")
            for item in conflicts[:50]:
                f.write(f"- `{item['group_key']}`: {item['category_counter']}\n")
        else:
            f.write("- 未发现 group 内类别冲突。\n")
        f.write("\n")

        f.write("## 8. 无效样本\n\n")
        invalid_items = report.get("invalid_items", [])
        if invalid_items:
            for item in invalid_items[:30]:
                f.write(f"- id: `{item.get('id')}`, reason: `{item.get('reason')}`, category: `{item.get('category')}`\n")
        else:
            f.write("- 无。\n")


def parse_args() -> argparse.Namespace:
    root = get_project_root()

    parser = argparse.ArgumentParser(
        description="Split cleaned SFT JSONL into train and valid sets."
    )

    parser.add_argument("--input", type=Path, default=root / "data" / "interim" / "cleaned_sft.jsonl")
    parser.add_argument("--train-output", type=Path, default=root / "data" / "processed" / "sft_train.jsonl")
    parser.add_argument("--valid-output", type=Path, default=root / "data" / "processed" / "sft_valid.jsonl")
    parser.add_argument("--report-md", type=Path, default=root / "docs" / "sft_split_report.md")
    parser.add_argument("--report-json", type=Path, default=root / "docs" / "sft_split_report.json")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-valid-groups-per-category", type=int, default=1)
    parser.add_argument("--shuffle-items-within-split", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_items = read_jsonl(args.input)

    valid_format_items: List[Dict[str, Any]] = []
    invalid_items: List[Dict[str, Any]] = []

    for item in raw_items:
        ok, reason = validate_item(item)
        if ok:
            valid_format_items.append(item)
        else:
            invalid_items.append({
                "id": item.get("id"),
                "seed_id": item.get("seed_id"),
                "category": item.get("category"),
                "source": item.get("source"),
                "reason": reason,
            })

    groups = build_groups(valid_format_items)

    train_groups, valid_groups = stratified_group_split(
        groups=groups,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        min_valid_groups_per_category=args.min_valid_groups_per_category,
    )

    train_items = flatten_groups(train_groups)
    valid_items = flatten_groups(valid_groups)

    if args.shuffle_items_within_split:
        rng = random.Random(args.seed)
        rng.shuffle(train_items)
        rng.shuffle(valid_items)

    write_jsonl(args.train_output, train_items)
    write_jsonl(args.valid_output, valid_items)

    report = build_report(
        input_path=args.input,
        train_output=args.train_output,
        valid_output=args.valid_output,
        raw_items=raw_items,
        valid_items=valid_items,
        train_items=train_items,
        groups=groups,
        train_groups=train_groups,
        valid_groups=valid_groups,
        invalid_items=invalid_items,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

    write_json(args.report_json, report)
    write_markdown_report(args.report_md, report)

    print("[SUMMARY]")
    print(f"  input: {args.input}")
    print(f"  raw items: {len(raw_items)}")
    print(f"  valid format items: {len(valid_format_items)}")
    print(f"  invalid items: {len(invalid_items)}")
    print(f"  groups: {len(groups)}")
    print(f"  train groups: {len(train_groups)}")
    print(f"  valid groups: {len(valid_groups)}")
    print(f"  train items: {len(train_items)}")
    print(f"  valid items: {len(valid_items)}")
    print(f"  actual valid item ratio: {len(valid_items) / max(len(train_items) + len(valid_items), 1):.4f}")
    print(f"  train output: {args.train_output}")
    print(f"  valid output: {args.valid_output}")
    print(f"  markdown report: {args.report_md}")
    print(f"  json report: {args.report_json}")

    seed_leakage = report["seed_leakage"]
    if seed_leakage:
        print(f"[WARN] seed leakage detected: {len(seed_leakage)} seed_id(s)")
    else:
        print("[OK] no seed_id leakage between train and valid")


if __name__ == "__main__":
    main()
