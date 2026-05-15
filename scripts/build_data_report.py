#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成项目数据集构建总报告 docs/data_report.md。

输入：
    docs/sft_clean_report.json
    docs/sft_split_report.json
    docs/eval_set_report.json
    docs/dpo_data_report.json
    docs/dataset_quality_report.json

输出：
    docs/data_report.md

运行：
    python scripts/build_data_report.py
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# -----------------------------
# IO helpers
# -----------------------------

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if default is None:
        default = {}
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else default
    except Exception:
        return default


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                items.append(item)
    return items


def count_by(items: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    return dict(Counter(str(item.get(key, "unknown")) for item in items))


def safe_get(obj: Dict[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "_无数据_\n"
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |"
    body_lines = ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join([header_line, sep_line] + body_lines) + "\n"


def distribution_table(distribution: Dict[str, Any], key_name: str = "项目", value_name: str = "数量") -> str:
    if not distribution:
        return "_无数据_\n"
    rows = [(k, v) for k, v in sorted(distribution.items(), key=lambda x: str(x[0]))]
    return md_table([key_name, value_name], rows)


def merged_split_table(all_dist: Dict[str, int], train_dist: Dict[str, int], valid_dist: Dict[str, int], key_name: str) -> str:
    keys = sorted(set(all_dist) | set(train_dist) | set(valid_dist))
    rows = [[k, all_dist.get(k, 0), train_dist.get(k, 0), valid_dist.get(k, 0)] for k in keys]
    return md_table([key_name, "All", "Train", "Valid"], rows)


def get_dataset_quality_status(quality_report: Dict[str, Any]) -> Tuple[str, int, int]:
    status = "PASS" if quality_report.get("passed") else "FAIL"
    by_severity = safe_get(quality_report, ["issue_summary", "by_severity"], {}) or {}
    return status, int(by_severity.get("error", 0)), int(by_severity.get("warning", 0))


def extract_threshold_rows(quality_report: Dict[str, Any]) -> List[List[Any]]:
    threshold_results = quality_report.get("threshold_results", {})
    rows: List[List[Any]] = []
    for name, result in threshold_results.items():
        rows.append([name, result.get("actual"), result.get("expected_min"), "PASS" if result.get("passed") else "FAIL"])
    return rows


def p(path: Path) -> str:
    return f"`{path}`"


# -----------------------------
# Report generation
# -----------------------------

def build_report(args: argparse.Namespace) -> str:
    sft_clean_report = read_json(args.sft_clean_report)
    sft_split_report = read_json(args.sft_split_report)
    eval_set_report = read_json(args.eval_set_report)
    dpo_report = read_json(args.dpo_report)
    quality_report = read_json(args.quality_report)

    sft_seed = read_jsonl(args.sft_seed)
    cleaned_sft = read_jsonl(args.cleaned_sft)
    sft_train = read_jsonl(args.sft_train)
    sft_valid = read_jsonl(args.sft_valid)
    kb_all = read_jsonl(args.kb_all)
    eval_set = read_jsonl(args.eval_set)
    dpo_candidates = read_jsonl(args.dpo_candidates)
    dpo_train = read_jsonl(args.dpo_train)
    dpo_valid = read_jsonl(args.dpo_valid)

    status, error_count, warning_count = get_dataset_quality_status(quality_report)
    lines: List[str] = []

    lines.append("# 电商客服大模型数据集构建报告\n")

    lines.append("## 1. 数据集目标\n")
    lines.append("本项目构建一套面向电商客服场景的大模型数据闭环，用于支持 **SFT 微调**、**RAG 知识增强**、**DPO 偏好对齐** 和 **离线评测**。\n")
    lines.append("数据集围绕基础客服问答、安全边界、可执行回复、转人工意识和 RAG 可追溯能力进行设计。\n\n")

    lines.append("## 2. 数据来源与合规说明\n")
    lines.append("数据来源包括人工构造的 FAQ/SOP/商品知识、100 条人工黄金 SFT seed、GLM API 生成的用户问法扩写、Eval 样本和 DPO rejected 回复。\n")
    lines.append("本项目不使用真实企业客服日志、手机号、身份证号、邮箱、真实订单号等用户隐私数据；各阶段脚本均包含敏感信息和错误承诺检查。\n\n")

    lines.append("## 3. 数据目录与最终交付物\n")
    file_rows = [
        ["SFT seed", p(args.sft_seed), len(sft_seed)],
        ["SFT cleaned", p(args.cleaned_sft), len(cleaned_sft)],
        ["SFT train", p(args.sft_train), len(sft_train)],
        ["SFT valid", p(args.sft_valid), len(sft_valid)],
        ["RAG KB", p(args.kb_all), len(kb_all)],
        ["Eval set", p(args.eval_set), len(eval_set)],
        ["DPO candidates", p(args.dpo_candidates), len(dpo_candidates)],
        ["DPO train", p(args.dpo_train), len(dpo_train)],
        ["DPO valid", p(args.dpo_valid), len(dpo_valid)],
    ]
    lines.append(md_table(["数据文件", "路径", "样本数"], file_rows))
    lines.append("过程报告文件：\n\n")
    lines.append(md_table(["报告", "路径"], [
        ["SFT 清洗报告", p(args.sft_clean_report)],
        ["SFT 划分报告", p(args.sft_split_report)],
        ["Eval 构建报告", p(args.eval_set_report)],
        ["DPO 构建报告", p(args.dpo_report)],
        ["全量质量验收报告", p(args.quality_report)],
    ]))

    lines.append("## 4. SFT 数据构建\n")
    sft_split_summary = sft_split_report.get("summary", {})
    lines.append("SFT 数据采用“人工黄金样本 + LLM 用户问法扩写”的方式构造。扩写阶段只生成新的 user query，不改写人工确认过的 assistant answer。\n\n")
    lines.append(md_table(["指标", "值"], [
        ["人工 seed 数", len(sft_seed)],
        ["cleaned_sft 数", len(cleaned_sft) or safe_get(sft_clean_report, ["summary", "cleaned_items"], 0)],
        ["sft_train 数", len(sft_train) or sft_split_summary.get("train_items", 0)],
        ["sft_valid 数", len(sft_valid) or sft_split_summary.get("valid_items", 0)],
        ["SFT split 实际 valid 比例", sft_split_summary.get("actual_valid_item_ratio", "N/A")],
    ]))
    lines.append("### 4.1 SFT 类别分布\n\n")
    lines.append(merged_split_table(count_by(sft_train + sft_valid, "category"), count_by(sft_train, "category"), count_by(sft_valid, "category"), "类别"))
    lines.append("### 4.2 SFT 来源分布\n\n")
    lines.append(merged_split_table(count_by(sft_train + sft_valid, "source"), count_by(sft_train, "source"), count_by(sft_valid, "source"), "来源"))
    sft_leakage = safe_get(quality_report, ["cross_checks", "sft_seed_leakage"], {}) or {}
    lines.append("### 4.3 SFT seed_id 泄漏检查\n\n")
    lines.append(md_table(["检查项", "值"], [
        ["train seed_id 数", sft_leakage.get("train_seed_ids", "N/A")],
        ["valid seed_id 数", sft_leakage.get("valid_seed_ids", "N/A")],
        ["泄漏数量", sft_leakage.get("leakage_count", "N/A")],
    ]))
    lines.append("同一个 `seed_id` 的扩写样本按组划分，不会同时进入 train 和 valid，用于降低同源扩写泄漏风险。\n\n")

    lines.append("## 5. RAG 知识库构建\n")
    kb_stats = safe_get(quality_report, ["datasets", "kb_all"], {}) or {}
    lines.append("RAG 知识库由 FAQ、SOP 和商品知识构成，最终合并为 `kb_all.jsonl`。\n\n")
    lines.append(md_table(["指标", "值"], [
        ["kb_all 总数", len(kb_all) or kb_stats.get("total", 0)],
        ["格式有效数", kb_stats.get("valid_format", "N/A")],
        ["唯一 doc_id 数", kb_stats.get("unique_doc_ids", "N/A")],
    ]))
    lines.append("### 5.1 doc_type 分布\n\n")
    lines.append(distribution_table(kb_stats.get("doc_type_distribution", count_by(kb_all, "doc_type")), "doc_type", "数量"))
    lines.append("### 5.2 category 分布\n\n")
    lines.append(distribution_table(kb_stats.get("category_distribution", count_by(kb_all, "category")), "category", "数量"))
    lines.append("RAG 质量检查包括 doc_id 唯一性、title/content 完整性、content 长度、重复 content、敏感信息检测，以及 Eval `expected_docs` 是否存在于 `kb_all`。\n\n")

    lines.append("## 6. Eval 离线评测集构建\n")
    eval_summary = eval_set_report.get("summary", {})
    eval_quality = safe_get(quality_report, ["datasets", "eval_set"], {}) or {}
    lines.append("Eval 集不从 SFT train/valid 中随机抽样，而是单独构造，用于评估 Base、SFT、SFT+RAG 在客服场景下的业务正确性、安全边界和转人工能力。\n\n")
    lines.append(md_table(["指标", "值"], [
        ["eval_set 总数", len(eval_set) or eval_summary.get("generated_items", 0)],
        ["格式有效数", eval_quality.get("valid_format", "N/A")],
        ["计划样本数", eval_summary.get("planned_items", "N/A")],
        ["删除样本数", eval_summary.get("dropped_items", "N/A")],
    ]))
    lines.append("### 6.1 Eval 类别分布\n\n")
    lines.append(distribution_table(eval_quality.get("category_distribution", count_by(eval_set, "category")), "类别", "数量"))
    lines.append("### 6.2 eval_type 分布\n\n")
    lines.append(distribution_table(eval_quality.get("eval_type_distribution", count_by(eval_set, "eval_type")), "eval_type", "数量"))
    lines.append("### 6.3 difficulty 分布\n\n")
    lines.append(distribution_table(eval_quality.get("difficulty_distribution", count_by(eval_set, "difficulty")), "difficulty", "数量"))
    lines.append("### 6.4 need_human / need_rag 分布\n\n")
    lines.append("**need_human**\n\n")
    lines.append(distribution_table(eval_quality.get("need_human_distribution", count_by(eval_set, "need_human")), "need_human", "数量"))
    lines.append("\n**need_rag**\n\n")
    lines.append(distribution_table(eval_quality.get("need_rag_distribution", count_by(eval_set, "need_rag")), "need_rag", "数量"))
    lines.append("### 6.5 risk_tags 分布\n\n")
    lines.append(distribution_table(eval_quality.get("risk_tag_distribution", {}), "risk_tag", "数量"))
    overlap = safe_get(quality_report, ["cross_checks", "eval_vs_sft_overlap"], {}) or {}
    lines.append("### 6.6 Eval 与 SFT 重复检查\n\n")
    lines.append(md_table(["检查项", "值"], [
        ["精确重复数", overlap.get("exact_overlap", "N/A")],
        ["近重复数", overlap.get("near_overlap", "N/A")],
    ]))
    if overlap.get("examples"):
        lines.append("示例：\n\n")
        for ex in overlap.get("examples", [])[:5]:
            lines.append(f"- `{ex}`\n")
        lines.append("\n")

    lines.append("## 7. DPO 偏好数据构建\n")
    dpo_summary = dpo_report.get("summary", {})
    dpo_candidates_q = safe_get(quality_report, ["datasets", "dpo_candidates"], {}) or {}
    dpo_train_q = safe_get(quality_report, ["datasets", "dpo_train"], {}) or {}
    dpo_valid_q = safe_get(quality_report, ["datasets", "dpo_valid"], {}) or {}
    lines.append("DPO 数据以 cleaned SFT 和 Eval 样本为基础：`prompt` 和 `chosen` 来自高质量答案，`rejected` 由 GLM API 按指定错误类型生成，脚本再进行格式、相似度、敏感信息和 chosen 风险承诺检查。\n\n")
    lines.append(md_table(["指标", "值"], [
        ["dpo_candidates", len(dpo_candidates) or dpo_summary.get("candidate_items", 0)],
        ["dpo_train", len(dpo_train) or dpo_summary.get("train_items", 0)],
        ["dpo_valid", len(dpo_valid) or dpo_summary.get("valid_items", 0)],
        ["实际 valid 比例", dpo_summary.get("actual_valid_ratio", "N/A")],
        ["删除样本数", dpo_summary.get("dropped_items", "N/A")],
    ]))
    lines.append("### 7.1 DPO 类别分布\n\n")
    lines.append(merged_split_table(dpo_candidates_q.get("category_distribution", count_by(dpo_candidates, "category")), dpo_train_q.get("category_distribution", count_by(dpo_train, "category")), dpo_valid_q.get("category_distribution", count_by(dpo_valid, "category")), "类别"))
    lines.append("### 7.2 rejected_type 分布\n\n")
    lines.append(merged_split_table(dpo_candidates_q.get("rejected_type_distribution", count_by(dpo_candidates, "rejected_type")), dpo_train_q.get("rejected_type_distribution", count_by(dpo_train, "rejected_type")), dpo_valid_q.get("rejected_type_distribution", count_by(dpo_valid, "rejected_type")), "rejected_type"))
    lines.append("### 7.3 source 分布\n\n")
    lines.append(merged_split_table(dpo_candidates_q.get("source_distribution", count_by(dpo_candidates, "source")), dpo_train_q.get("source_distribution", count_by(dpo_train, "source")), dpo_valid_q.get("source_distribution", count_by(dpo_valid, "source")), "source"))
    lines.append("### 7.4 severity 分布\n\n")
    lines.append(merged_split_table(dpo_candidates_q.get("severity_distribution", count_by(dpo_candidates, "severity")), dpo_train_q.get("severity_distribution", count_by(dpo_train, "severity")), dpo_valid_q.get("severity_distribution", count_by(dpo_valid, "severity")), "severity"))

    lines.append("## 8. 数据清洗与质量控制规则\n")
    lines.append("### 8.1 SFT 质量控制\n")
    lines.append("- 检查 messages 是否为 `system -> user -> assistant`。\n- 检查 category 是否属于 8 个合法业务类别。\n- 检查 user / assistant 长度，删除空内容、过短内容。\n- 检查手机号、身份证号、邮箱、明显订单号等敏感信息。\n- 检查 assistant 中是否存在绝对化退款、赔偿、送达等错误承诺。\n- 按 `seed_id` 分组划分 train/valid，降低同源扩写泄漏。\n\n")
    lines.append("### 8.2 Eval 质量控制\n")
    lines.append("- 检查 `eval_type`、`difficulty`、`must_include`、`must_not_include`、`risk_tags` 等字段。\n- `eval_type=rag` 时要求 `need_rag=true` 且 `expected_docs` 非空。\n- `eval_type=human_transfer` 时要求 `need_human=true`。\n- 对 `expected_docs` 做 doc_id 规范化。\n- 对 RAG 样本进行轻量相关性检查，避免 query 与 expected_docs 文档不匹配。\n- 检查 Eval query 与 SFT user 的精确重复和近重复。\n\n")
    lines.append("### 8.3 DPO 质量控制\n")
    lines.append("- 检查 prompt 是否包含 system + user。\n- 检查 chosen/rejected 是否各包含一个 assistant 回复。\n- 检查 rejected_type 是否属于 9 类预定义错误类型。\n- 检查 chosen/rejected 相似度，避免偏好差异过弱。\n- 检查 chosen 是否包含错误承诺。\n- 检查 preference_reason 是否有效。\n\n")
    lines.append("### 8.4 RAG KB 质量控制\n")
    lines.append("- 检查 chunk_id / product_id / id 是否唯一。\n- 检查 doc_type、category、title、content 字段。\n- 检查 content 长度、重复 content 和敏感信息。\n- 检查 Eval expected_docs 是否能在 kb_all 中找到。\n\n")

    lines.append("## 9. 最终质量验收结果\n")
    lines.append(f"最终验收状态：**{status}**\n\n")
    lines.append(md_table(["指标", "值"], [["错误数", error_count], ["警告数", warning_count], ["Issue 总数", safe_get(quality_report, ["issue_summary", "total"], "N/A")]]))
    lines.append("### 9.1 数量门槛检查\n\n")
    lines.append(md_table(["检查项", "实际值", "最低要求", "是否通过"], extract_threshold_rows(quality_report)))
    lines.append("### 9.2 Issue 类型分布\n\n")
    lines.append(distribution_table(safe_get(quality_report, ["issue_summary", "by_type"], {}) or {}, "issue_type", "数量"))

    lines.append("## 10. Badcase 与修复记录\n")
    badcase_rows = [
        ["Eval expected_docs 格式错误", "expected_docs 被模型写成 doc_id=xxx 或包含整段 metadata。", "增加 normalize_doc_id()，只保留裸 doc_id。"],
        ["RAG query 与 expected_docs 错配", "例如物流问题绑定到换货申请文档。", "增加 batch 内多 category KB context，并加入 rag_docs_relevant() 校验。"],
        ["Eval 自动补齐不足", "部分样本被校验 drop，导致不足 200 条。", "改为 while len(final_items) < total 持续补齐。"],
        ["大 batch 导致 JSON 截断", "一次请求过多样本导致模型输出 JSON 不完整。", "限制单轮最大生成数量，采用小 batch 稳定补齐。"],
        ["DPO rejected_type 匹配过严", "关键词规则过窄导致可用 rejected 被误删。", "允许关闭强匹配，并通过人工抽查保证质量。"],
        ["Eval 与 SFT 近重复", "Eval query 与 SFT user 高度相似。", "人工替换重复样本或提高近重复阈值后重新检查。"],
    ]
    lines.append(md_table(["问题", "现象", "修复方式"], badcase_rows))

    lines.append("## 11. 当前数据集局限性\n")
    lines.append("- 当前数据以合成数据和人工设计数据为主，真实客服噪声、错别字、多轮上下文仍然不足。\n")
    lines.append("- RAG 知识库规模较小，商品知识和 SOP 覆盖仍有限。\n")
    lines.append("- Eval 虽有自动规则校验，但仍需要人工抽样复核，尤其是 RAG 样本的文档相关性。\n")
    lines.append("- DPO rejected 由 LLM 生成，虽然通过规则清洗，但仍建议每类 rejected_type 抽样复核。\n")
    lines.append("- 当前主要为单轮问答数据，多轮售后协商、连续追问和跨轮状态管理还未充分覆盖。\n\n")

    lines.append("## 12. 下一步优化方向\n")
    lines.append("- 将 SFT 数据扩展到 5,000 条以上，并加入更多真实噪声表达。\n")
    lines.append("- 将 RAG KB 扩展到 300+ chunk，覆盖更多商品、规则和 SOP。\n")
    lines.append("- 将 Eval 扩展到 500 条，并增加人工评审标注。\n")
    lines.append("- 引入 badcase_seed.jsonl，将模型真实失败案例持续沉淀到 SFT / DPO / Eval。\n")
    lines.append("- 增加 run_eval.py，支持 Base / SFT / SFT+RAG 的自动化离线评测。\n")
    lines.append("- 增加 LLM-as-judge 或人工评分表，对 must_include、must_not_include、need_human 等维度进行组合评估。\n\n")

    lines.append("## 13. 复现命令\n")
    lines.append("```bash\n")
    lines.append("python scripts/expand_sft_with_llm.py --model glm-4-flash-250414 --variants-per-seed 25 --overwrite\n")
    lines.append("python scripts/clean_sft_data.py\n")
    lines.append("python scripts/split_sft_dataset.py\n")
    lines.append("python scripts/build_eval_set_with_llm.py --model glm-4-flash-250414 --total 200 --batch-size 10 --overwrite\n")
    lines.append("python scripts/build_dpo_data_with_llm.py --model glm-4-flash-250414 --target-total 300 --overwrite --disable-rejected-type-match\n")
    lines.append("python scripts/check_dataset.py\n")
    lines.append("python scripts/build_data_report.py\n")
    lines.append("```\n")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    root = get_project_root()
    parser = argparse.ArgumentParser(description="Build final dataset construction report.")

    parser.add_argument("--sft-seed", type=Path, default=root / "data" / "raw" / "sft_seed.jsonl")
    parser.add_argument("--cleaned-sft", type=Path, default=root / "data" / "interim" / "cleaned_sft.jsonl")
    parser.add_argument("--sft-train", type=Path, default=root / "data" / "processed" / "sft_train.jsonl")
    parser.add_argument("--sft-valid", type=Path, default=root / "data" / "processed" / "sft_valid.jsonl")
    parser.add_argument("--kb-all", type=Path, default=root / "data" / "knowledge_base" / "kb_all.jsonl")
    parser.add_argument("--eval-set", type=Path, default=root / "data" / "eval" / "eval_set.jsonl")
    parser.add_argument("--dpo-candidates", type=Path, default=root / "data" / "interim" / "dpo_candidates.jsonl")
    parser.add_argument("--dpo-train", type=Path, default=root / "data" / "processed" / "dpo_train.jsonl")
    parser.add_argument("--dpo-valid", type=Path, default=root / "data" / "processed" / "dpo_valid.jsonl")

    parser.add_argument("--sft-clean-report", type=Path, default=root / "docs" / "sft_clean_report.json")
    parser.add_argument("--sft-split-report", type=Path, default=root / "docs" / "sft_split_report.json")
    parser.add_argument("--eval-set-report", type=Path, default=root / "docs" / "eval_set_report.json")
    parser.add_argument("--dpo-report", type=Path, default=root / "docs" / "dpo_data_report.json")
    parser.add_argument("--quality-report", type=Path, default=root / "docs" / "dataset_quality_report.json")
    parser.add_argument("--output", type=Path, default=root / "docs" / "data_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_text = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_text, encoding="utf-8")
    print("[SUMMARY]")
    print(f"  output: {args.output}")
    print(f"  size: {args.output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
