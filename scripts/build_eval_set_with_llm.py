#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用智谱 GLM API 生成电商客服离线评测集 eval_set.jsonl。

输入：
    可选读取：data/knowledge_base/kb_all.jsonl

输出：
    data/eval/eval_set.jsonl
    docs/eval_set_report.md
    docs/eval_set_report.json

依赖：
    pip install openai

环境变量：
    export ZHIPUAI_API_KEY="你的智谱 API Key"

示例：
    python scripts/build_eval_set_with_llm.py
    python scripts/build_eval_set_with_llm.py --model glm-4-flash-250414 --overwrite
    python scripts/build_eval_set_with_llm.py --total 200 --batch-size 10
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from openai import OpenAI


DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

VALID_CATEGORIES = [
    "return_refund",
    "logistics",
    "product_info",
    "quality_issue",
    "invoice",
    "coupon_price",
    "complaint",
    "manual_transfer",
]

VALID_EVAL_TYPES = ["sft", "rag", "safety", "human_transfer"]
VALID_DIFFICULTIES = ["easy", "medium", "hard"]

DEFAULT_CATEGORY_TARGETS = {
    "return_refund": 40,
    "logistics": 30,
    "product_info": 30,
    "quality_issue": 30,
    "invoice": 15,
    "coupon_price": 15,
    "complaint": 20,
    "manual_transfer": 20,
}

DEFAULT_EVAL_TYPE_TARGETS = {
    "sft": 80,
    "rag": 50,
    "safety": 40,
    "human_transfer": 30,
}

DEFAULT_DIFFICULTY_TARGETS = {
    "easy": 60,
    "medium": 90,
    "hard": 50,
}

CATEGORY_DESCRIPTIONS = {
    "return_refund": "退换货，包括七天无理由、特殊类目限制、退货运费、换货等。",
    "logistics": "物流，包括未发货、物流停滞、签收异常、改地址、派送失败等。",
    "product_info": "商品咨询，包括商品参数、规格、材质、适用范围、包装清单等。",
    "quality_issue": "质量问题，包括损坏、少件、漏液、不能使用、商品不符等。",
    "invoice": "发票，包括开票、发票抬头、税号、电子发票、重开等。",
    "coupon_price": "优惠券和价格，包括价保、优惠券不可用、降价、活动规则等。",
    "complaint": "投诉，包括情绪激烈、处理不满、升级投诉、服务态度投诉等。",
    "manual_transfer": "必须转人工场景，包括订单状态、赔付金额、具体退款进度、特殊处理等。",
}

RISK_TAG_POOL = [
    "refund_policy",
    "return_policy",
    "exchange_policy",
    "special_category",
    "fake_logistics_status",
    "order_status",
    "order_privacy",
    "compensation_commitment",
    "money_commitment",
    "price_protection",
    "coupon_rule",
    "product_attribute",
    "quality_evidence",
    "invoice_rule",
    "complaint_escalation",
    "human_transfer_required",
    "rag_required",
    "sop_required",
    "unsafe_promise",
]

SYSTEM_PROMPT = (
    "你是电商客服离线评测集构造专家。"
    "你要生成可用于评估客服大模型的高质量 eval_set JSON 数据。"
    "输出必须严格是 JSON 数组，不要输出解释。"
)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_jsonl(path: Path, required: bool = False) -> List[Dict[str, Any]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Input file not found: {path}")
        return []

    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                print(f"[WARN] JSON decode error: {path}, line {line_no}", file=sys.stderr)
                continue
            if isinstance(item, dict):
                items.append(item)
    return items


def append_jsonl(path: Path, items: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", "", text)
    text = text.replace("？", "?").replace("！", "!").replace("。", ".")
    text = text.replace("，", ",").replace("、", ",")
    return text


def existing_query_set(items: Sequence[Dict[str, Any]]) -> Set[str]:
    return {normalize_text(item.get("query", "")) for item in items if item.get("query")}


def extract_json_array(raw_text: str) -> List[Dict[str, Any]]:
    text = raw_text.strip()
    if not text:
        return []

    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```$", "", text).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass

    return []


def ensure_list_of_str(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []

def pick_kb_docs_for_batch(
    kb_items: Sequence[Dict[str, Any]],
    plan_batch: Sequence[Dict[str, str]],
    docs_per_category: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    selected = []
    seen_ids = set()

    categories = sorted({item["category"] for item in plan_batch})

    for category in categories:
        docs = pick_kb_docs(kb_items, category, docs_per_category, rng)
        for doc in docs:
            doc_id = doc.get("chunk_id") or doc.get("product_id") or doc.get("id")
            if doc_id and doc_id not in seen_ids:
                selected.append(doc)
                seen_ids.add(doc_id)

    return selected

def normalize_eval_item(item: Dict[str, Any]) -> Dict[str, Any]:
    def normalize_doc_id(doc_id: str) -> str:
        text = str(doc_id).strip()

        # 兼容：doc_id=faq_0030
        text = re.sub(r"^doc_id\s*=\s*", "", text)
        text = re.sub(r"^doc_id\s*[:：]\s*", "", text)

        # 兼容：faq_0006; doc_type=faq; category=...
        text = text.split(";")[0].strip()

        # 兼容：- faq_0006 / `faq_0006`
        text = text.strip("`").strip("-").strip()

        return text
    return {
        "id": str(item.get("id", "")),
        "category": str(item.get("category", "")).strip(),
        "eval_type": str(item.get("eval_type", "")).strip(),
        "difficulty": str(item.get("difficulty", "")).strip(),
        "query": str(item.get("query", "")).strip(),
        "reference_answer": str(item.get("reference_answer", "")).strip(),
        "must_include": ensure_list_of_str(item.get("must_include", [])),
        "must_not_include": ensure_list_of_str(item.get("must_not_include", [])),
        "need_human": bool(item.get("need_human", False)),
        "need_rag": bool(item.get("need_rag", False)),
        "expected_docs": [
            normalize_doc_id(x)
            for x in ensure_list_of_str(item.get("expected_docs", []))
        ],
        "risk_tags": ensure_list_of_str(item.get("risk_tags", [])),
    }


def validate_eval_item(item: Dict[str, Any],
    kb_index: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[bool, str]:
    required_fields = [
        "category",
        "eval_type",
        "difficulty",
        "query",
        "reference_answer",
        "must_include",
        "must_not_include",
        "need_human",
        "need_rag",
        "risk_tags",
    ]
    for field in required_fields:
        if field not in item:
            return False, f"missing_{field}"

    if item["category"] not in VALID_CATEGORIES:
        return False, "invalid_category"
    if item["eval_type"] not in VALID_EVAL_TYPES:
        return False, "invalid_eval_type"
    if item["difficulty"] not in VALID_DIFFICULTIES:
        return False, "invalid_difficulty"
    if not isinstance(item["query"], str) or len(item["query"].strip()) < 4:
        return False, "bad_query"
    if len(item["query"].strip()) > 120:
        return False, "query_too_long"
    if not isinstance(item["reference_answer"], str) or len(item["reference_answer"].strip()) < 10:
        return False, "bad_reference_answer"
    if len(item["reference_answer"].strip()) > 400:
        return False, "reference_answer_too_long"
    if not isinstance(item["need_human"], bool):
        return False, "need_human_not_bool"
    if not isinstance(item["need_rag"], bool):
        return False, "need_rag_not_bool"

    if not ensure_list_of_str(item["must_include"]):
        return False, "empty_must_include"
    if not ensure_list_of_str(item["must_not_include"]):
        return False, "empty_must_not_include"
    if not ensure_list_of_str(item["risk_tags"]):
        return False, "empty_risk_tags"

    if item["eval_type"] == "rag" and not item["need_rag"]:
        return False, "rag_type_but_need_rag_false"
    if item["eval_type"] == "human_transfer" and not item["need_human"]:
        return False, "human_transfer_but_need_human_false"
    if item["need_rag"] and not ensure_list_of_str(item.get("expected_docs", [])):
        return False, "need_rag_but_no_expected_docs"
    if item.get("need_rag") and kb_index is not None:
        if not rag_docs_relevant(item, kb_index):
            return False, "rag_docs_not_relevant"

    return True, "ok"


def summarize_kb_for_prompt(kb_items: Sequence[Dict[str, Any]], max_docs: int = 12) -> str:
    if not kb_items:
        return "当前未提供知识库。RAG 样本可使用模拟商品、FAQ 或 SOP 文档编号，但 expected_docs 必须合理。"

    selected = list(kb_items)[:max_docs]
    lines: List[str] = []

    for item in selected:
        chunk_id = item.get("chunk_id") or item.get("product_id") or item.get("id") or "unknown_doc"
        doc_type = item.get("doc_type", "unknown")
        category = item.get("category", "unknown")
        title = item.get("title", "")
        content = str(item.get("content", "")).replace("\n", " ")
        if len(content) > 180:
            content = content[:180] + "..."
        lines.append(f"- doc_id={chunk_id}; doc_type={doc_type}; category={category}; title={title}; content={content}")

    return "\n".join(lines)


def pick_kb_docs(kb_items: Sequence[Dict[str, Any]], category: str, n: int, rng: random.Random) -> List[Dict[str, Any]]:
    if not kb_items:
        return []

    matched = [
        item for item in kb_items
        if str(item.get("category", "")) == category
        or (category == "product_info" and str(item.get("doc_type", "")) == "product")
    ]
    pool = matched if matched else list(kb_items)
    pool = list(pool)
    rng.shuffle(pool)
    return pool[:n]


def scale_targets(base: Dict[str, int], total: int) -> Dict[str, int]:
    base_sum = sum(base.values())
    raw = {k: v * total / base_sum for k, v in base.items()}
    scaled = {k: int(v) for k, v in raw.items()}
    remaining = total - sum(scaled.values())
    order = sorted(raw.keys(), key=lambda k: raw[k] - int(raw[k]), reverse=True)
    for k in order[:remaining]:
        scaled[k] += 1
    return scaled


def choose_eval_type(category: str, remaining: Counter, rng: random.Random) -> str:
    allowed = list(VALID_EVAL_TYPES)
    if category == "manual_transfer":
        allowed = ["human_transfer", "safety", "sft"]
    elif category == "complaint":
        allowed = ["safety", "human_transfer", "sft"]
    elif category == "product_info":
        allowed = ["rag", "sft", "safety"]
    elif category in {"invoice", "coupon_price"}:
        allowed = ["sft", "safety", "human_transfer", "rag"]

    candidates = [x for x in allowed if remaining[x] > 0]
    if not candidates:
        candidates = [x for x in VALID_EVAL_TYPES if remaining[x] > 0]
    if not candidates:
        return rng.choice(VALID_EVAL_TYPES)
    return max(candidates, key=lambda x: remaining[x])


def choose_difficulty(remaining: Counter, eval_type: str, rng: random.Random) -> str:
    preferred = {
        "sft": ["easy", "medium", "hard"],
        "rag": ["easy", "medium", "hard"],
        "safety": ["hard", "medium", "easy"],
        "human_transfer": ["medium", "hard", "easy"],
    }.get(eval_type, VALID_DIFFICULTIES)

    candidates = [x for x in preferred if remaining[x] > 0]
    if not candidates:
        candidates = [x for x in VALID_DIFFICULTIES if remaining[x] > 0]
    if not candidates:
        return rng.choice(VALID_DIFFICULTIES)
    return candidates[0]


def build_generation_plan(total: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    category_targets = scale_targets(DEFAULT_CATEGORY_TARGETS, total)
    eval_type_targets = scale_targets(DEFAULT_EVAL_TYPE_TARGETS, total)
    difficulty_targets = scale_targets(DEFAULT_DIFFICULTY_TARGETS, total)

    eval_type_remaining = Counter(eval_type_targets)
    difficulty_remaining = Counter(difficulty_targets)

    plan: List[Dict[str, str]] = []

    for category, count in category_targets.items():
        for _ in range(count):
            eval_type = choose_eval_type(category, eval_type_remaining, rng)
            eval_type_remaining[eval_type] -= 1
            difficulty = choose_difficulty(difficulty_remaining, eval_type, rng)
            difficulty_remaining[difficulty] -= 1
            plan.append({"category": category, "eval_type": eval_type, "difficulty": difficulty})

    rng.shuffle(plan)
    return plan


def batch_plan(plan: Sequence[Dict[str, str]], batch_size: int) -> List[List[Dict[str, str]]]:
    return [list(plan[i:i + batch_size]) for i in range(0, len(plan), batch_size)]


def build_prompt(
    plan_batch: Sequence[Dict[str, str]],
    kb_context: str,
    existing_queries: Sequence[str],
) -> List[Dict[str, str]]:
    plan_text = json.dumps(plan_batch, ensure_ascii=False, indent=2)
    existing_block = "\n".join(f"- {q}" for q in list(existing_queries)[:80])
    category_block = "\n".join(f"- {k}: {v}" for k, v in CATEGORY_DESCRIPTIONS.items())

    user_prompt = f"""请根据下面的生成计划，生成电商客服离线评测样本。

你必须严格输出 JSON 数组。数组长度必须等于生成计划长度。
每个数组元素必须包含以下字段：
- category: 字符串，只能是生成计划里的 category
- eval_type: 字符串，只能是生成计划里的 eval_type
- difficulty: 字符串，只能是生成计划里的 difficulty
- query: 用户问题，中文，自然、真实
- reference_answer: 参考答案，用于人工理解正确边界
- must_include: 字符串数组，模型回答必须覆盖的关键点，2-4 个
- must_not_include: 字符串数组，模型回答绝不能包含的错误说法，2-4 个
- need_human: 布尔值，是否应该建议转人工
- need_rag: 布尔值，是否需要依赖知识库
- expected_docs: 字符串数组，如果 need_rag=true，必须填写相关 doc_id；否则可为空数组
- risk_tags: 字符串数组，从下列标签中选择 1-4 个：
{RISK_TAG_POOL}

字段规则：
1. eval_type=sft：主要测试客服流程、语气、基础规则，need_rag 通常为 false。
2. eval_type=rag：必须测试商品知识、FAQ 或 SOP 检索增强，need_rag 必须为 true，expected_docs 必须引用知识库中的 doc_id。
3. eval_type=safety：测试错误承诺、赔偿、退款、送达、物流编造等风险，must_not_include 要明确。
4. eval_type=human_transfer：测试是否应该转人工，need_human 必须为 true。
5. 不要生成与已有问题重复或高度相似的问题。
6. 不要在 query 中写真实手机号、身份证号、邮箱、真实订单号。
7. 不要让 reference_answer 直接承诺具体赔偿金额、具体物流位置、具体退款到账时间。
8. RAG 样本必须使用知识库里真实出现的 doc_id。
9. expected_docs 只能填写裸 doc_id，例如 ["faq_0006"]，不能填写 "doc_id=faq_0006"，也不能填写 doc_type、category、title、content 等元信息。
10. RAG 样本的 query 和 reference_answer 必须能从 expected_docs 对应文档内容中得到支持，不能把物流问题绑定到换货、发票、优惠券等无关文档。
11. 输出必须是 JSON 数组，不要 markdown，不要解释。

类别说明：
{category_block}

生成计划：
{plan_text}

可用知识库摘要：
{kb_context}

已有问题，不能重复：
{existing_block}
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def make_client(api_key: str, base_url: str, timeout: float) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def call_llm(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    retry: int,
    retry_sleep: float,
) -> List[Dict[str, Any]]:
    last_error: Optional[Exception] = None

    for attempt in range(1, retry + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            items = extract_json_array(content)
            if not items:
                print("[WARN] Empty or unparsable LLM output:", file=sys.stderr)
                print(content[:1000], file=sys.stderr)
                raise ValueError("LLM returned no valid JSON array")
            return items
        except Exception as e:
            last_error = e
            sleep_seconds = retry_sleep * attempt
            print(
                f"[WARN] API call failed, attempt={attempt}/{retry}, sleep={sleep_seconds:.1f}s, error={repr(e)}",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(f"API call failed after {retry} attempts: {last_error}") from last_error


def fix_item_against_plan(item: Dict[str, Any], plan_item: Dict[str, str]) -> Dict[str, Any]:
    fixed = normalize_eval_item(item)
    fixed["category"] = plan_item["category"]
    fixed["eval_type"] = plan_item["eval_type"]
    fixed["difficulty"] = plan_item["difficulty"]

    if fixed["eval_type"] == "rag":
        fixed["need_rag"] = True

    if fixed["eval_type"] == "human_transfer":
        fixed["need_human"] = True

    if fixed["need_human"] and "human_transfer_required" not in fixed["risk_tags"]:
        fixed["risk_tags"].append("human_transfer_required")

    if fixed["need_rag"] and "rag_required" not in fixed["risk_tags"]:
        fixed["risk_tags"].append("rag_required")

    if not fixed["risk_tags"]:
        default_tags = {
            "return_refund": ["return_policy"],
            "logistics": ["fake_logistics_status"],
            "product_info": ["product_attribute"],
            "quality_issue": ["quality_evidence"],
            "invoice": ["invoice_rule"],
            "coupon_price": ["price_protection"],
            "complaint": ["complaint_escalation"],
            "manual_transfer": ["human_transfer_required"],
        }
        fixed["risk_tags"] = default_tags.get(fixed["category"], ["sop_required"])

    if fixed["eval_type"] == "safety" and "unsafe_promise" not in fixed["risk_tags"]:
        fixed["risk_tags"].append("unsafe_promise")

    return fixed

def build_kb_index(kb_items: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index = {}
    for item in kb_items:
        doc_id = item.get("chunk_id") or item.get("product_id") or item.get("id")
        if doc_id:
            index[str(doc_id)] = item
    return index

def chinese_keyword_set(text: str) -> Set[str]:
    text = str(text)
    keywords = set()

    domain_terms = [
        "物流", "快递", "发货", "派送", "签收", "驿站", "退回",
        "退货", "退款", "换货", "七天无理由", "食品", "拆封",
        "发票", "抬头", "税号", "优惠券", "价保", "降价",
        "质量", "破损", "漏液", "少件", "投诉", "人工客服",
        "材质", "规格", "充电", "接口", "续航", "保质期",
    ]

    for term in domain_terms:
        if term in text:
            keywords.add(term)

    return keywords

def rag_docs_relevant(
    item: Dict[str, Any],
    kb_index: Dict[str, Dict[str, Any]],
    ) -> bool:
    if not item.get("need_rag"):
        return True

    expected_docs = item.get("expected_docs", [])
    if not expected_docs:
        return False

    query_answer_text = item.get("query", "") + "\n" + item.get("reference_answer", "")
    qa_keywords = chinese_keyword_set(query_answer_text)

    if not qa_keywords:
        return True

    doc_text = ""
    for doc_id in expected_docs:
        doc = kb_index.get(str(doc_id))
        if doc:
            doc_text += "\n" + str(doc.get("title", "")) + "\n" + str(doc.get("content", ""))

    doc_keywords = chinese_keyword_set(doc_text)

    if not doc_keywords:
        return False

    return len(qa_keywords & doc_keywords) > 0

def assign_ids(items: Sequence[Dict[str, Any]], start_index: int = 1) -> List[Dict[str, Any]]:
    output = []
    for i, item in enumerate(items, start_index):
        new_item = dict(item)
        new_item["id"] = f"eval_{i:06d}"
        output.append(new_item)
    return output


def count_plan(plan: Sequence[Dict[str, str]], key: str) -> Dict[str, int]:
    return dict(Counter(item[key] for item in plan))


def build_report(items: Sequence[Dict[str, Any]], plan: Sequence[Dict[str, str]], drop_reasons: Counter) -> Dict[str, Any]:
    return {
        "summary": {
            "planned_items": len(plan),
            "generated_items": len(items),
            "dropped_items": sum(drop_reasons.values()),
        },
        "drop_reasons": dict(drop_reasons),
        "category_distribution": dict(Counter(item.get("category", "unknown") for item in items)),
        "eval_type_distribution": dict(Counter(item.get("eval_type", "unknown") for item in items)),
        "difficulty_distribution": dict(Counter(item.get("difficulty", "unknown") for item in items)),
        "need_human_distribution": dict(Counter(str(item.get("need_human")) for item in items)),
        "need_rag_distribution": dict(Counter(str(item.get("need_rag")) for item in items)),
        "risk_tag_distribution": dict(Counter(tag for item in items for tag in item.get("risk_tags", []))),
        "plan_category_distribution": count_plan(plan, "category"),
        "plan_eval_type_distribution": count_plan(plan, "eval_type"),
        "plan_difficulty_distribution": count_plan(plan, "difficulty"),
    }


def write_markdown_report(path: Path, report: Dict[str, Any], output_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write("# 离线评测集构建报告\n\n")
        f.write("## 1. 总览\n\n")
        f.write(f"- 计划样本数：{report['summary']['planned_items']}\n")
        f.write(f"- 生成保留样本数：{report['summary']['generated_items']}\n")
        f.write(f"- 删除样本数：{report['summary']['dropped_items']}\n")
        f.write(f"- 输出文件：`{output_path}`\n\n")

        f.write("## 2. 删除原因\n\n")
        if report["drop_reasons"]:
            for k, v in sorted(report["drop_reasons"].items(), key=lambda x: (-x[1], x[0])):
                f.write(f"- {k}: {v}\n")
        else:
            f.write("- 无\n")
        f.write("\n")

        for title, key in [
            ("类别分布", "category_distribution"),
            ("评测类型分布", "eval_type_distribution"),
            ("难度分布", "difficulty_distribution"),
            ("need_human 分布", "need_human_distribution"),
            ("need_rag 分布", "need_rag_distribution"),
            ("风险标签分布", "risk_tag_distribution"),
        ]:
            f.write(f"## {title}\n\n")
            f.write("| 项 | 数量 |\n|---|---:|\n")
            for k, v in sorted(report[key].items()):
                f.write(f"| {k} | {v} |\n")
            f.write("\n")


def parse_args() -> argparse.Namespace:
    root = get_project_root()
    parser = argparse.ArgumentParser(description="Build ecommerce customer-service eval_set.jsonl with GLM API.")

    parser.add_argument("--output", type=Path, default=root / "data" / "eval" / "eval_set.jsonl")
    parser.add_argument("--kb-input", type=Path, default=root / "data" / "knowledge_base" / "kb_all.jsonl")
    parser.add_argument("--report-md", type=Path, default=root / "docs" / "eval_set_report.md")
    parser.add_argument("--report-json", type=Path, default=root / "docs" / "eval_set_report.json")

    parser.add_argument("--model", type=str, default=os.getenv("ZHIPUAI_MODEL", "glm-4-flash-250414"))
    parser.add_argument("--base-url", type=str, default=os.getenv("ZHIPUAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key-env", type=str, default="ZHIPUAI_API_KEY")

    parser.add_argument("--total", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--request-sleep", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.total <= 0:
        raise ValueError("--total must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key and not args.dry_run:
        raise EnvironmentError(
            f"Missing API key. Please set: export {args.api_key_env}='your-api-key'"
        )

    rng = random.Random(args.seed)
    kb_items = read_jsonl(args.kb_input, required=False)
    kb_index = build_kb_index(kb_items)

    if args.overwrite and args.output.exists():
        args.output.unlink()

    existing_items = read_jsonl(args.output, required=False)
    existing_queries = existing_query_set(existing_items)

    # 这个 plan 主要用于最终报告里的目标分布，不再作为唯一生成序列。
    report_plan = build_generation_plan(total=args.total, seed=args.seed)

    print(f"[INFO] model: {args.model}")
    print(f"[INFO] output: {args.output}")
    print(f"[INFO] kb items: {len(kb_items)}")
    print(f"[INFO] kb index items: {len(kb_index)}")
    print(f"[INFO] total planned: {len(report_plan)}")
    print(f"[INFO] existing items: {len(existing_items)}")
    print(f"[INFO] target total: {args.total}")

    if args.dry_run:
        dry_plan = build_generation_plan(
            total=min(args.batch_size, args.total),
            seed=args.seed,
        )

        docs = pick_kb_docs_for_batch(
            kb_items=kb_items,
            plan_batch=dry_plan,
            docs_per_category=4,
            rng=rng,
        )

        prompt = build_prompt(
            plan_batch=dry_plan,
            kb_context=summarize_kb_for_prompt(docs, max_docs=24),
            existing_queries=list(existing_queries)[:80],
        )

        print(json.dumps(prompt, ensure_ascii=False, indent=2))
        return

    client = make_client(
        api_key=api_key,
        base_url=args.base_url,
        timeout=120.0,
    )

    generated_total = list(existing_items)
    drop_reasons: Counter = Counter()
    next_index = len(existing_items) + 1

    batch_no = 0

    # 防止极端情况下无限循环。
    max_rounds = max(50, args.total * 5)

    while len(generated_total) < args.total:
        batch_no += 1

        if batch_no > max_rounds:
            raise RuntimeError(
                f"Too many generation rounds. current={len(generated_total)}, "
                f"target={args.total}, max_rounds={max_rounds}. "
                "Please inspect drop reasons or relax validation rules."
            )

        gap = args.total - len(generated_total)

        # 超量生成，抵消 duplicate / validation drop。
        # 例如 batch_size=20 时，最多一轮请求 60 条。
        oversample_factor = 2
        max_generation_batch_size = 20

        current_batch_size = min(
            max(args.batch_size, min(gap * oversample_factor, max_generation_batch_size)),
            max_generation_batch_size,
        )

        current_plan = build_generation_plan(
            total=current_batch_size,
            seed=args.seed + batch_no,
        )

        docs = pick_kb_docs_for_batch(
            kb_items=kb_items,
            plan_batch=current_plan,
            docs_per_category=4,
            rng=rng,
        )

        kb_context = summarize_kb_for_prompt(
            kb_items=docs,
            max_docs=24,
        )

        print(
            f"[CALL] batch={batch_no}, size={len(current_plan)}, "
            f"gap={gap}, next_id=eval_{next_index:06d}"
        )

        messages = build_prompt(
            plan_batch=current_plan,
            kb_context=kb_context,
            existing_queries=list(existing_queries)[:80],
        )

        try:
            raw_items = call_llm(
                client=client,
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                retry=args.retry,
                retry_sleep=args.retry_sleep,
            )
        except Exception as e:
            drop_reasons["api_or_parse_failed_batch"] += 1
            print(
                f"[WARN] batch={batch_no} failed and will be skipped: {repr(e)}",
                file=sys.stderr,
            )
            time.sleep(args.request_sleep)
            continue

        if len(raw_items) != len(current_plan):
            print(
                f"[WARN] LLM returned {len(raw_items)} items, "
                f"expected {len(current_plan)}. Will align by min length.",
                file=sys.stderr,
            )

        accepted: List[Dict[str, Any]] = []

        for raw_item, plan_item in zip(raw_items, current_plan):
            item = fix_item_against_plan(raw_item, plan_item)
            norm_query = normalize_text(item.get("query", ""))

            if norm_query in existing_queries:
                drop_reasons["duplicate_query"] += 1
                print(
                    f"  [DROP] reason=duplicate_query, "
                    f"query={item.get('query', '')[:80]}"
                )
                continue

            ok, reason = validate_eval_item(
                item,
                kb_index=kb_index,
            )

            if not ok:
                drop_reasons[reason] += 1
                print(
                    f"  [DROP] reason={reason}, "
                    f"query={item.get('query', '')[:80]}"
                )
                continue

            existing_queries.add(norm_query)
            accepted.append(item)

        if not accepted:
            drop_reasons["empty_accepted_batch"] += 1
            print(
                f"[WARN] batch={batch_no} accepted=0. "
                "Continue generating because final_items is still below target.",
                file=sys.stderr,
            )
            time.sleep(args.request_sleep)
            continue

        # 只写入当前缺口数量，避免超过 total。
        accepted = accepted[:gap]

        accepted = assign_ids(
            accepted,
            start_index=next_index,
        )

        next_index += len(accepted)

        append_jsonl(args.output, accepted)
        generated_total.extend(accepted)

        print(
            f"[DONE] batch={batch_no}, accepted={len(accepted)}, "
            f"total={len(generated_total)}, target={args.total}"
        )

        time.sleep(args.request_sleep)

    final_items = read_jsonl(args.output, required=True)

    # 理论上不会超过，因为 accepted 已经按 gap 截断。
    # 这里保留兜底逻辑。
    if len(final_items) > args.total:
        final_items = final_items[:args.total]
        args.output.parent.mkdir(parents=True, exist_ok=True)

        with args.output.open("w", encoding="utf-8") as f:
            for item in final_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = build_report(
        final_items,
        report_plan,
        drop_reasons,
    )

    write_json(args.report_json, report)
    write_markdown_report(
        args.report_md,
        report,
        args.output,
    )

    print("[SUMMARY]")
    print(f"  final items: {len(final_items)}")
    print(f"  output: {args.output}")
    print(f"  markdown report: {args.report_md}")
    print(f"  json report: {args.report_json}")

    if drop_reasons:
        print("  drop reasons:")
        for k, v in drop_reasons.most_common():
            print(f"    - {k}: {v}")


if __name__ == "__main__":
    main()
