#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build DPO preference data for the ecommerce customer-service LLM project.

Inputs:
  data/interim/cleaned_sft.jsonl
  data/eval/eval_set.jsonl

Outputs:
  data/interim/dpo_candidates.jsonl
  data/processed/dpo_train.jsonl
  data/processed/dpo_valid.jsonl
  docs/dpo_data_report.md
  docs/dpo_data_report.json

Core idea:
  - prompt/chosen come from cleaned SFT samples and eval samples.
  - GLM API only generates rejected + preference_reason.
  - rejected_type is assigned by this script to cover known bad-answer modes.

Usage:
  pip install openai
  export ZHIPUAI_API_KEY="your-api-key"
  python scripts/build_dpo_data_with_llm.py --dry-run
  python scripts/build_dpo_data_with_llm.py --model glm-4-flash-250414 --target-total 300 --overwrite
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from openai import OpenAI


DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

VALID_CATEGORIES = [
    "return_refund", "logistics", "product_info", "quality_issue",
    "invoice", "coupon_price", "complaint", "manual_transfer",
]

REJECTED_TYPES = [
    "fabricated_policy",
    "compensation_commitment",
    "wrong_intent",
    "missing_action",
    "rude_tone",
    "fake_order_or_logistics_status",
    "over_marketing",
    "no_human_transfer",
    "unsafe_overpromise",
]

SEVERITY_BY_REJECTED_TYPE = {
    "fabricated_policy": "high",
    "compensation_commitment": "high",
    "wrong_intent": "medium",
    "missing_action": "medium",
    "rude_tone": "medium",
    "fake_order_or_logistics_status": "high",
    "over_marketing": "low",
    "no_human_transfer": "high",
    "unsafe_overpromise": "high",
}

DEFAULT_CATEGORY_TARGETS = {
    "return_refund": 60,
    "logistics": 45,
    "product_info": 40,
    "quality_issue": 45,
    "invoice": 20,
    "coupon_price": 25,
    "complaint": 35,
    "manual_transfer": 30,
}

DEFAULT_REJECTED_TYPE_TARGETS = {
    "fabricated_policy": 45,
    "compensation_commitment": 40,
    "wrong_intent": 35,
    "missing_action": 35,
    "rude_tone": 30,
    "fake_order_or_logistics_status": 40,
    "over_marketing": 20,
    "no_human_transfer": 35,
    "unsafe_overpromise": 20,
}

SYSTEM_PROMPT = (
    "你是电商客服 DPO 偏好数据构造助手。"
    "你的任务是基于给定 prompt 和 chosen，按照指定错误类型生成 rejected。"
    "rejected 必须比 chosen 明显更差，但仍要像真实客服模型可能犯的错误。"
    "输出必须是 JSON 对象。"
)

DEFAULT_SFT_SYSTEM_PROMPT = (
    "你是电商平台客服助手，需要基于售后政策回答用户问题。"
    "回答应礼貌、准确、可执行。遇到订单状态、赔付金额、投诉升级等无法确认的问题时，需要建议转人工。"
)

PHONE_PATTERNS = [r"(?<!\d)1[3-9]\d{9}(?!\d)"]
ID_CARD_PATTERNS = [r"\b\d{17}[\dXx]\b", r"\b\d{15}\b"]
EMAIL_PATTERNS = [r"[\w\.-]+@[\w\.-]+\.\w+"]
ORDER_ID_PATTERNS = [
    r"订单号\s*[:：]?\s*[A-Za-z0-9_-]{5,}",
    r"订单\s*[:：]?\s*[A-Za-z0-9_-]{8,}",
    r"单号\s*[:：]?\s*[A-Za-z0-9_-]{8,}",
]

RISKY_CHOSEN_PATTERNS = [
    r"(一定|肯定|保证|百分百|必须).{0,10}(退款|退货|送达|赔偿|赔付|解决|处理)",
    r"(明天|今天|后天).{0,8}(一定|肯定|保证).{0,8}(到|送达|退款|处理)",
    r"(三倍|十倍|假一赔十|全额).{0,8}(赔偿|赔付|退款)",
    r"(无条件退款|无条件退货|包退|包换|包赔)",
    r"(赔|赔偿|赔付|补偿)\s*\d+(?:\.\d+)?\s*(元|块钱|人民币|rmb|RMB)",
]

REJECTED_TYPE_MARKERS = {
    "fabricated_policy": [r"所有商品", r"一律", r"全部.*能退", r"都可以退", r"15\s*天", r"无条件退", r"拆封.*也.*能退"],
    "compensation_commitment": [r"赔", r"赔偿", r"赔付", r"补偿", r"\d+\s*(元|块钱|人民币)", r"三倍", r"十倍"],
    "wrong_intent": [r"退货", r"退款", r"物流", r"发票", r"优惠券", r"重新下单", r"再买"],
    "missing_action": [r"可以处理", r"会处理", r"等着", r"稍后", r"没问题", r"可以的", r"不清楚"],
    "rude_tone": [r"不行", r"不能", r"规则就是这样", r"自己看", r"别问", r"没办法", r"不归我管"],
    "fake_order_or_logistics_status": [r"已经发货", r"已经签收", r"已经到", r"正在派送", r"明天到", r"今天到", r"包裹在"],
    "over_marketing": [r"再买", r"推荐您购买", r"加购", r"下单", r"买一个", r"活动很划算"],
    "no_human_transfer": [r"直接", r"马上", r"一定", r"给您处理", r"不用人工", r"系统会自动"],
    "unsafe_overpromise": [r"保证", r"一定", r"百分百", r"必须", r"肯定", r"马上解决", r"一定满意"],
}


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


def write_jsonl(path: Path, items: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


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
    text = text.replace("：", ":").replace("；", ";")
    return text


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def contains_any_pattern(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def get_messages(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = item.get("messages", [])
    return messages if isinstance(messages, list) else []


def extract_text_by_role(item: Dict[str, Any], role: str) -> str:
    parts: List[str] = []
    for msg in get_messages(item):
        if isinstance(msg, dict) and msg.get("role") == role:
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
    return "\n".join(parts).strip()


def extract_sft_texts(item: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        extract_text_by_role(item, "system"),
        extract_text_by_role(item, "user"),
        extract_text_by_role(item, "assistant"),
    )


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


def chosen_has_risky_commitment(chosen_text: str) -> bool:
    return contains_any_pattern(chosen_text, RISKY_CHOSEN_PATTERNS)


def rejected_type_roughly_matches(rejected_text: str, rejected_type: str) -> bool:
    patterns = REJECTED_TYPE_MARKERS.get(rejected_type, [])
    if not patterns:
        return True
    return contains_any_pattern(rejected_text, patterns)


def build_prompt(system: str, user: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system or DEFAULT_SFT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def assistant_msg(text: str) -> List[Dict[str, str]]:
    return [{"role": "assistant", "content": text}]


def extract_sources_from_sft(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for item in items:
        category = item.get("category")
        if category not in VALID_CATEGORIES:
            continue
        system, user, assistant = extract_sft_texts(item)
        if not user or not assistant:
            continue
        if chosen_has_risky_commitment(assistant):
            continue
        sources.append({
            "source": "from_sft",
            "source_id": item.get("id"),
            "category": category,
            "prompt": build_prompt(system, user),
            "chosen": assistant_msg(assistant),
            "metadata": {"seed_id": item.get("seed_id"), "original_source": item.get("source")},
        })
    return sources


def extract_sources_from_eval(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for item in items:
        category = item.get("category")
        if category not in VALID_CATEGORIES:
            continue
        query = str(item.get("query", "")).strip()
        reference_answer = str(item.get("reference_answer", "")).strip()
        if not query or not reference_answer:
            continue
        if chosen_has_risky_commitment(reference_answer):
            continue
        sources.append({
            "source": "from_eval",
            "source_id": item.get("id"),
            "category": category,
            "prompt": build_prompt(DEFAULT_SFT_SYSTEM_PROMPT, query),
            "chosen": assistant_msg(reference_answer),
            "metadata": {
                "eval_type": item.get("eval_type"),
                "difficulty": item.get("difficulty"),
                "risk_tags": item.get("risk_tags", []),
                "need_human": item.get("need_human"),
                "need_rag": item.get("need_rag"),
                "expected_docs": item.get("expected_docs", []),
            },
        })
    return sources


def scale_targets(base: Dict[str, int], total: int) -> Dict[str, int]:
    base_sum = sum(base.values())
    raw = {k: v * total / base_sum for k, v in base.items()}
    scaled = {k: int(v) for k, v in raw.items()}
    remaining = total - sum(scaled.values())
    order = sorted(raw.keys(), key=lambda k: raw[k] - int(raw[k]), reverse=True)
    for k in order[:remaining]:
        scaled[k] += 1
    return scaled


def choose_rejected_type(category: str, remaining: Counter, rng: random.Random) -> str:
    allowed_by_category = {
        "return_refund": ["fabricated_policy", "missing_action", "rude_tone", "unsafe_overpromise", "wrong_intent"],
        "logistics": ["fake_order_or_logistics_status", "no_human_transfer", "unsafe_overpromise", "wrong_intent", "missing_action"],
        "product_info": ["wrong_intent", "over_marketing", "missing_action", "fabricated_policy", "unsafe_overpromise"],
        "quality_issue": ["missing_action", "rude_tone", "wrong_intent", "unsafe_overpromise", "fabricated_policy"],
        "invoice": ["fabricated_policy", "missing_action", "wrong_intent", "rude_tone", "unsafe_overpromise"],
        "coupon_price": ["compensation_commitment", "fabricated_policy", "missing_action", "unsafe_overpromise", "wrong_intent"],
        "complaint": ["compensation_commitment", "rude_tone", "no_human_transfer", "unsafe_overpromise", "wrong_intent"],
        "manual_transfer": ["no_human_transfer", "compensation_commitment", "fake_order_or_logistics_status", "unsafe_overpromise", "missing_action"],
    }
    allowed = allowed_by_category.get(category, REJECTED_TYPES)
    candidates = [x for x in allowed if remaining[x] > 0]
    if not candidates:
        candidates = [x for x in REJECTED_TYPES if remaining[x] > 0]
    if not candidates:
        return rng.choice(REJECTED_TYPES)
    return max(candidates, key=lambda x: remaining[x])


def build_generation_plan(target_total: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    category_targets = scale_targets(DEFAULT_CATEGORY_TARGETS, target_total)
    type_targets = scale_targets(DEFAULT_REJECTED_TYPE_TARGETS, target_total)
    type_remaining = Counter(type_targets)
    plan: List[Dict[str, str]] = []
    for category, count in category_targets.items():
        for _ in range(count):
            rejected_type = choose_rejected_type(category, type_remaining, rng)
            type_remaining[rejected_type] -= 1
            plan.append({
                "category": category,
                "rejected_type": rejected_type,
                "severity": SEVERITY_BY_REJECTED_TYPE.get(rejected_type, "medium"),
            })
    rng.shuffle(plan)
    return plan


def sample_sources_for_plan(sources: Sequence[Dict[str, Any]], plan: Sequence[Dict[str, str]], seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for source in sources:
        by_category[source["category"]].append(source)
    for category in by_category:
        rng.shuffle(by_category[category])
    cursors = Counter()
    paired = []
    all_sources = list(sources)
    for plan_item in plan:
        category = plan_item["category"]
        pool = by_category.get(category, [])
        if pool:
            source = pool[cursors[category] % len(pool)]
            cursors[category] += 1
        else:
            source = rng.choice(all_sources)
        paired.append({**plan_item, "source_item": source})
    return paired


def rejected_type_instruction(rejected_type: str) -> str:
    return {
        "fabricated_policy": "编造不存在或过度泛化的售后政策，例如把特殊类目说成一定支持退货，或说所有商品都能退。",
        "compensation_commitment": "错误承诺具体赔偿、补偿或退款金额。",
        "wrong_intent": "答非所问，回答到另一个客服意图上，但仍像客服回答。",
        "missing_action": "回复空泛，没有可执行步骤，用户看完不知道下一步怎么办。",
        "rude_tone": "语气生硬、不耐烦、缺少安抚，明显不符合客服风格。",
        "fake_order_or_logistics_status": "编造具体订单、物流、库存、退款进度或快递位置。",
        "over_marketing": "不解决问题，转而推荐用户再购买、加购或参加活动。",
        "no_human_transfer": "遇到具体订单状态、赔付金额、投诉升级等应转人工场景，却没有转人工并直接处理或承诺。",
        "unsafe_overpromise": "做出绝对化承诺，例如保证解决、一定退款、一定送达、百分百处理。",
    }.get(rejected_type, rejected_type)


def build_generation_prompt(item: Dict[str, Any]) -> List[Dict[str, str]]:
    source = item["source_item"]
    prompt = source["prompt"]
    chosen_text = source["chosen"][0]["content"]
    rejected_type = item["rejected_type"]
    user_prompt = f"""请为下面这条电商客服偏好样本生成 rejected 回复。

任务：
给定 prompt 和 chosen，你需要生成一个 rejected，使它符合指定 rejected_type，且明显比 chosen 更差。

要求：
1. 只生成 rejected 和 preference_reason；
2. 不要改写 prompt；
3. 不要改写 chosen；
4. rejected 必须像真实客服模型可能犯的错误，不要写成恶搞、违法、辱骂或极端内容；
5. rejected 应与用户问题相关，但必须体现指定错误类型；
6. rejected 长度建议 10-120 个中文字符；
7. preference_reason 要说明 chosen 为什么更好、rejected 为什么更差；
8. 输出必须是 JSON 对象，不要 markdown，不要解释。

指定 category：
{item['category']}

指定 rejected_type：
{rejected_type}

rejected_type 说明：
{rejected_type_instruction(rejected_type)}

severity：
{item['severity']}

prompt：
{json.dumps(prompt, ensure_ascii=False, indent=2)}

chosen：
{chosen_text}

输出格式：
{{
  "rejected": "这里写错误客服回复",
  "preference_reason": "这里写偏好原因"
}}
"""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def make_client(api_key: str, base_url: str, timeout: float) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def call_llm(client: OpenAI, model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int, retry: int, retry_sleep: float) -> Dict[str, Any]:
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
            obj = extract_json_object(content)
            if not obj:
                print("[WARN] Empty or unparsable LLM output:", file=sys.stderr)
                print(content[:800], file=sys.stderr)
                raise ValueError("LLM returned no JSON object")
            return obj
        except Exception as e:
            last_error = e
            sleep_seconds = retry_sleep * attempt
            print(f"[WARN] API call failed, attempt={attempt}/{retry}, sleep={sleep_seconds:.1f}s, error={repr(e)}", file=sys.stderr)
            time.sleep(sleep_seconds)
    raise RuntimeError(f"API call failed after {retry} attempts: {last_error}") from last_error


def build_dpo_item(item: Dict[str, Any], llm_obj: Dict[str, Any], dpo_id: str) -> Dict[str, Any]:
    source = item["source_item"]
    rejected_text = str(llm_obj.get("rejected", "")).strip()
    preference_reason = str(llm_obj.get("preference_reason", "")).strip()
    return {
        "id": dpo_id,
        "category": item["category"],
        "source": source["source"],
        "source_id": source["source_id"],
        "rejected_type": item["rejected_type"],
        "severity": item["severity"],
        "prompt": source["prompt"],
        "chosen": source["chosen"],
        "rejected": [{"role": "assistant", "content": rejected_text}],
        "preference_reason": preference_reason,
        "metadata": source.get("metadata", {}),
    }


def validate_dpo_item(item: Dict[str, Any], similarity_threshold: float, enforce_rejected_type_match: bool) -> Tuple[bool, str]:
    required = ["id", "category", "source", "rejected_type", "severity", "prompt", "chosen", "rejected", "preference_reason"]
    for field in required:
        if field not in item:
            return False, f"missing_{field}"
    if item["category"] not in VALID_CATEGORIES:
        return False, "invalid_category"
    if item["rejected_type"] not in REJECTED_TYPES:
        return False, "invalid_rejected_type"
    prompt = item.get("prompt", [])
    chosen = item.get("chosen", [])
    rejected = item.get("rejected", [])
    if not isinstance(prompt, list) or len(prompt) < 2:
        return False, "bad_prompt"
    if not isinstance(chosen, list) or len(chosen) != 1:
        return False, "bad_chosen"
    if not isinstance(rejected, list) or len(rejected) != 1:
        return False, "bad_rejected"
    prompt_roles = [msg.get("role") for msg in prompt if isinstance(msg, dict)]
    if "system" not in prompt_roles or "user" not in prompt_roles:
        return False, "prompt_missing_system_or_user"
    if chosen[0].get("role") != "assistant":
        return False, "chosen_not_assistant"
    if rejected[0].get("role") != "assistant":
        return False, "rejected_not_assistant"
    chosen_text = str(chosen[0].get("content", "")).strip()
    rejected_text = str(rejected[0].get("content", "")).strip()
    if len(chosen_text) < 10:
        return False, "chosen_too_short"
    if len(rejected_text) < 4:
        return False, "rejected_too_short"
    if len(rejected_text) > 180:
        return False, "rejected_too_long"
    sensitive, reason = contains_sensitive_info(json.dumps(item, ensure_ascii=False))
    if sensitive:
        return False, f"sensitive_{reason}"
    if chosen_has_risky_commitment(chosen_text):
        return False, "chosen_has_risky_commitment"
    if similarity(chosen_text, rejected_text) >= similarity_threshold:
        return False, "chosen_rejected_too_similar"
    if item["rejected_type"] != "wrong_intent" and not rejected_type_roughly_matches(rejected_text, item["rejected_type"]):
        if enforce_rejected_type_match:
            return False, "rejected_type_not_matched"
    if len(str(item.get("preference_reason", "")).strip()) < 8:
        return False, "preference_reason_too_short"
    return True, "ok"


def dpo_key(item: Dict[str, Any]) -> str:
    prompt_user = ""
    for msg in item.get("prompt", []):
        if msg.get("role") == "user":
            prompt_user = msg.get("content", "")
            break
    rejected_text = item.get("rejected", [{}])[0].get("content", "")
    return normalize_text(prompt_user) + "||" + item.get("rejected_type", "") + "||" + normalize_text(rejected_text)


def stratified_split(items: Sequence[Dict[str, Any]], valid_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not (0 < valid_ratio < 1):
        raise ValueError("--valid-ratio must be in (0, 1).")
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        key = f"{item.get('category')}||{item.get('rejected_type')}"
        buckets[key].append(item)
    train: List[Dict[str, Any]] = []
    valid: List[Dict[str, Any]] = []
    for bucket_items in buckets.values():
        bucket_items = list(bucket_items)
        rng.shuffle(bucket_items)
        if len(bucket_items) == 1:
            train.extend(bucket_items)
            continue
        n_valid = max(1, int(round(len(bucket_items) * valid_ratio)))
        n_valid = min(n_valid, len(bucket_items) - 1)
        valid.extend(bucket_items[:n_valid])
        train.extend(bucket_items[n_valid:])
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def count_by(items: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    return dict(Counter(str(item.get(key, "unknown")) for item in items))


def build_report(candidates: Sequence[Dict[str, Any]], train: Sequence[Dict[str, Any]], valid: Sequence[Dict[str, Any]], drop_reasons: Counter, plan: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "summary": {
            "planned_items": len(plan),
            "candidate_items": len(candidates),
            "train_items": len(train),
            "valid_items": len(valid),
            "actual_valid_ratio": round(len(valid) / max(len(train) + len(valid), 1), 4),
            "dropped_items": sum(drop_reasons.values()),
        },
        "drop_reasons": dict(drop_reasons),
        "category_distribution": {"all": count_by(candidates, "category"), "train": count_by(train, "category"), "valid": count_by(valid, "category")},
        "rejected_type_distribution": {"all": count_by(candidates, "rejected_type"), "train": count_by(train, "rejected_type"), "valid": count_by(valid, "rejected_type")},
        "source_distribution": {"all": count_by(candidates, "source"), "train": count_by(train, "source"), "valid": count_by(valid, "source")},
        "severity_distribution": {"all": count_by(candidates, "severity"), "train": count_by(train, "severity"), "valid": count_by(valid, "severity")},
    }


def write_distribution_table(f, title: str, dist: Dict[str, Dict[str, int]]) -> None:
    f.write(f"## {title}\n\n")
    keys = sorted(set(dist.get("all", {})) | set(dist.get("train", {})) | set(dist.get("valid", {})))
    f.write("| 项 | All | Train | Valid |\n|---|---:|---:|---:|\n")
    for key in keys:
        f.write(f"| {key} | {dist.get('all', {}).get(key, 0)} | {dist.get('train', {}).get(key, 0)} | {dist.get('valid', {}).get(key, 0)} |\n")
    f.write("\n")


def write_markdown_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]
    with path.open("w", encoding="utf-8") as f:
        f.write("# DPO 数据构建报告\n\n")
        f.write("## 1. 总览\n\n")
        f.write(f"- 计划样本数：{summary['planned_items']}\n")
        f.write(f"- 候选样本数：{summary['candidate_items']}\n")
        f.write(f"- 训练集样本数：{summary['train_items']}\n")
        f.write(f"- 验证集样本数：{summary['valid_items']}\n")
        f.write(f"- 实际验证集比例：{summary['actual_valid_ratio']}\n")
        f.write(f"- 删除样本数：{summary['dropped_items']}\n\n")
        f.write("## 2. 删除原因\n\n")
        if report["drop_reasons"]:
            for k, v in sorted(report["drop_reasons"].items(), key=lambda x: (-x[1], x[0])):
                f.write(f"- {k}: {v}\n")
        else:
            f.write("- 无\n")
        f.write("\n")
        write_distribution_table(f, "3. 类别分布", report["category_distribution"])
        write_distribution_table(f, "4. rejected_type 分布", report["rejected_type_distribution"])
        write_distribution_table(f, "5. 来源分布", report["source_distribution"])
        write_distribution_table(f, "6. 严重程度分布", report["severity_distribution"])
        f.write("## 7. 数据格式说明\n\n")
        f.write("每条 DPO 样本包含 `prompt`、`chosen`、`rejected`、`rejected_type`、`severity` 和 `preference_reason`。")
        f.write("prompt/chosen 来自清洗后的 SFT 或离线评测集，rejected 由 GLM API 按指定错误类型生成。\n")


def parse_args() -> argparse.Namespace:
    root = get_project_root()
    parser = argparse.ArgumentParser(description="Build DPO preference data with GLM-generated rejected answers.")
    parser.add_argument("--sft-input", type=Path, default=root / "data" / "interim" / "cleaned_sft.jsonl")
    parser.add_argument("--eval-input", type=Path, default=root / "data" / "eval" / "eval_set.jsonl")
    parser.add_argument("--candidate-output", type=Path, default=root / "data" / "interim" / "dpo_candidates.jsonl")
    parser.add_argument("--train-output", type=Path, default=root / "data" / "processed" / "dpo_train.jsonl")
    parser.add_argument("--valid-output", type=Path, default=root / "data" / "processed" / "dpo_valid.jsonl")
    parser.add_argument("--report-md", type=Path, default=root / "docs" / "dpo_data_report.md")
    parser.add_argument("--report-json", type=Path, default=root / "docs" / "dpo_data_report.json")
    parser.add_argument("--model", type=str, default=os.getenv("ZHIPUAI_MODEL", "glm-4-flash-250414"))
    parser.add_argument("--base-url", type=str, default=os.getenv("ZHIPUAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key-env", type=str, default="ZHIPUAI_API_KEY")
    parser.add_argument("--target-total", type=int, default=300)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--sft-source-ratio", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--request-sleep", type=float, default=0.2)
    parser.add_argument("--similarity-threshold", type=float, default=0.82)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--disable-rejected-type-match", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_total <= 0:
        raise ValueError("--target-total must be positive")
    if not (0 < args.valid_ratio < 1):
        raise ValueError("--valid-ratio must be in (0, 1).")
    if not (0 <= args.sft_source_ratio <= 1):
        raise ValueError("--sft-source-ratio must be in [0, 1].")
    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key and not args.dry_run:
        raise EnvironmentError(f"Missing API key. Please set: export {args.api_key_env}='your-api-key'")

    rng = random.Random(args.seed)
    sft_items = read_jsonl(args.sft_input, required=True)
    eval_items = read_jsonl(args.eval_input, required=False)
    sft_sources = extract_sources_from_sft(sft_items)
    eval_sources = extract_sources_from_eval(eval_items)
    n_sft = int(round(args.target_total * args.sft_source_ratio))
    n_eval = args.target_total - n_sft
    if not eval_sources:
        n_sft = args.target_total
        n_eval = 0
    rng.shuffle(sft_sources)
    rng.shuffle(eval_sources)
    selected_sources: List[Dict[str, Any]] = []
    selected_sources.extend(sft_sources[:max(0, n_sft)])
    selected_sources.extend(eval_sources[:max(0, n_eval)])
    if len(selected_sources) < args.target_total:
        all_sources = sft_sources + eval_sources
        if not all_sources:
            raise ValueError("No valid sources found from SFT or eval inputs.")
        while len(selected_sources) < args.target_total:
            selected_sources.append(rng.choice(all_sources))

    if args.overwrite:
        for path in [args.candidate_output, args.train_output, args.valid_output]:
            if path.exists():
                path.unlink()

    plan = build_generation_plan(args.target_total, args.seed)
    paired_plan = sample_sources_for_plan(selected_sources, plan, args.seed)
    existing_candidates = read_jsonl(args.candidate_output, required=False)
    existing_keys = {dpo_key(item) for item in existing_candidates}
    start_index = len(existing_candidates) + 1
    remaining_paired_plan = paired_plan[len(existing_candidates):]

    print(f"[INFO] model: {args.model}")
    print(f"[INFO] sft sources: {len(sft_sources)}")
    print(f"[INFO] eval sources: {len(eval_sources)}")
    print(f"[INFO] selected sources: {len(selected_sources)}")
    print(f"[INFO] target total: {args.target_total}")
    print(f"[INFO] existing candidates: {len(existing_candidates)}")
    print(f"[INFO] remaining items: {len(remaining_paired_plan)}")

    if args.dry_run:
        first = remaining_paired_plan[0]
        messages = build_generation_prompt(first)
        print(json.dumps(messages, ensure_ascii=False, indent=2))
        return

    client = make_client(api_key=api_key, base_url=args.base_url, timeout=120.0)
    drop_reasons: Counter = Counter()
    accepted_count = len(existing_candidates)
    next_id_num = start_index
    for idx, item in enumerate(remaining_paired_plan, 1):
        dpo_id = f"dpo_{next_id_num:06d}"
        print(f"[CALL] {idx}/{len(remaining_paired_plan)} id={dpo_id} category={item['category']} rejected_type={item['rejected_type']}")
        messages = build_generation_prompt(item)
        llm_obj = call_llm(
            client=client,
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            retry=args.retry,
            retry_sleep=args.retry_sleep,
        )
        dpo_item = build_dpo_item(item=item, llm_obj=llm_obj, dpo_id=dpo_id)
        key = dpo_key(dpo_item)
        if key in existing_keys:
            drop_reasons["duplicate_dpo_key"] += 1
            print("  [DROP] reason=duplicate_dpo_key")
            continue
        ok, reason = validate_dpo_item(
            dpo_item,
            similarity_threshold=args.similarity_threshold,
            enforce_rejected_type_match=not args.disable_rejected_type_match,
        )
        if not ok:
            drop_reasons[reason] += 1
            rejected_text = dpo_item.get("rejected", [{}])[0].get("content", "")
            print(f"  [DROP] reason={reason}, rejected={rejected_text[:80]}")
            continue
        append_jsonl(args.candidate_output, [dpo_item])
        existing_keys.add(key)
        accepted_count += 1
        next_id_num += 1
        print(f"[DONE] accepted total={accepted_count}")
        time.sleep(args.request_sleep)

    candidates = read_jsonl(args.candidate_output, required=True)
    train, valid = stratified_split(candidates, valid_ratio=args.valid_ratio, seed=args.seed)
    write_jsonl(args.train_output, train)
    write_jsonl(args.valid_output, valid)
    report = build_report(candidates=candidates, train=train, valid=valid, drop_reasons=drop_reasons, plan=plan)
    write_json(args.report_json, report)
    write_markdown_report(args.report_md, report)
    print("[SUMMARY]")
    print(f"  candidates: {len(candidates)}")
    print(f"  train: {len(train)}")
    print(f"  valid: {len(valid)}")
    print(f"  candidate output: {args.candidate_output}")
    print(f"  train output: {args.train_output}")
    print(f"  valid output: {args.valid_output}")
    print(f"  report md: {args.report_md}")
    print(f"  report json: {args.report_json}")
    if drop_reasons:
        print("  drop reasons:")
        for k, v in drop_reasons.most_common():
            print(f"    - {k}: {v}")


if __name__ == "__main__":
    main()
