#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于 SFT 黄金样本，使用 GLM API 扩写用户问题。

输入：
    data/raw/sft_seed.jsonl

输出：
    data/interim/sft_candidates.jsonl

核心原则：
    1. 只让大模型扩写 user query；
    2. 不让大模型改写 assistant answer；
    3. 每个扩写问题复用原始 seed 的 assistant answer；
    4. 支持断点续跑；
    5. 输出 messages JSONL 格式；
    6. 尽量保证每条 seed 最终保留 variants_per_seed 条有效扩写。

依赖：
    pip install openai

环境变量：
    export ZHIPUAI_API_KEY=""

示例：
    python scripts/expand_sft_with_llm.py --model glm-4.7-flash --variants-per-seed 15
    python scripts/expand_sft_with_llm.py --dry-run
    python scripts/expand_sft_with_llm.py --overwrite
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from openai import OpenAI


DEFAULT_SYSTEM_PROMPT_FOR_EXPANSION = (
    "你是电商客服 SFT 数据构造助手。"
    "你的任务是生成用户问题的语义等价改写，用于训练客服模型。"
    "你只能改写用户问题，不能生成、改写或补充客服回答。"
    "所有改写必须保持原始问题的业务意图、限定条件和问题粒度一致。"
    "不得引入原问题没有的信息、事实、政策、金额、订单状态、物流状态或处理承诺。"
    "输出必须是 JSON 数组，数组元素必须是字符串。"
)

DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


# -----------------------------
# JSONL IO
# -----------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}, line {line_no}: {e}") from e
            items.append(item)

    return items


def append_jsonl(path: Path, items: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


# -----------------------------
# Seed parsing
# -----------------------------

def extract_user_and_assistant(item: Dict[str, Any]) -> Tuple[str, str]:
    user_text: Optional[str] = None
    assistant_text: Optional[str] = None

    for msg in item.get("messages", []):
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            user_text = content
        elif role == "assistant":
            assistant_text = content

    if not user_text or not assistant_text:
        raise ValueError(f"Seed item missing user or assistant message: {item.get('id')}")

    return user_text.strip(), assistant_text.strip()


# -----------------------------
# Text normalization and matching
# -----------------------------

def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    text = text.replace("？", "?").replace("！", "!").replace("。", ".")
    text = text.replace("，", ",").replace("、", ",")
    return text.lower()


def contains_any_pattern(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def has_question_intent(text: str) -> bool:
    question_markers = [
        "吗", "么", "嘛", "？", "?", "怎么", "咋", "如何", "为啥", "为什么",
        "能不能", "可不可以", "可以吗", "行不行", "是不是", "有没有",
        "多久", "多少", "哪里", "在哪", "怎么办", "啥时候"
    ]
    return any(m in text for m in question_markers)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def is_near_duplicate(a: str, b: str, threshold: float) -> bool:
    return similarity(a, b) >= threshold


# -----------------------------
# Information extraction rules
# -----------------------------

PHONE_PATTERNS = [
    r"(?<!\d)1[3-9]\d{9}(?!\d)",
]

ID_CARD_PATTERNS = [
    r"\b\d{17}[\dXx]\b",
    r"\b\d{15}\b",
]

EMAIL_PATTERNS = [
    r"[\w\.-]+@[\w\.-]+\.\w+",
]

MONEY_PATTERNS = [
    r"\d+(?:\.\d+)?\s*(元|块钱|人民币|rmb|RMB)",
    r"(满|减|返|赔|补|优惠)\s*\d+",
]

ORDER_ID_PATTERNS = [
    r"订单号\s*[:：]?\s*[A-Za-z0-9_-]{5,}",
    r"订单\s*[:：]?\s*[A-Za-z0-9_-]{6,}",
    r"单号\s*[:：]?\s*[A-Za-z0-9_-]{6,}",
]

LOGISTICS_STATUS_PATTERNS = [
    r"(已发货|未发货|没发货|没有发货|还没发货|尚未发货)",
    r"(派送中|配送中|运输中|运送中|路上|在路上)",
    r"(已签收|签收了|拒收|丢件|破损|滞留|清关中)",
]

TIME_SPECIFIC_PATTERNS = [
    r"\d+\s*(天|日|小时|分钟|周|个月)",
    r"(今天|明天|昨天|后天|今晚|上午|下午|晚上|本周|下周|这个月|下个月)",
]

RISKY_POLICY_PATTERNS = [
    r"(一定|肯定|保证|百分百|必须).{0,8}(退款|退货|送达|赔偿|赔付|处理|解决)",
    r"(三倍|十倍|假一赔十|全额).{0,6}(赔偿|赔付|退款|退货)",
    r"\d+\s*(天|日)\s*无理由",
    r"(包退|包换|包赔|免赔|无条件退款|无条件退货)",
]


def introduced_pattern(text: str, original_user: str, patterns: Sequence[str]) -> bool:
    """
    只判断“扩写新增的信息”。
    如果原问题本身已经有金额、时间、7天无理由等信息，扩写保留不算问题。
    """
    return contains_any_pattern(text, patterns) and not contains_any_pattern(original_user, patterns)


def has_new_sensitive_or_specific_info(text: str, original_user: str) -> Tuple[bool, str]:
    checks = [
        ("new_phone", PHONE_PATTERNS),
        ("new_id_card", ID_CARD_PATTERNS),
        ("new_email", EMAIL_PATTERNS),
        ("new_money", MONEY_PATTERNS),
        ("new_order_id", ORDER_ID_PATTERNS),
        ("new_logistics_status", LOGISTICS_STATUS_PATTERNS),
        ("new_specific_time", TIME_SPECIFIC_PATTERNS),
    ]

    for reason, patterns in checks:
        if introduced_pattern(text, original_user, patterns):
            return True, reason

    return False, "ok"


# -----------------------------
# Variant validation
# -----------------------------

def looks_like_answer(text: str) -> bool:
    """
    判断候选是否像客服回答。
    注意：用户也可能说“请问”“平台会不会”，所以不能简单用关键词硬杀。
    """
    text = text.strip()

    strong_answer_markers = [
        "感谢您的理解",
        "祝您生活愉快",
        "我们会尽快",
        "我们将尽快",
        "请您提供订单号",
        "请您耐心等待",
        "建议您先",
        "您可以在订单详情页",
        "您可以联系在线客服",
        "很抱歉给您带来不便",
    ]

    if any(marker in text for marker in strong_answer_markers):
        return True

    # 如果有明显疑问意图，即使包含“平台会”“可以”等词，也更可能是用户问题。
    if has_question_intent(text):
        return False

    weak_answer_markers = [
        "您好",
        "建议您",
        "您可以",
        "很抱歉",
        "我们会",
        "平台会",
        "请您",
    ]

    return any(marker in text for marker in weak_answer_markers)


def has_risky_new_commitment(text: str, original_user: str) -> bool:
    return introduced_pattern(text, original_user, RISKY_POLICY_PATTERNS)


def is_valid_variant(text: str, original_user: str) -> Tuple[bool, str]:
    text = text.strip()

    if len(text) < 4:
        return False, "too_short"

    if len(text) > 120:
        return False, "too_long"

    if normalize_text(text) == normalize_text(original_user):
        return False, "same_as_original"

    if looks_like_answer(text):
        return False, "looks_like_answer"

    has_new_info, reason = has_new_sensitive_or_specific_info(text, original_user)
    if has_new_info:
        return False, reason

    if has_risky_new_commitment(text, original_user):
        return False, "new_risky_policy_or_commitment"

    return True, "ok"


# -----------------------------
# LLM output parsing
# -----------------------------

def extract_json_array(raw_text: str) -> List[str]:
    text = raw_text.strip()

    if not text:
        return []

    # 去掉 markdown 代码块包裹
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # 1. 优先按完整 JSON 解析
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [
                str(x).strip(" \t\r\n\"'，,")
                for x in parsed
                if str(x).strip(" \t\r\n\"'，,")
            ]
    except json.JSONDecodeError:
        pass

    # 2. 如果模型输出前后带解释文字，尝试抽取中间的 JSON 数组
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return [
                    str(x).strip(" \t\r\n\"'，,")
                    for x in parsed
                    if str(x).strip(" \t\r\n\"'，,")
                ]
        except json.JSONDecodeError:
            pass

    # 3. 最后兜底：按行解析
    lines: List[str] = []
    for line in text.splitlines():
        line = line.strip()

        # 去掉列表编号、项目符号，例如：
        # 1. xxx
        # 1、xxx
        # - xxx
        # * xxx
        line = re.sub(r"^[\-\*\d\.\)、\s]+", "", line)

        # 去掉 JSON 数组里常见的尾部逗号和引号
        line = line.strip(" \t\r\n\"'，,")

        # 再去一次可能残留的右侧符号
        line = line.rstrip(" \t\r\n\"'，,")

        # 跳过纯结构符号
        if not line or line in {"[", "]", "{", "}", ","}:
            continue

        # 跳过明显不是问题的解释性文字
        if line.startswith(("输出", "以下", "示例", "JSON", "json")):
            continue

        lines.append(line)

    return lines


# -----------------------------
# Prompt
# -----------------------------

def build_expansion_prompt(
    original_user: str,
    assistant_answer: str,
    variants_per_seed: int,
    category: str,
    existing_examples: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    existing_block = ""
    if existing_examples:
        existing_text = "\n".join([f"- {x}" for x in existing_examples[:30]])
        existing_block = f"""
已有候选问题，不能重复或高度相似：
{existing_text}
"""

    user_prompt = f"""请基于下面的原始用户问题，扩写 {variants_per_seed} 个语义等价但表达不同的用户问题。

核心目标：
你只改写“用户怎么问”，不能改变用户真正想解决的问题。

硬性要求：
1. 必须保持与原始用户问题相同的业务意图、适用条件和问题粒度；
2. 不要扩大、缩小或转移问题范围；
3. 不要新增原问题中没有出现的商品名、品牌、金额、时间、地点、订单状态、物流状态、售后状态、赔偿要求、平台政策或处理承诺；
4. 不要删除原问题中的关键限定条件，例如会员身份、订单类型、商品类型、售后阶段、物流阶段等；
5. 不要把泛化问题改成具体订单查询；
6. 不要输出客服答案、处理建议或解释；
7. 不要出现“您好、建议您、您可以、请您、我们会”等客服回答口吻；
8. 只输出 JSON 数组，数组元素必须是字符串；
9. 每个问题长度建议在 4 到 60 个中文字符之间；
10. 各问题之间要尽量避免近似重复，句式、语气、长短应有明显差异。

多样性要求：
请尽量覆盖以下表达风格：
- 简短口语表达
- 完整正式表达
- 焦虑或不满表达
- 追问式表达
- 移动端输入风格
- 轻微省略表达

场景类别：
{category}

原始用户问题：
{original_user}

标准客服答案，仅用于理解意图和边界，不能改写，不能输出：
{assistant_answer}
{existing_block}
输出格式：
[
  "问题1",
  "问题2",
  "问题3"
]
"""
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT_FOR_EXPANSION},
        {"role": "user", "content": user_prompt},
    ]


# -----------------------------
# LLM call
# -----------------------------

def call_llm_for_variants(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    retry: int,
    retry_sleep: float,
) -> List[str]:
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
            variants = extract_json_array(content)

            if variants:
                return variants

            print("[WARN] LLM returned no parsable variants. Raw content:", file=sys.stderr)
            print(content[:1000], file=sys.stderr)

            raise ValueError("LLM returned no parsable variants")
        except Exception as e:
            last_error = e
            sleep_seconds = retry_sleep * attempt
            print(
                f"[WARN] API call failed, attempt={attempt}/{retry}, "
                f"sleep={sleep_seconds:.1f}s, error={repr(e)}",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(f"API call failed after {retry} attempts: {last_error}") from last_error


# -----------------------------
# Rebuild SFT sample
# -----------------------------

def rebuild_sft_item(
    seed_item: Dict[str, Any],
    new_id: str,
    new_user_text: str,
    source: str = "llm_user_rewrite",
) -> Dict[str, Any]:
    new_item = {
        "id": new_id,
        "category": seed_item.get("category", "unknown"),
        "source": source,
        "seed_id": seed_item.get("id"),
        "messages": [],
    }

    for msg in seed_item.get("messages", []):
        if msg.get("role") == "user":
            new_item["messages"].append({
                "role": "user",
                "content": new_user_text,
            })
        else:
            new_item["messages"].append(deepcopy(msg))

    return new_item


# -----------------------------
# Existing output loading
# -----------------------------

def load_existing_state(output_path: Path) -> Tuple[Dict[str, int], Set[str], int]:
    """
    返回：
    1. 每个 seed_id 已经写入多少条；
    2. 已存在的 user 文本归一化集合；
    3. 已存在的最大 sft_llm_ ID 数字。
    """
    seed_counts: Dict[str, int] = defaultdict(int)
    existing_user_texts: Set[str] = set()
    max_numeric_id = 0

    if not output_path.exists():
        return seed_counts, existing_user_texts, max_numeric_id

    id_pattern = re.compile(r"^sft_llm_(\d+)$")

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            seed_id = item.get("seed_id")
            if seed_id:
                seed_counts[str(seed_id)] += 1

            try:
                user_text, _ = extract_user_and_assistant(item)
                existing_user_texts.add(normalize_text(user_text))
            except Exception:
                pass

            item_id = str(item.get("id", ""))
            match = id_pattern.match(item_id)
            if match:
                max_numeric_id = max(max_numeric_id, int(match.group(1)))

    return seed_counts, existing_user_texts, max_numeric_id


def make_client(api_key: str, base_url: str, timeout: float) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )


# -----------------------------
# Args
# -----------------------------

def parse_args() -> argparse.Namespace:
    root = get_project_root()

    parser = argparse.ArgumentParser(
        description="Expand SFT user queries from seed data using GLM API."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "raw" / "sft_seed.jsonl",
        help="Path to seed SFT JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "interim" / "sft_candidates.jsonl",
        help="Path to output expanded SFT candidates JSONL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("ZHIPUAI_MODEL", "glm-4-flash-250414"),
        help="GLM model name. If your console uses another name, override it here.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("ZHIPUAI_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible base URL for ZhipuAI.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="ZHIPU AI API_KEY",
        help="Environment variable name that stores API key.",
    )
    parser.add_argument(
        "--variants-per-seed",
        type=int,
        default=15,
        help="Target number of valid user query variants to keep for each seed item.",
    )
    parser.add_argument(
        "--oversample-factor",
        type=float,
        default=1.5,
        help="Request more raw variants each attempt to compensate for filtering.",
    )
    parser.add_argument(
        "--refill-attempts",
        type=int,
        default=2,
        help="Additional refill attempts when valid variants are fewer than target.",
    )
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=0.88,
        help="Similarity threshold for near-duplicate filtering.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for rewriting.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max output tokens for each API call.",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=3,
        help="Retry times for each API call.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=2.0,
        help="Base sleep seconds between retries.",
    )
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=0.3,
        help="Sleep seconds after each successful request to reduce rate-limit risk.",
    )
    parser.add_argument(
        "--min-request-count",
        type=int,
        default=5,
        help="Minimum number of variants to request in each LLM call.",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=0,
        help="Only process first N seeds. 0 means all seeds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for output shuffle within each seed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file and process all seeds from scratch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the first prompt without calling API.",
    )

    return parser.parse_args()


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.variants_per_seed <= 0:
        raise ValueError("--variants-per-seed must be positive.")

    if args.oversample_factor < 1:
        raise ValueError("--oversample-factor must be >= 1.")

    if not (0 < args.near_duplicate_threshold <= 1):
        raise ValueError("--near-duplicate-threshold must be in (0, 1].")

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key and not args.dry_run:
        raise EnvironmentError(
            f"Missing API key. Please set env var: export {args.api_key_env}='ZHIPUAI_API_KEY'"
        )

    seed_items = read_jsonl(args.input)
    if args.max_seeds and args.max_seeds > 0:
        seed_items = seed_items[:args.max_seeds]

    print(f"[INFO] Input seed file: {args.input}")
    print(f"[INFO] Output file: {args.output}")
    print(f"[INFO] Seed items to consider: {len(seed_items)}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Base URL: {args.base_url}")
    print(f"[INFO] Target variants per seed: {args.variants_per_seed}")
    print(f"[INFO] Refill attempts: {args.refill_attempts}")
    print(f"[INFO] Near-duplicate threshold: {args.near_duplicate_threshold}")

    if args.overwrite and args.output.exists():
        args.output.unlink()
        print(f"[INFO] Removed existing output file: {args.output}")

    seed_counts, existing_user_texts, max_numeric_id = load_existing_state(args.output)

    # 把原始 seed 的 user 问题也加入去重池，避免生成和 seed 一样的问题。
    for seed_item in seed_items:
        try:
            user_text, _ = extract_user_and_assistant(seed_item)
            existing_user_texts.add(normalize_text(user_text))
        except Exception:
            continue

    if args.dry_run:
        if not seed_items:
            raise ValueError("No seed items found for dry run.")

        first = seed_items[0]
        original_user, assistant_answer = extract_user_and_assistant(first)
        still_needed = args.variants_per_seed
        request_count = max(
            args.min_request_count,
            still_needed,
            int(still_needed * args.oversample_factor),
        )

        prompt_messages = build_expansion_prompt(
            original_user=original_user,
            assistant_answer=assistant_answer,
            variants_per_seed=request_count,
            category=first.get("category", "unknown"),
        )
        print("\n[DRY RUN] Prompt messages:")
        print(json.dumps(prompt_messages, ensure_ascii=False, indent=2))
        return

    client = make_client(
        api_key=api_key,
        base_url=args.base_url,
        timeout=120.0,
    )

    total_written = 0
    total_skipped = 0
    total_invalid = 0
    drop_reasons: Counter[str] = Counter()

    for seed_index, seed_item in enumerate(seed_items, 1):
        seed_id = str(seed_item.get("id", f"seed_{seed_index:04d}"))
        already_written_for_seed = seed_counts.get(seed_id, 0)

        if already_written_for_seed >= args.variants_per_seed:
            total_skipped += 1
            print(
                f"[SKIP] {seed_index}/{len(seed_items)} seed_id={seed_id} "
                f"already has {already_written_for_seed}/{args.variants_per_seed}"
            )
            continue

        category = seed_item.get("category", "unknown")
        original_user, assistant_answer = extract_user_and_assistant(seed_item)

        target_remaining = args.variants_per_seed - already_written_for_seed
        new_items: List[Dict[str, Any]] = []
        local_texts: List[str] = []
        raw_total_for_seed = 0

        print(
            f"[CALL] {seed_index}/{len(seed_items)} seed_id={seed_id} "
            f"category={category}, need={target_remaining}"
        )

        max_attempts_for_seed = 1 + max(0, args.refill_attempts)

        for attempt in range(1, max_attempts_for_seed + 1):
            still_needed = target_remaining - len(new_items)
            if still_needed <= 0:
                break

            request_count = max(
                args.min_request_count,
                still_needed,
                int(still_needed * args.oversample_factor),
            )

            prompt_messages = build_expansion_prompt(
                original_user=original_user,
                assistant_answer=assistant_answer,
                variants_per_seed=request_count,
                category=category,
                existing_examples=local_texts,
            )

            print(
                f"  [EXPAND] attempt={attempt}/{max_attempts_for_seed}, "
                f"request={request_count}, still_needed={still_needed}"
            )

            raw_variants = call_llm_for_variants(
                client=client,
                model=args.model,
                messages=prompt_messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                retry=args.retry,
                retry_sleep=args.retry_sleep,
            )

            raw_total_for_seed += len(raw_variants)
            random.shuffle(raw_variants)

            for variant in raw_variants:
                if len(new_items) >= target_remaining:
                    break

                variant = variant.strip()
                if not variant:
                    continue

                ok, reason = is_valid_variant(variant, original_user)
                norm = normalize_text(variant)

                if not ok:
                    total_invalid += 1
                    drop_reasons[reason] += 1
                    print(f"    [DROP] reason={reason}, text={variant[:60]}")
                    continue

                if norm in existing_user_texts:
                    total_invalid += 1
                    drop_reasons["duplicate_existing"] += 1
                    print(f"    [DROP] reason=duplicate_existing, text={variant[:60]}")
                    continue

                if any(is_near_duplicate(variant, x, args.near_duplicate_threshold) for x in local_texts):
                    total_invalid += 1
                    drop_reasons["near_duplicate_local"] += 1
                    print(f"    [DROP] reason=near_duplicate_local, text={variant[:60]}")
                    continue

                # 这里不对全量 existing_user_texts 做近重复，因为历史数据可能很大，逐条比对会很慢。
                # 如果你后续数据量不大，可以再加一层全局近重复检测。

                local_texts.append(variant)
                existing_user_texts.add(norm)

                max_numeric_id += 1
                new_id = f"sft_llm_{max_numeric_id:06d}"

                new_items.append(
                    rebuild_sft_item(
                        seed_item=seed_item,
                        new_id=new_id,
                        new_user_text=variant,
                    )
                )

            time.sleep(args.request_sleep)

        append_jsonl(args.output, new_items)
        total_written += len(new_items)
        seed_counts[seed_id] = already_written_for_seed + len(new_items)

        print(
            f"[DONE] seed_id={seed_id}, raw={raw_total_for_seed}, "
            f"written_now={len(new_items)}, "
            f"seed_total={seed_counts[seed_id]}/{args.variants_per_seed}, "
            f"total_written={total_written}"
        )

    print("\n[SUMMARY]")
    print(f"  seeds skipped as complete: {total_skipped}")
    print(f"  generated samples written this run: {total_written}")
    print(f"  invalid or duplicate variants dropped: {total_invalid}")
    print(f"  output: {args.output}")

    if drop_reasons:
        print("\n[DROP REASONS]")
        for reason, count in drop_reasons.most_common():
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()